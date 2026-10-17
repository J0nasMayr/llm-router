# Step 3 revisited: from traffic shaping to model placement

## 1. Verdict

Yes, this makes sense for the thesis, under three conditions:

1. The supervisor agrees that model placement (which models are loaded on which node, i.e. replica scaling per model) counts as "real allocation".
2. You have GPU access: at least one cloud-class GPU (40–80 GB) and two edge-class nodes (real, MIG slices, or memory-capped processes).
3. About four weeks are left before the deadline.

Even if the answer to 1–3 is no, do Phase 0 (Section 7). It fixes a blind spot that would distort every experiment.

## 2. Why the current system needs this

Findings in the code:

1. **One model per worker.** The worker holds exactly one model (`inference_service.py:71`). A task for another model unloads it, loads the new one from disk and runs a 20-token warm-up (`:91–119`).
2. **Load time is invisible.** Latency is measured only around `generate` (`:128–141`). Telemetry has no list of loaded models and no load times (`:237–248`). So T̂ (`orchestrator.py:101–110`) ignores the most expensive part of a model change.
3. **Dispatch ignores what is loaded.** The node with the lowest T̂ wins (`orchestrator.py:133–185`), whether the model is loaded there or not.
4. **The LP makes it worse.** It knows nothing about memory. In a test with 13 edge models and 2 nodes, it sent the traffic of 5 models to node A and of 8 models to node B. Each node holds one model at a time, so every change of model on a node is a full reload.
5. **No node can hold its tier.** Edge models have about 45 B parameters in total (about 90 GB in bf16), cloud models about 134 B (about 270 GB). A choice of which models stay loaded is unavoidable.
6. **GreenServ never met this problem.** Its experiments replay precomputed results from a database (16 models × 2,500 queries), and the paper assumes that "all model weights are locally available". This system runs live, so the problem is new.

Load times range from seconds (models up to about 3 B) to roughly a minute (27–34 B). They must be measured on the testbed. With T_max = 30 s, a single cold start of a large model can break the deadline on its own.

## 3. What "real allocation" becomes

- **Resource:** the GPU memory of each node.
- **Decision:** which models are loaded on which node, i.e. the replica set of each model.
- **Actions:** add a replica (scale out), remove a replica (scale in), unload a model completely (scale to zero), and split traffic among replicas (kept from today).

This is the same kind of decision as ENOVA (replicas per GPU type) and SageServe (model placement), and the same as HAS-GPU's horizontal scaling. The original step-3 notes (`step3_paper_choice.md`) already planned "replicas + weights"; only the weights were built. This plan completes the original design.

## 4. Design

### 4.1 Worker (the actuator)

- **Several resident models:** a dict of loaded models instead of one slot.
- **Memory budget per node:** new flag `--memory-budget-gb`. Default: total GPU memory minus headroom for the KV cache. The flag also allows emulating small edge GPUs on a large GPU (`torch.cuda.set_per_process_memory_fraction`).
- **Desired state:** the controller writes the target set of models to the Redis key `placement:node:<id>`. Between tasks, the worker compares it with what it holds and fixes the difference: first unload, then load, one model per step. A desired state is safer than commands: no message can get lost, and applying it twice does no harm.
- **On-demand fallback stays:** a task for a model that is not loaded still runs. The worker loads it and evicts the least recently used model outside the target set. The result is marked `cold_start = true`, with its `load_time`.
- **Measurement:** the worker measures the load time of each model and reports it.

### 4.2 Telemetry

New fields: `resident_models`, `load_time_by_model`, `loading` (model and start time), `memory_budget_gb`.

### 4.3 Feasibility filter (Stage 1)

```
T̂(m,n) = δ_tier + (W_n + 1)·τ̄(m,n) + L(m,n)·[m not loaded on n] + remaining load time on n
```

- Before the first measurement, L(m,n) is estimated from the model size.
- **Effect:** small models that are not loaded can still be used when the deadline allows; large ones cannot. No special case is needed.
- **Memory rule:** today, a node with less than 0.5 GB free is removed. With several models loaded on purpose, nodes will be full, so the rule changes to: "the model fits into the node's budget, after unloading models outside the target set".

### 4.4 Bandit: shadow demand

For each query, record the model the bandit *would* choose if every model were available: the argmax of the UCB score without the mask, with random tie-breaking. LinUCB already computes all scores before applying the mask (`linucb.py:65–69`), so this takes a few lines. The controller forecasts this shadow demand instead of the realized demand.

**Why:** with realized demand, a model that is not loaded is never chosen, so its demand stays at zero and it is never loaded (lock-in). Shadow demand breaks this loop. It also answers the gap stated in related work: allocation systems take the traffic per model as given; here, the allocation follows what the router wants.

### 4.5 Controller: placement MILP

The LP becomes a small mixed-integer program. The solver stays the same: `scipy.optimize.milp` with HiGHS, available in the venv's scipy 1.11.

**Variables**

- `y[m,n] ∈ {0,1}`: model m is loaded on node n
- `z[m] ∈ {0,1}`: model m is served by at least one loaded replica
- `x[m,n] ∈ [0,1]`: share of m's traffic sent to n (as today)

**Inputs**

- `d[m]`: predicted shadow demand (req/s)
- `τ̄[m,n]`: average inference time
- `L[m,n]`: load time; `L[m]` is its average over the tier's nodes
- `s[m]`: memory need, about 2 GB per billion parameters plus a margin (from `parameter_count` in `models.yaml`)
- `S[n]`: memory budget of node n
- `ρ`: target node load, e.g. 0.85 (= θ_up)
- `y0`: current placement
- `H`: planning horizon, e.g. 300 s

**Objective** (every term in node-seconds per second)

```
min  Σ d[m]·τ̄[m,n]·x[m,n]                  inference time
   + Σ d[m]·L[m]·(1 − z[m])                   cold starts for models without a loaded replica
   + Σ (L[m,n]/H)·y[m,n]·(1 − y0[m,n])        new loads, spread over the horizon H
```

**Constraints**

```
Σ_n x[m,n] = z[m]                    ∀m     (served models split all their traffic)
x[m,n] ≤ y[m,n]                      ∀m,n   (traffic only to loaded replicas)
Σ_m s[m]·y[m,n] ≤ S[n]               ∀n     (memory budget)
Σ_m d[m]·τ̄[m,n]·x[m,n] ≤ ρ           ∀n     (no overload)
y[m,n] = 0 if tier(m) ≠ tier(n)
```

**Properties**

- **Always feasible:** "load nothing" is a valid solution, so the equal-weight fallback is no longer needed.
- **Small:** 19 models × 4 nodes is about 170 variables, solved in milliseconds.
- **Sensible priorities:** it loads popular large models first, since they are the most expensive to cold-start, and leaves tiny models to on-demand loading.
- **Scaling:** it adds replicas when one node cannot carry a model's demand.
- **Stability:** the load-cost term prevents churn. A tiny tie-break term keeps the current placement when nothing is gained.

### 4.6 Trigger (reused)

- Kalman filter per model, now on shadow demand.
- A rule firing only means "re-plan". The MILP decides whether anything changes, so a false alarm costs nothing.
- **Changes:**
  - "scale-up" also covers a model without a replica whose demand exceeds R_min;
  - "scale-down" now means removing a replica;
  - "memory pressure" is redefined as free memory below the KV headroom (no longer 0.5 GB);
  - the cooldown is at least the longest load time (60–120 s).

### 4.7 Startup

The first placement is the static plan: the MILP solved once with demand taken from GreenServ's offline data (how often each model is the best choice at λ = 0.5). The same plan serves as the "static" baseline.

## 5. Evaluation

### 5.1 Policies (RQ3)

| Policy | Description |
|---|---|
| B1 single-slot | Today's worker (GreenServ's): one model, load on demand |
| B2 LRU cache | Several models per node, load on demand, evict least recently used, no controller. The strongest simple alternative |
| B3 static | Placement computed once at startup, never changed |
| P dynamic | The proposed controller |
| Ablation | P with realized demand instead of shadow demand (shows the lock-in) |

### 5.2 Workload

- **Phases with different dataset mixes:**
  - A: MMLU, HellaSwag, WinoGrande (small models are enough);
  - B: GSM8K, CNN/DM (larger models preferred);
  - C: mixed.
- **Arrivals:** Poisson, at 2–3 load levels.
- **Duration:** at least 10 minutes per phase, so that load times are amortized.

### 5.3 Metrics

Deadline violations, p95 latency, cold starts per 100 requests, share of time spent loading, number of placement changes, accuracy, and energy (including loading energy, if measured).

### 5.4 Expected outcome

- **Against B1:** a large gain.
- **Against B3:** a clear gain after phase changes.
- **Against B2:** a smaller gain, mainly at phase changes, for large models and under memory pressure. A small gain over LRU would still be a valid result, but the workload should be designed so that coordination between nodes matters.

### 5.5 Live or simulated

- **Live:** needs GPUs (see Section 10).
- **Simulation:** a trace-driven simulator could reuse GreenServ's database (accuracy, energy and latency per query and model) together with measured load times. It would run the real router and controller code with a simulated clock: many seeds and load levels at no GPU cost, and the same method GreenServ used.
- **Cost of the simulator:** about one extra week. Decide after Phase 2.

## 6. Thesis changes

- **RQ3 (new):** "Does adapting which models each node keeps loaded to the router's demand reduce deadline violations and cold starts, compared to on-demand loading, an LRU cache and a static placement?"
- **§3.3:** rewritten as the placement MILP plus the actuator. The traffic split becomes part of it. The trigger subsection mostly stays.
- **§3.2, Stage 1:** load time in T̂, and the new memory rule.
- **Introduction:**
  - the problem statement adds "which models to keep loaded";
  - the approach paragraph and contribution 3 are updated.
- **Related work §2.3:**
  - ENOVA, SageServe and HAS-GPU now match directly;
  - consider adding systems that handle model loading, e.g. ServerlessLLM (OSDI '24), AlpaServe (OSDI '23) and Clockwork (OSDI '20). Verify them before citing.
- **Figure:** a new arrow from the controller to the nodes (target placement); telemetry adds the loaded models and their load times.
- **Story:** decisions on two timescales, placement every few minutes and filter plus bandit per query, like DynamoLLM's hierarchy of controllers.

## 7. Phases and effort

| Phase | Content | Effort |
|---|---|---|
| P0 (do in any case) | Telemetry reports loaded models and load times; T̂ includes the load time; dispatch therefore prefers nodes that hold the model | 1 day |
| P1 | Multi-model worker, memory budget, desired-state reconcile, LRU fallback, cold-start metrics | 2–3 days |
| P2 | Shadow demand, MILP, trigger changes, policy switch (single / LRU / static / dynamic), unit tests on small MILP instances | 3 days |
| P3 | Load generator with phases (needed for RQ1–RQ2 in any case) | 2 days |
| P4 | Experiments | 3–5 days live, plus 4–6 days for a simulator |
| P5 | Thesis text and figure | 4–6 days |

- **Total:** about 3–4 weeks. P3 and part of P5 are needed in any case.
- **Stopping points:** after P0, the system is consistent; after P2, the design is complete and testable.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Testbed unknown | The memory budget makes the design hardware-agnostic; edge nodes can be emulated with memory caps or MIG |
| Out of memory (KV cache, eager attention on long inputs) | Headroom per node; inputs are already truncated at 8,192 tokens |
| Loading blocks a node | The worker loads between tasks; T̂ includes the remaining load time, so the filter avoids the node meanwhile |
| Churn or oscillation together with the bandit | Load-cost term, cooldown at least as long as a load, Kalman smoothing |
| Exploration loads models only to try them | Load-cost term and R_min limit this; demand based on the mean reward (no exploration bonus) is an ablation |
| Running out of time | Freeze the scope after P2; build the simulator only if needed |

## 9. Alternatives considered

| Alternative | Why not |
|---|---|
| Keep traffic shaping and reframe it | Cheapest, but it probably fails the requirement, and the reload problem remains (P0 is needed anyway) |
| One pinned model per node | Simpler, and still replica scaling. But with 3–4 nodes, the bandit could only use 3–4 of 19 models, so routing loses its point |
| Several worker processes per GPU | True parallel replicas, but needs process management, adds CUDA overhead and makes latencies noisy |
| GPU partitioning (MIG/MPS, as in HAS-GPU) | Hardware-specific; MIG cannot change while processes are running |
| Batch and concurrency settings (ENOVA's configuration) | Requires switching from HF `generate` to vLLM |
| Quantized model variants as vertical scaling | Interesting future work, but it changes the bandit's arms |

## 10. Decisions

Decided:

- **Hardware:** each node has its own GPU, so the worker needs a memory budget only, not a memory cap per process.
- **Model pool:** GreenServ's 16 models. Not applied yet: `models.yaml` still lists 19, and the thesis says 13 edge and 6 cloud models (it becomes 11 and 5).
- **Scope:** P0 only for now.

Still open:

1. **Supervisor:** does model placement (replica scaling per model) count as "real allocation"?
2. **Deadline:** how many weeks are left?
3. **Optional:** allow edge models on cloud nodes, i.e. offloading from edge to cloud? It is one change in the MILP and strengthens the edge-cloud story, but adds evaluation work.

## 11. P0 status (done, not committed)

- **Worker** (`inference_service.py`):
  - times every model load (unload, load and warm-up);
  - reports `resident_models`, `loading` (model and elapsed seconds) and `load_time_by_model` (average of the last 5 loads) in its telemetry;
  - adds `load_time` to each task's result.
  - Also fixed: after a failed load, the worker no longer reports the old model as loaded.
- **Filter** (`orchestrator.py`): T̂ adds the pending load time, i.e. the rest of a running load plus the load of the model itself if the node does not hold it. Before a node's first load of a model, the load time is estimated from the model's bf16 size and `feasibility.load_bandwidth_gb_per_s` (2 GB/s). Workers that do not report their models keep the old formula.
- **Tests:** `tests/test_load_time.py`, 10 tests with a fake model loader.
- **Known limit:** queued tasks are assumed to need no further model change. With one model per worker, T̂ can therefore be optimistic when tasks for different models wait in the same queue. P1 removes most of these changes.
- **Thesis:** updated to match P0:
  - §3.1: T̂ with the load term L(m,n), and ℓ(m,n) defined with its estimate and the queue limitation;
  - §3.2: Stage 1 telemetry and the fallback for new nodes, the Stage 3 dispatch effect, and Stage 4's measurement of load time;
  - parameter table: B and the load-time window;
  - figure: telemetry box;
  - evaluation: metric (vi);
  - related work: the telemetry list.
