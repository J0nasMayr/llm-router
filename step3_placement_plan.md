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
- `T_plan`: planning horizon, e.g. 300 s (not `H`, which the thesis uses for the memory headroom)

**Objective** (every term in node-seconds per second)

```
min  Σ d[m]·τ̄[m,n]·x[m,n]                  inference time
   + Σ d[m]·L[m]·(1 − z[m])                   cold starts for models without a loaded replica
   + Σ (L[m,n]/T_plan)·y[m,n]·(1 − y0[m,n])   new loads, spread over the horizon T_plan
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
| P1 | Multi-model worker, memory budget, LRU fallback, cold-start metrics (done; the desired-state reconcile moved to P2) | 2–3 days |
| P2 | Shadow demand, MILP, desired-state reconcile in the worker, trigger changes, policy switch (single / LRU / static / dynamic), unit tests on small MILP instances | 3–4 days |
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
- **Scope:** P0, then P1. P2 and later wait for the supervisor.

Still open:

1. **Supervisor:** does model placement (replica scaling per model) count as "real allocation"?
2. **Deadline:** how many weeks are left?
3. **Optional:** allow edge models on cloud nodes, i.e. offloading from edge to cloud? It is one change in the MILP and strengthens the edge-cloud story, but adds evaluation work.

## 11. P0 status (done, commit c95a989)

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

## 12. P1 status (done, not committed)

- **Worker** (`inference_service.py`):
  - keeps several models in GPU memory and unloads the least recently used one when a new model does not fit;
  - its memory budget is `--memory-budget-gb`, by default the GPU's memory minus `--memory-headroom-gb` (4 GB, for activations and the KV cache);
  - `--max-models 1` reproduces GreenServ's one-model worker (baseline B1);
  - measures each model's memory after loading;
  - reports `model_memory_gb` and `memory_budget_gb` in its telemetry.
- **Shared helper** (`model_memory.py`): bf16 size estimate (2 GB per billion parameters), the size on a node (measured or estimated), and the fit check.
- **Filter** (`orchestrator.py`): skips nodes whose budget cannot hold the model. The pending load time now counts no own load when the node holds the model while it is loading another one.
- **Controller** (`allocation_controller.py`): applies the same fit check, so the LP gives a model no traffic share on a node that cannot hold it. Without it, the LP could put a model's whole share on such a node, and the filter would then mask the model completely.
- **Tests:** 20 in total. `tests/test_model_residency.py` covers the budget, LRU order, the single-model mode, oversized models, the release of unloaded models before the next load, and the fit checks. `tests/worker_fixture.py` holds the shared fake worker.
- **Unchanged:** the node-level rule "less than G_min free → drop the node" stays as a safety net. With the 4 GB headroom, a full node still has about 4 GB free.
- **Thesis:** updated to match P1:
  - §3.1: memory budget S_n and headroom H, LRU unloading, the load-term reason, the queue limitation, the fit condition s_m ≤ S_n in M_t*;
  - §3.2: Stage 1 telemetry and fit check, Stage 4 unloading and memory measurement;
  - §3.3: who loads and unloads, and the ineligible pairs;
  - parameter table: H and the maximum number of models;
  - figure: filter and telemetry boxes;
  - evaluation: testbed sentence;
  - related work: last sentence of §2.3.

## 13. Coherence pass after P1 (not committed)

- **Thesis:**
  - unloading wording unified ("the models it has used least recently until the new one fits");
  - telemetry list in §2.3;
  - §2.2 and §2.4 now mention loading and memory;
  - introduction: motivation sentence on load time, filter inputs, contributions 1 and 2;
  - evaluation: metric (iii) without loading energy, all variants on the same multi-model workers, comments updated.
- **Code:**
  - `process_query` returns the worker's measurements (incl. `load_time`) with every response, plus the reward metrics when a reference is given;
  - a failed task now returns the worker's error message instead of `'response'`;
  - tests: `tests/test_orchestrator_response.py` (23 tests in total).
- **Still open (fixed by P2 by design, or needs a small fix if P2 is dropped):**
  - models whose estimated load time alone exceeds T_max (Yi-34B, Aya-32B) can only run on a node that already holds them;
  - the controller's LP and the weighted dispatch ignore which models a node holds.


## 14. P2 status (done, not committed)

- **Model pool:** `config/models.yaml` holds GreenServ's 16 models: 11 edge, 5 cloud. The thesis numbers follow in the thesis pass.
- **Shadow demand:**
  - LinUCB records the model it would choose without the mask (`last_preferred_model`, random tie-breaking);
  - the orchestrator counts it next to the realized demand (`consume_shadow_window`, `consume_rps_window`).
- **Placement program** (`solve_placement`, `scipy.optimize.milp`):
  - variables y[m,n] (n keeps m loaded) and x[m,n] (share of m's requests n serves);
  - a model may be served partly: the rest finds no replica and costs a load plus the inference. This replaces the planned z[m], keeps very popular models placed when they overload their nodes, and keeps the program always feasible;
  - load costs are spread over T_plan (300 s), so a model is worth loading once about one request for it is expected within T_plan;
  - the current placement is the last plan; a tiny bonus for it avoids churn between equally good plans.
- **Controller:**
  - policies `dynamic` (first plan from the prior, then re-plan on triggers) and `static` (first plan only);
  - new trigger rule "place": a model without replica whose demand exceeds R_min;
  - publishes each node's target to Redis (`placement:node:<id>`, highest demand first).
- **Routing:**
  - a model with replicas goes to its replicas, by weight;
  - if no replica meets the deadline, other nodes serve as overflow and load it on demand;
  - models without replica are not restricted.
- **Worker:**
  - follows its target placement, one missing target model per step, and only while its queue is empty, so loading for the plan never delays a waiting query;
  - for a task, it unloads models outside the target first and target models only as a last resort; for the plan, it never unloads other target models;
  - reports `target_models`.
- **Prior:** `scripts/placement_prior.py` replays GreenServ's 2,500 profiled queries, computes each model's reward like the live system (α = β = 0.5), and writes the share of queries each model wins to `config/placement_prior.yaml`. Llama-3.2-1B wins 34%; the 5 cloud models win 3% together.
- **Configuration** (`config/allocation.yaml`):

  | Variant | Setting |
  |---|---|
  | B1, one model per node | `policy: none` and workers with `--max-models 1` |
  | B2, LRU cache | `policy: none` |
  | B3, fixed placement | `policy: static` |
  | P, proposed | `policy: dynamic` |
  | Ablation | `demand: realized` |

- **Tests:**
  - 47 in total;
  - `tests/test_placement.py` covers the program, the controller, the worker's target following, routing, shadow demand and the prior;
  - an end-to-end test runs the real worker, controller and orchestrator against a shared in-memory store.
- **Example first plan** (2 edge nodes with 20 GB budget, 1 cloud node with 76 GB, prior at 1 req/s):

  | Node | Models | Memory |
  |---|---|---|
  | edge-2 | Llama-3.2-1B, Qwen-0.5B, Qwen-1.5B, Llama-3.2-3B, Phi-4-mini | 20 GB |
  | edge-1 | Mistral-7B, Qwen-3B | 20 GB |
  | cloud-1 | Gemma-12B, Phi-4 | 54 GB |

  Yi-34B and Gemma-27B are not placed (less than one expected request per 300 s).
- **Known limits:**
  - an on-demand load of a model without replica can unload a target model on a full node; the node restores it when its queue is empty;
  - the static policy gives no target to nodes that join after its plan;
  - load times are estimates until a node has loaded the model once.
- **Thesis (done, not committed):**
  - §3.3 rewritten as "Demand-Driven Model Placement": shadow demand and trigger with the new placement rule, the placement program, following the plan;
  - §3.1 and §3.2 updated for target placements, shadow choices and traffic shares;
  - parameter table: T_plan, prior rate and target read interval added, d_min removed;
  - figure: controller writes target placements to Redis; 11 + 5 models;
  - introduction: problem statement, approach, RQ3 reworded, contributions 2 and 3;
  - related work: §2.3, §2.4 and the summary table;
  - evaluation: RQ3 variants (e) on-demand loading, (f) LRU cache, (g) fixed placement, plus the ablation.
- **Next:** P3 (load generator).

## 15. Fixes after the P2 coherence review (not committed)

- **Placement rule:**
  - fires for a model without replica once its demand exceeds 1/T_plan, the placement program's break-even (about 0.003 req/s), instead of R_min (0.1 req/s);
  - before, the dynamic policy could stop re-planning at low load and act like the fixed placement;
  - R_min remains only the keep-alive floor of the scale-down rule.
- **Plan expiry:**
  - target placements are written with a 16 s expiry, and the controller renews them every tick (also under the static policy);
  - when a target expires, the node drops its protection and falls back to the LRU cache;
  - a plan therefore never outlives its controller, and baseline runs (`policy: none`) cannot follow an old plan.
- **Tests:** 50 in total.
- **Thesis:**
  - placement rule and its threshold explained;
  - expiry added to "Following the plan";
  - parameter table: R_min description, plus the 16 s expiry.

## 16. Goals aligned with the official topic (not committed)

- **Topic:** "Dynamic LLM Scheduling in the Edge–Cloud Continuum: Prompt-Aware & Resource-Aware" names three goals: higher accuracy, lower latency and resource utilization. The thesis now states these three goals:
  - in the introduction's problem statement;
  - in the opening of Chapter 3;
  - in §3.1.
- **Split of the goals:**
  - the router trades accuracy against energy per query, within a latency deadline;
  - the placement controller serves the router's demand with as little node time as possible, for lower latency and better use of GPU memory and time (new paragraph in §3.3);
  - the evaluation's opening maps RQ1–RQ3 to the goals.
- **Not covered from the topic page** (to name as future work):
  - prompt length and sensitivity as inputs;
  - CPU load and measured bandwidth;
  - precision levels (quantized variants);
  - edge–cloud collaboration;
  - execution location as a decision of its own (offloading).
