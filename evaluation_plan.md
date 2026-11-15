# Evaluation plan

The design of Chapter 3 is implemented, and the thesis in `/tex` is authoritative for it.
This file holds what is still open for the evaluation (Chapter 4).
It replaces `step3_placement_plan.md`; the design notes and the status log of that file are in its git history.

## 1. Variants for RQ3

| Variant | Description | Setting |
|---|---|---|
| (e) on-demand loading | One model per node, loaded on demand, as in GreenServ's worker | `policy: none` and workers with `--max-models 1` |
| (f) LRU cache | Several models per node, loaded on demand, least recently used unloaded first, no controller. The strongest simple alternative | `policy: none` |
| (g) fixed placement | The controller's first plan from the prior, never changed | `policy: static` |
| Full system | The placement controller | `policy: dynamic` |
| Ablation | The full system with realized instead of shadow demand (shows the lock-in) | `demand: realized` |

The settings are in `config/placement.yaml`. The baselines (a)–(d) for RQ1 and RQ2 are described in `tex/sources/evaluation.tex`.

## 2. Workload

- **Phases with different dataset mixes:**
  - A: MMLU, HellaSwag, WinoGrande (small models are enough);
  - B: GSM8K, CNN/DM (larger models preferred);
  - C: mixed.
- **Arrivals:** Poisson, at 2–3 load levels.
- **Duration:** at least 10 minutes per phase, so that load times are amortized.

## 3. Metrics

Deadline violations, p95 latency, model loads per 100 queries, share of time spent loading, number of placement changes, accuracy and energy. Energy covers the generation only; the energy of model loads is not measured. Energy is measured with Zeus, as in GreenServ's paper; the setup section of Chapter 4 should say so.

## 4. Expected outcome for RQ3

- **Against (e):** a large gain.
- **Against (g):** a clear gain after phase changes.
- **Against (f):** a smaller gain, mainly at phase changes, for large models and under memory pressure. A small gain over LRU would still be a valid result, but the workload should be designed so that coordination between nodes matters.

## 5. Live or simulated

- **Live:** needs GPUs. Each node has its own GPU; a smaller edge GPU can be emulated with `--memory-budget-gb`.
- **Simulation:** a trace-driven simulator could reuse GreenServ's database (accuracy, energy and latency per query and model) together with measured load times. It would run the real router and controller code with a simulated clock: many seeds and load levels at no GPU cost, and the same method GreenServ used.
- **Cost of the simulator:** about one extra week. Not decided yet.

## 6. Remaining work

| Step | Content | Effort |
|---|---|---|
| Load generator | Phases and Poisson arrivals; needed for RQ1–RQ3 | 2 days |
| Experiments | The runs of Sections 1–3 | 3–5 days live, plus 4–6 days for a simulator |
| Thesis text | Results in Chapter 4, conclusion, abstract | open |

Before the runs, the code still needs:

- a switch to disable the feasibility filter, for baseline (d) and the ablation of RQ1 (TODO in `evaluation.tex`);
- the reward weights α and β in the config, for the sensitivity sweep of RQ2 (TODO in `evaluation.tex`);
- a report of the loads a node does for its target placement. Today, only on-demand loads appear in the results (`load_time`), so the full system would seem to load less than (e) and (f);
- a decision on the result timeout. The scheduler gives up after T_max + 30 s = 60 s; slower queries return no latency, energy or accuracy and do not update the bandit. The thesis does not mention it;
- counters for triggers, plans and placement changes. Today, they are only log lines;
- a check of every testbed node: install `requirements.txt` (Zeus 0.11) and start one worker as the user that will run the experiments. It exits at startup if Zeus cannot measure its GPU (Zeus 0.10 needed root on Intel CPUs, for the CPU's RAPL counters; 0.11 does not).

## 7. Known limits

- An on-demand load of a model without replica can unload a target model on a full node; the node restores it when its queue is empty.
- The static policy gives no target to nodes that join after its plan.
- Load times are estimates until a node has loaded the model once.
- The estimated load time of Yi-34B (34 s) alone exceeds T_max (30 s). Until a node has measured a shorter load, the filter admits the model only on a node that already holds it.
- Queued tasks are assumed to need no further model load, so T̂ can be too optimistic.
- A worker's first query pays a one-time 0.3–1.3 s for setting up Zeus (`ZeusMeasurement` creates a new monitor per task; later queries pay 3–4 ms). It is not in τ̄, but it is in the end-to-end latency; one warm-up query per worker before measuring avoids it.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Testbed unknown | The memory budget makes the design hardware-agnostic; edge nodes can be emulated with a smaller budget or MIG |
| Out of memory (KV cache, eager attention on long inputs) | 4 GB headroom per node; inputs are truncated at 8,192 tokens |
| Loading blocks a node | The worker loads for the plan only while its queue is empty; T̂ includes the remaining load time, so the filter avoids the node meanwhile |
| Churn or oscillation together with the bandit | Load-cost term, 30 s cooldown, Kalman smoothing. The cooldown is shorter than the estimated load of the largest model (34 s), other than first planned |
| Exploration loads models only to try them | Only the load-cost term. The placement rule fires at 1/T_plan, and a single shadow choice lifts the predicted rate above it. Planning with the mean reward (no exploration bonus) would be an ablation; it is not implemented |
| Running out of time | Build the simulator only if needed |

## 9. Open questions

1. **Supervisor:** does model placement (replica scaling per model) count as "real allocation"?
2. **Deadline:** how many weeks are left?
3. **Optional:** allow edge models on cloud nodes, i.e. offloading from edge to cloud? It is one change in the placement program and strengthens the edge-cloud story, but adds evaluation work.

## 10. Future work

The official topic, "Dynamic LLM Scheduling in the Edge–Cloud Continuum: Prompt-Aware & Resource-Aware", names more than the thesis covers. To name as future work:

- prompt length and sensitivity as inputs;
- CPU load and measured bandwidth;
- precision levels (quantized variants);
- edge–cloud collaboration;
- execution location as a decision of its own (offloading).
