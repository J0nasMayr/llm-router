# Why ENOVA fits best for Step 3

ENOVA's contribution is a control loop that, given live serving telemetry, decides *how many replicas of each model to run on which GPU type and how to weight requests across them.* That maps one-to-one onto the layer above this thesis's existing router: the bandit already picks *which* model serves a query, and the Step 2 work already publishes `queue_length`, `avg_task_latency`, and a hardware snapshot per node — exactly the inputs ENOVA's monitor consumes (its `n_r`, `n_p`, `m_u`, `t_r`). Its replica/weight decision (a small linear program over per-GPU match scores) is light enough to implement in a thesis scope, and slots in cleanly as a controller that periodically updates a `model_id → eligible_node_set` map consulted by `_pick_node_for_model`. ENOVA is also the only paper in the set whose unit of allocation is *the model*, which is what Step 3 asks for.

# Why the others don't

- **NexusSched** targets a different axis: its predictive performance model improves *routing* decisions across already-deployed engines — essentially a smarter version of the `(queue_length+1) * avg_task_latency` heuristic already in Step 2 — but does not reallocate capacity between models.
- **SageServe** is the right architectural match (it scales VMs and places models per-region under SLA) but its mechanism (ARIMA traffic forecasts plus an ILP over a multi-region, IW/NIW-stratified cluster) assumes data-center scale, production traces, and workload tiers this thesis does not have; porting it would be either dishonest or a year of work.
- **CLONE** operates inside a single model on a single edge device (pruning, LoRA swapping, layer-level DVFS); it is orthogonal to allocating resources across models.
