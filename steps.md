# Extend this to edge as well as cloud:
- Have a clear distinction between edge and cloud models. (this is done using `tier` in [models.py](./config/models.yaml))
- Separate the redis queue into two different queues `llm_tasks_edge` and `llm_tasks_cloud`
- Add `tier` and `dedicated_queue` to the inference service to separate between the queues
- Support this behaviour in the redis client and the orchestrator.


# Filter nodes based on resource states
## Implement Node-Level Telemetry

- Update [inference_service.py](./src/services/inference_service.py) to include a lightweight background thread or async loop.
- Periodically calculate the individual node's metrics: its specific queue length, average inference latency, and hardware state.
- Push these metrics to Redis under a node-specific key (e.g., telemetry:node:<node_id>), including its tier in the payload.

## Retrieve Telemetry in the Orchestrator

- Update [orchestrator.py](./src/services/orchestrator.py) to fetch the latest telemetry data for all active nodes from Redis at the very beginning of the process_query() function.
- Establish a baseline or user-defined SLA (Service Level Agreement) for maximum acceptable latency.

## Calculate Expected Latency and Filter Nodes (Stage 1)

- Define a mathematical constraint for expected latency (e.g., expected_latency = (node_queue_length * node_avg_task_latency) + predicted_new_task_time).
- Evaluate this constraint against the telemetry data for every individual node.
- Map Surviving Nodes to Models: Create a dynamic available_models list. If an edge node survives the filter, all edge models are added to the available list. If a cloud node survives, all cloud models are added. (If multiple nodes of the same tier survive, the tier's models simply remain available).

## Implement Dynamic Masking in the Bandit (Stage 2)

- Update [router_service.py](./src/services/router_service.py) to pass the available_models list down to the underlying bandit algorithm.
- Modify the MAB implementations (e.g., LinUCB) to accept this dynamic mask.
- During the selection phase, force the score (e.g., Upper Confidence Bound) of unavailable models to negative infinity to guarantee the algorithm only chooses from physically viable options.

## Node Selection (Post-MAB Routing)

- Once the MAB selects a model (e.g., llama-3.1-8b, which is an edge model), the Orchestrator checks the list of surviving nodes that belong to that model's tier.
- If there are multiple valid nodes for that tier, it selects the optimal one (e.g., the node with the lowest expected latency).
- The task is then enqueued specifically to that chosen node's queue (e.g., llm_tasks_node_3).

# Implement dynamic resource allocation per model

- For this I have several candidate papers that maybe already do something similar:
    - https://dl.acm.org/doi/pdf/10.1145/3771576
    - https://arxiv.org/pdf/2407.09486
    - https://www.usenix.org/system/files/atc25-tian.pdf
    - https://arxiv.org/pdf/2509.23384v3
- There is also this paper that summarizes general schedulers/routers:
    - https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176238087.79673350/v1