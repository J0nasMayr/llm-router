# Base project

This project is a fork of a recent paper that presents a novel llm routing system called "greenserve".
This is the project for my bachelor thesis, which aims to extend the functionality of "greenserve".
To understand what "greenserve" does exactly, have a look at the README.md (this is the original readme-file of "greenserve").

## Which three points this project adds to "greenserve"

The thesis in /tex describes them (Chapter 3): a telemetry-driven feasibility filter, the contextual bandit on top of it, and a demand-driven model placement controller. Open plans for the evaluation are in step3_placement_plan.md.

## Tech Stack
- Python with a venv and pip (`requirements.txt`, see the Makefile)
- Make
- Redis for task queues, results, node telemetry and target placements

# Execution paths (path of the control flow)

[router.py](./src/api/router.py) is the entry point for inference; This is where the `api-endpoint` for inference is defined/served ([run_api.py](./run_api.py) starts the API).

[orchestrator.py](./src/services/orchestrator.py) then does steps:
- _filter_nodes: feasibility filter on the node telemetry (Stage 1)
- extract_features ([feature_service.py](./src/services/feature_service.py))
- select_model ([router_service.py](./src/services/router_service.py)), LinUCB with the infeasible models masked
- _pick_queue_for_model: dispatch to a node (traffic shares of the plan, else lowest expected latency)
- enqueue_task ([redis_client.py](./src/queue/redis_client.py))
- get_result ([redis_client.py](./src/queue/redis_client.py))

Running alongside:
- [inference_service.py](./src/services/inference_service.py): the worker on each node. It executes tasks from its queue, keeps several models in GPU memory, follows its target placement and publishes telemetry.
- [allocation_controller.py](./src/services/allocation_controller.py): the placement controller (Kalman trigger + placement program). The thesis calls it the "placement controller".

## Rules
- Do not commit anything. Simply edit the code and then let me have a look.

# The latex source in /tex

This is my bachelor thesis right now. The folder division is quite straight forward with /tex/figures, /tex/sources and 3 source files in /tex (main.tex, references.bib, DPSbachelor.cls).

