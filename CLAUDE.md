# Base project

This project is a fork of GreenServ (https://github.com/TZData1/llm-inference-router, commit `8e055dd`), an LLM routing system presented in a recent paper.
This is the project for my bachelor thesis, which aims to extend the functionality of GreenServ.
The top of README.md says what this fork inherits from GreenServ, what it extends and what it adds. Below the horizontal line, README.md is GreenServ's original readme-file; have a look at it to understand what GreenServ does exactly.

## Which three points this project adds to GreenServ

This is described in README.md and also in Chapter 3 of the thesis.

## Tech Stack
- Python with a venv and pip (`requirements.txt`, see the Makefile)
- Make
- Redis for task queues, results, node telemetry and target placements

## Commands (from the repository root)
- API: `./venv/bin/python run_api.py` (port 8000)
- Worker, one per node: `./venv/bin/python -m src.services.inference_service --tier edge --node-id edge-1`
- Tests (no GPU needed): `./venv/bin/python -m unittest discover -s tests -v`

# Execution paths (path of the control flow)

[router.py](./src/api/router.py) is the entry point for inference; This is where the `api-endpoint` for inference is defined/served ([run_api.py](./run_api.py) starts the API).

[orchestrator.py](./src/services/orchestrator.py) then does steps:
- _filter_nodes: feasibility filter on the node telemetry (Stage 1)
- extract_features ([feature_service.py](./src/services/feature_service.py))
- select_model ([router_service.py](./src/services/router_service.py)), LinUCB with the infeasible models masked
- _pick_queue_for_model: dispatch to a node (traffic shares of the plan, else lowest expected latency)
- enqueue_task ([redis_client.py](./src/queue/redis_client.py))
- get_result ([redis_client.py](./src/queue/redis_client.py))
- evaluate_and_calculate_reward ([evaluation_service.py](./src/services/evaluation_service.py)) and the update of the bandit, if the query has a reference

Running alongside:
- [inference_service.py](./src/services/inference_service.py): the worker on each node. It executes tasks from its queue, keeps several models in GPU memory, follows its target placement and publishes telemetry.
- [allocation_controller.py](./src/services/allocation_controller.py): the placement controller (Kalman trigger + placement program). The thesis calls it the "placement controller".

# Where the parameters are

- [config/allocation.yaml](./config/allocation.yaml): placement policy (dynamic, static, none), demand signal (shadow, realized), prior, planning horizon, cooldown
- [config/feature_extraction.yaml](./config/feature_extraction.yaml): context features; its `feasibility` block holds the latency deadline and the assumed load speed
- [config/mab.yaml](./config/mab.yaml): bandit parameters
- [config/models.yaml](./config/models.yaml): the 16 models with their tier (edge or cloud)
- Everything else is a constructor default or a command-line flag: the other controller parameters in allocation_controller.py, the worker parameters in inference_service.py, the reward weights in evaluation_service.py, the network delays in orchestrator.py.
- The parameter table at the end of [model.tex](./tex/sources/model.tex) lists all of these values.

# Inherited from GreenServ

/experiments, /data, /db, /paper_figures and the Makefile are GreenServ's and unchanged. The experiments X0-X4 are offline replays of GreenServ's data; the thesis does not use them.

## Rules
- Do not commit anything. Simply edit the code and then let me have a look.
- The thesis describes the code. When one of the two changes, check the other one (Chapter 3 in model.tex, including its parameter table).

# The latex source in /tex

This is my bachelor thesis right now. The folder division is quite straight forward with /tex/figures, /tex/sources and 3 source files in /tex (main.tex, references.bib, DPSbachelor.cls).
The chapters are in /tex/sources: introduction, relatedWork, model (Chapter 3), evaluation and conclusion.
