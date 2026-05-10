# Base project

This project is a fork of a recent paper that presents a novel llm routing system called "greenserve".
This is the project for my bachelor thesis, which aims to extend the functionality of "greenserve".
To understand what "greenserve" does exactly, have a look at the README.md (this is the original readme-file of "greenserve").

## Which three points this project adds to "greenserve"

All this information can be found in the steps.md file.

## Tech Stack
- Python with uv as package manager
- Make

# Execution paths (path of the control flow)

[router.py](./src/api/router.py) is the entry point for inference; This is where the `api-endpoint` for inference is defined/served.

[orchestrator.py](./src/services/orchestrator.py) then does steps:
- extract_features ([feature_service.py](./src/services/feature_service.py))
- select_model ([router_service.py](./src/services/router_service.py))
- enqueue_task ([redis_client.py](./src/queue/redis_client.py))
- get_result ([redis_client.py](./src/queue/redis_client.py))

## Rules
- Do not commit anything. Simply edit the code and then let me have a look.
