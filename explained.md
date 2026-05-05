## Execution paths


[router.py](./src/api/router.py) is the entry point for inference; This is where the `api-endpoint` for inference is defined/served.

[orchestrator.py](./src/services/orchestrator.py) then does steps:
- extract_features ([feature_service.py](./src/services/feature_service.py))
- select_model ([router_service.py](./src/services/router_service.py))
- enqueue_task ([redis_client.py](./src/queue/redis_client.py))
- get_result ([redis_client.py](./src/queue/redis_client.py))

