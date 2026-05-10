# src/services/orchestrator.py
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates routing, inference and evaluation"""

    def __init__(
        self,
        feature_service,
        router_service,
        evaluation_service,
        queue_client,
        model_configs,
        trade_off_lambda=0.5,
    ):
        self.feature_service = feature_service
        self.router_service = router_service
        self.evaluation_service = evaluation_service
        self.queue_client = queue_client
        self.trade_off_lambda = trade_off_lambda
        self.model_configs = model_configs
        self.model_usage_counts = {}

    async def process_query(
        self,
        query_text,
        reference=None,
        metadata=None,
        model_id=None,
        wait_for_result=True,
        extraction_method=None,
        evaluation_metric=None,
        generation_parameters=None,
    ):
        """Process a query through the routing pipeline"""

        features = self.feature_service.extract_features(query_text, metadata)
        if model_id is None:
            model_id = self.router_service.select_model(features)
        if model_id not in self.model_usage_counts:
            self.model_usage_counts[model_id] = 0
        self.model_usage_counts[model_id] += 1
        task = {
            "id": f"task_{int(time.time() * 1000)}",
            "query_text": query_text,
            "selected_model": model_id,
            "extraction_method": extraction_method,
            "evaluation_metric": evaluation_metric,
            "generation_parameters": generation_parameters,
        }

        # Determine the tier based on the selected model
        # Default to cloud if something goes wrong
        model_tier = self.model_configs.get(model_id, {}).get("tier", "cloud")
        target_queue = f"llm_inference_tasks_{model_tier}"

        # Pass the targeted queue to the redis client
        task_id = self.queue_client.enqueue_task(task, queue_name=target_queue)

        response = {"task_id": task_id, "model_id": model_id}
        if not wait_for_result:
            return response
        try:
            timeout = 60
            start_time = time.time()
            result = None

            while time.time() - start_time < timeout:
                result = self.queue_client.get_result(task_id)
                if result:
                    break
                await asyncio.sleep(0.2)

            if not result:
                response["error"] = "Timeout waiting for response"
                return response
            response["response"] = result["response"]
            if reference:
                task_type = (
                    features.get("task_type", "default") if features else "default"
                )

                energy_consumption = result.get("energy_consumption", 0)
                input_tokens = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)
                metrics = self.evaluation_service.evaluate_and_calculate_reward(
                    response=result["response"],
                    reference=reference,
                    task_type=task_type,
                    energy_consumption=energy_consumption,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    evaluation_metric=evaluation_metric or "exact_match",
                    extraction_method=extraction_method or "raw",
                    lambda_weight=self.trade_off_lambda,
                )
                self.router_service.update(features, model_id, metrics["reward"])
                response["metrics"] = metrics

        except Exception as e:
            logger.error(f"Error waiting for result: {e}")
            response["error"] = str(e)

        return response

    def get_metrics(self):
        """Get orchestrator metrics"""
        return {
            "model_usage": self.model_usage_counts,
            "total_queries_processed": sum(self.model_usage_counts.values()),
        }
