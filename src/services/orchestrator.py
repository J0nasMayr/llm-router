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
        sla_max_latency_seconds=60.0,
    ):
        self.feature_service = feature_service
        self.router_service = router_service
        self.evaluation_service = evaluation_service
        self.queue_client = queue_client
        self.trade_off_lambda = trade_off_lambda
        self.model_configs = model_configs
        self.model_usage_counts = {}
        self.sla_max_latency_seconds = sla_max_latency_seconds

    def _expected_latency(self, telemetry):
        """expected_latency = (queue_length + 1) * avg_task_latency.

        avg_task_latency of 0 means the node hasn't processed anything yet — treat
        as immediately available rather than infinitely fast: the (+1) ensures we
        still account for the new task.
        """
        avg = float(telemetry.get("avg_task_latency", 0.0) or 0.0)
        qlen = int(telemetry.get("queue_length", 0) or 0)
        return (qlen + 1) * avg

    def _filter_nodes(self, telemetry_list):
        """Return (surviving_nodes, available_models). Each surviving node gets an
        `expected_latency` field added. available_models is the set of model_ids
        whose tier has at least one surviving node."""
        surviving = []
        surviving_tiers = set()
        for t in telemetry_list:
            elat = self._expected_latency(t)
            if elat <= self.sla_max_latency_seconds:
                t = {**t, "expected_latency": elat}
                surviving.append(t)
                if t.get("tier"):
                    surviving_tiers.add(t["tier"])
        available_models = [
            mid
            for mid, cfg in self.model_configs.items()
            if cfg.get("tier") in surviving_tiers
        ]
        return surviving, available_models

    def _pick_node_for_model(self, model_id, surviving_nodes):
        """Return queue_name of the surviving node with the lowest expected_latency
        whose tier matches the chosen model. None if no match."""
        target_tier = self.model_configs.get(model_id, {}).get("tier")
        if not target_tier:
            return None
        candidates = [n for n in surviving_nodes if n.get("tier") == target_tier]
        if not candidates:
            return None
        best = min(candidates, key=lambda n: n.get("expected_latency", float("inf")))
        return best.get("queue_name") or f"llm_tasks_node_{best.get('node_id')}"

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

        # Stage 1: pull telemetry and filter nodes by expected-latency SLA.
        telemetry_list = []
        try:
            telemetry_list = self.queue_client.get_all_node_telemetry() or []
        except Exception as e:
            logger.warning(f"Telemetry fetch failed, falling back to tier routing: {e}")
        surviving_nodes, available_models = self._filter_nodes(telemetry_list)

        features = self.feature_service.extract_features(query_text, metadata)
        if model_id is None:
            # Stage 2: bandit chooses among physically viable arms (mask).
            # Empty available_models (no telemetry yet) => bandit sees all arms.
            model_id = self.router_service.select_model(
                features,
                available_models=available_models or None,
            )
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

        # Stage 3: pick the optimal surviving node of the chosen model's tier.
        # Falls back to the step-1 tier-broadcast queue when no telemetry is
        # available (cold start) or no node matches.
        target_queue = self._pick_node_for_model(model_id, surviving_nodes)
        if target_queue is None:
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
