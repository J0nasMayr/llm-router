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
        alpha=0.5,
        beta=0.5,
        sla_max_latency_seconds=60.0,
        min_free_gpu_memory_gb=0.5,
        network_delay_by_tier=None,
    ):
        self.feature_service = feature_service
        self.router_service = router_service
        self.evaluation_service = evaluation_service
        self.queue_client = queue_client
        self.alpha = alpha
        self.beta = beta
        self.model_configs = model_configs
        self.model_usage_counts = {}
        self.sla_max_latency_seconds = sla_max_latency_seconds
        self.min_free_gpu_memory_gb = min_free_gpu_memory_gb
        self.network_delay_by_tier = network_delay_by_tier or {
            "edge": 0.0,
            "cloud": 0.05,
        }

    def _network_delay(self, tier):
        return float(self.network_delay_by_tier.get(tier, 0.0))

    def _node_free_memory_gb(self, telemetry):
        hw = telemetry.get("hardware_state") or {}
        if "gpu_memory_free_gb" in hw:
            return float(hw["gpu_memory_free_gb"])
        # Older telemetry only reports used/total — derive free if both present.
        total = hw.get("gpu_memory_total_gb")
        used = hw.get("gpu_memory_used_gb")
        if total is not None and used is not None:
            return max(0.0, float(total) - float(used))
        return None

    def _avg_latency_for_model(self, telemetry, model_id):
        """Per-model avg latency on this node, falling back to the node-wide avg
        when this model hasn't run here yet."""
        per_model = telemetry.get("latency_by_model") or {}
        if model_id in per_model:
            return float(per_model[model_id])
        return float(telemetry.get("avg_task_latency", 0.0) or 0.0)

    def _expected_latency(self, telemetry, model_id):
        """expected_latency = network_delay(tier) + (queue_length + 1) * avg_task_latency.

        avg_task_latency of 0 means the node hasn't processed this model yet —
        treat as immediately available rather than infinitely fast: the (+1)
        still accounts for the new task.
        """
        avg = self._avg_latency_for_model(telemetry, model_id)
        qlen = int(telemetry.get("queue_length", 0) or 0)
        return self._network_delay(telemetry.get("tier")) + (qlen + 1) * avg

    def _filter_nodes(self, telemetry_list):
        """Filter nodes by free GPU memory and return the survivors. Latency is
        evaluated per-model later in `_node_options_for_model`, since T(m,q)
        depends on the chosen model's per-model average latency."""
        surviving = []
        for t in telemetry_list:
            free_gb = self._node_free_memory_gb(t)
            if free_gb is not None and free_gb < self.min_free_gpu_memory_gb:
                continue
            surviving.append(t)
        return surviving

    def _node_options_for_model(self, model_id, surviving_nodes):
        """Return [(node, expected_latency)] of nodes whose tier matches `model_id`
        and whose per-model expected latency satisfies the SLA. Sorted by latency."""
        target_tier = self.model_configs.get(model_id, {}).get("tier")
        if not target_tier:
            return []
        options = []
        for n in surviving_nodes:
            if n.get("tier") != target_tier:
                continue
            elat = self._expected_latency(n, model_id)
            if elat <= self.sla_max_latency_seconds:
                options.append((n, elat))
        options.sort(key=lambda x: x[1])
        return options

    def _available_models(self, surviving_nodes):
        """Models with at least one feasible node under the per-model SLA."""
        return [
            mid
            for mid in self.model_configs
            if self._node_options_for_model(mid, surviving_nodes)
        ]

    def _pick_queue_for_model(self, model_id, surviving_nodes):
        options = self._node_options_for_model(model_id, surviving_nodes)
        if not options:
            return None
        best_node, _ = options[0]
        return best_node.get("queue_name") or f"llm_tasks_node_{best_node.get('node_id')}"

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

        # Stage 1: 
        # - Pull telemetry, prune nodes lacking free GPU memory
        # - Derive the per-model feasible set from per-model expected latency
        telemetry_list = []
        try:
            telemetry_list = self.queue_client.get_all_node_telemetry() or []
        except Exception as e:
            logger.warning(f"Telemetry fetch failed, falling back to tier routing: {e}")
        surviving_nodes = self._filter_nodes(telemetry_list)
        available_models = self._available_models(surviving_nodes)

        features = self.feature_service.extract_features(query_text, metadata)
        if model_id is None:
            # Stage 2: 
            # - Bandit chooses among physically viable arms (mask)
            # - Empty available_models (no telemetry yet) => bandit sees all arms
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

        # Stage 3: 
        # Pick the lowest-latency surviving node of the chosen model's tier
        # Falls back to the tier-broadcast queue when no telemetry is available (cold start) or no node matches
        target_queue = self._pick_queue_for_model(model_id, surviving_nodes)
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
                    alpha=self.alpha,
                    beta=self.beta,
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
