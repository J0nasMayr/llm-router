# src/services/orchestrator.py
import asyncio
import logging
import random
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
        allocation_controller=None,
        result_timeout_seconds=None,
        dataset_configs=None,
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
        # How long to wait for a worker result. Must exceed the SLA deadline,
        # otherwise a request that just meets the SLA still times out here.
        self.result_timeout_seconds = (
            result_timeout_seconds
            if result_timeout_seconds is not None
            else sla_max_latency_seconds + 30.0
        )
        self.min_free_gpu_memory_gb = min_free_gpu_memory_gb
        self.network_delay_by_tier = network_delay_by_tier or {
            "edge": 0.0,
            "cloud": 0.05,
        }
        # Optional allocation controller (HAS-GPU-style trigger + ENOVA LP).
        # When set, routing restricts to nodes the LP marked eligible for the
        # model, and uses LP-produced weights for weighted selection.
        self.allocation_controller = allocation_controller
        # Per-model arrival counters consumed by the controller via
        # consume_rps_window(); reset after each consumption.
        self._rps_counters = {}
        self._rps_window_start = time.time()
        # Per-dataset scoring and generation settings (datasets.yaml), applied
        # when a request names its dataset in metadata.
        self.dataset_configs = dataset_configs or {}

    def _dataset_settings(self, dataset):
        """(extraction_method, evaluation_metric, generation_parameters) for a
        known dataset, mirroring how data/process.py prepares queries offline.
        All None for an unknown or missing dataset."""
        cfg = self.dataset_configs.get(dataset)
        if not cfg:
            return None, None, None
        extraction = cfg.get("extraction") or {}
        extraction_method = extraction.get("method", "raw")
        if extraction_method == "regex":
            extraction_method = f"regex_{extraction.get('pattern', '')}"
        return (
            extraction_method,
            cfg.get("evaluation_metric", "exact_match"),
            dict(cfg.get("generation") or {}),
        )

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

    def _allocation_weights(self, model_id):
        """LP-produced {node_id: weight} for the model, or None if no controller
        or no entry yet. Empty dict means the LP has run but no node is eligible."""
        if self.allocation_controller is None:
            return None
        if not self.allocation_controller.has_eligibility():
            return None
        return dict(self.allocation_controller.eligibility_for(model_id))

    def _node_options_for_model(self, model_id, surviving_nodes):
        """Return [(node, expected_latency)] of nodes whose tier matches `model_id`
        and whose per-model expected latency satisfies the SLA. Sorted by latency.
        When an LP eligibility map is present, restrict candidates to its keys."""
        target_tier = self.model_configs.get(model_id, {}).get("tier")
        if not target_tier:
            return []
        weights = self._allocation_weights(model_id)
        options = []
        for n in surviving_nodes:
            if n.get("tier") != target_tier:
                continue
            if weights is not None and n.get("node_id") not in weights:
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
        weights = self._allocation_weights(model_id)
        if weights:
            # Weighted random selection over SLA-passing options; falls back
            # to lowest-latency when no LP weight is available for the survivors.
            pool = [(n, weights.get(n.get("node_id"), 0.0)) for n, _ in options]
            total = sum(w for _, w in pool)
            if total > 0:
                pick = random.random() * total
                acc = 0.0
                for node, w in pool:
                    acc += w
                    if pick <= acc:
                        best_node = node
                        break
                else:
                    best_node = pool[-1][0]
            else:
                best_node = options[0][0]
        else:
            best_node = options[0][0]
        return best_node.get("queue_name") or f"llm_tasks_node_{best_node.get('node_id')}"

    def consume_rps_window(self):
        """Return per-model RPS observed since last call, then reset the window.
        Used by AllocationController.rps_provider."""
        now = time.time()
        elapsed = max(now - self._rps_window_start, 1e-6)
        rps = {m: c / elapsed for m, c in self._rps_counters.items()}
        self._rps_counters = {}
        self._rps_window_start = now
        return rps

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
        self._rps_counters[model_id] = self._rps_counters.get(model_id, 0) + 1
        # Explicit arguments win; otherwise use the dataset's own extraction,
        # metric and generation settings so online rewards are computed the
        # same way as in the offline GreenServ setup.
        ds_extraction, ds_metric, ds_generation = self._dataset_settings(
            (metadata or {}).get("dataset")
        )
        extraction_method = extraction_method or ds_extraction
        evaluation_metric = evaluation_metric or ds_metric
        generation_parameters = generation_parameters or ds_generation
        # Note: the queue client assigns the definitive task id in enqueue_task.
        task = {
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
            timeout = self.result_timeout_seconds
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

                # The worker nests its measurements under "metrics"
                # (see InferenceService.run_inference).
                worker_metrics = result.get("metrics") or {}
                energy_consumption = worker_metrics.get("energy_consumption", 0)
                input_tokens = worker_metrics.get("input_tokens", 0)
                output_tokens = worker_metrics.get("output_tokens", 0)
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
