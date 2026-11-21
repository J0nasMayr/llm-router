import asyncio
import logging
import random
import time

from src.queue.redis_client import node_queue_name, tier_queue_name
from src.services.node_telemetry import (
    fits_on_node,
    free_memory_gb,
    latency_on_node,
    load_time_on_node,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates routing, inference and evaluation"""

    def __init__(
        self,
        feature_service,
        router_service,
        evaluation_service,
        queue_client,
        demand_counter,
        model_configs,
        T_max,
        load_bandwidth_gb_per_s,
        min_free_gpu_memory_gb,
        network_delay_by_tier=None,
        placement_controller=None,
        result_timeout_seconds=None,
        dataset_configs=None,
    ):
        self.feature_service = feature_service
        self.router_service = router_service
        self.evaluation_service = evaluation_service
        self.queue_client = queue_client
        self.demand_counter = demand_counter
        self.model_configs = model_configs
        self.T_max = T_max
        # How long to wait for a worker result. Must exceed the deadline T_max,
        # otherwise a request that just meets it still times out here.
        self.result_timeout_seconds = (
            result_timeout_seconds
            if result_timeout_seconds is not None
            else T_max + 30.0
        )
        self.min_free_gpu_memory_gb = min_free_gpu_memory_gb
        self.network_delay_by_tier = network_delay_by_tier or {
            "edge": 0.0,
            "cloud": 0.05,
        }
        # Assumed disk-to-GPU speed for estimating the load time of a model a
        # node has not loaded yet.
        self.load_bandwidth_gb_per_s = load_bandwidth_gb_per_s
        # Optional placement controller. Once it has a placement, a model
        # with replicas is routed to the nodes that hold it, in proportion to
        # its traffic shares; see _node_options_for_model.
        self.placement_controller = placement_controller
        # Per-dataset scoring and generation settings (datasets.yaml), applied
        # when a request names its dataset in metadata.
        self.dataset_configs = dataset_configs or {}

    def _dataset_settings(self, dataset):
        """(extraction_method, evaluation_metric, generation_parameters) of a
        dataset in datasets.yaml, mirroring how data/process.py prepares
        queries offline; raw text, exact match and no generation settings for
        an unknown or missing dataset."""
        cfg = self.dataset_configs.get(dataset)
        if not cfg:
            return "raw", "exact_match", {}
        extraction = cfg["extraction"]
        extraction_method = extraction["method"]
        if extraction_method == "regex":
            extraction_method = f"regex_{extraction['pattern']}"
        return extraction_method, cfg["evaluation_metric"], dict(cfg["generation"])

    def _network_delay(self, tier):
        return float(self.network_delay_by_tier.get(tier, 0.0))

    def _load_time(self, telemetry, model_id):
        """Seconds this node needs to load `model_id`: the node's measured
        average, or before its first load of the model, an estimate from the
        model's bf16 size."""
        return load_time_on_node(
            telemetry,
            model_id,
            self.model_configs.get(model_id),
            self.load_bandwidth_gb_per_s,
        )

    def _pending_load_time(self, telemetry, model_id):
        """Load time a new task for `model_id` waits for on this node: the rest
        of a load already in progress, plus loading `model_id` itself unless
        the node holds (or is loading) it. 0 for workers that do not report
        which models they hold."""
        resident = telemetry.get("resident_models")
        if resident is None:
            return 0.0
        own_load = (
            0.0 if model_id in resident else self._load_time(telemetry, model_id)
        )
        loading = telemetry.get("loading")
        if loading:
            loading_model = loading.get("model_id")
            elapsed = float(loading.get("elapsed_s", 0.0) or 0.0)
            remaining = max(0.0, self._load_time(telemetry, loading_model) - elapsed)
            if loading_model == model_id:
                return remaining
            return remaining + own_load
        return own_load

    def _expected_latency(self, telemetry, model_id):
        """expected_latency = network_delay(tier) + pending_load_time + (queue_length + 1) * avg_latency.

        avg_latency is latency_on_node: 0 for a node that has run nothing yet,
        so only the network delay and the load time count for it. Queued tasks
        are assumed to need no further model change; this can be optimistic
        when they need models the node does not hold, as loading those may
        also unload `model_id`.
        """
        avg = latency_on_node(telemetry, model_id)
        qlen = int(telemetry.get("queue_length", 0) or 0)
        return (
            self._network_delay(telemetry.get("tier"))
            + self._pending_load_time(telemetry, model_id)
            + (qlen + 1) * avg
        )

    def _filter_nodes(self, telemetry_list):
        """Filter nodes by free GPU memory and return the survivors. Latency is
        evaluated per-model later in `_node_options_for_model`, since T(m,q)
        depends on the chosen model's per-model average latency."""
        surviving = []
        for t in telemetry_list:
            free_gb = free_memory_gb(t)
            if free_gb is not None and free_gb < self.min_free_gpu_memory_gb:
                continue
            surviving.append(t)
        return surviving

    def _traffic_shares(self, model_id):
        """{node_id: share}: the model's traffic shares w[m,n] over its
        replicas in the controller's plan, or None if there is no controller,
        no plan yet, or no replica of the model in it. A model without replicas
        is not restricted: it runs wherever the filter allows, loaded on
        demand."""
        if self.placement_controller is None:
            return None
        return self.placement_controller.traffic_shares(model_id) or None

    def _node_options_for_model(self, model_id, surviving_nodes):
        """Return [(node, expected_latency)] of nodes whose tier matches `model_id`,
        whose memory budget can hold the model, and whose per-model expected
        latency meets the deadline T_max. Sorted by latency. When the placement
        has replicas of the model, only their nodes are returned, unless none
        of them meets the deadline; then the other nodes serve as overflow and
        load the model on demand."""
        target_tier = self.model_configs.get(model_id, {}).get("tier")
        if not target_tier:
            return []
        shares = self._traffic_shares(model_id)
        options, overflow = [], []
        for n in surviving_nodes:
            if n.get("tier") != target_tier:
                continue
            if not fits_on_node(n, model_id, self.model_configs.get(model_id)):
                continue
            elat = self._expected_latency(n, model_id)
            if elat > self.T_max:
                continue
            if shares is None or n.get("node_id") in shares:
                options.append((n, elat))
            else:
                overflow.append((n, elat))
        if not options:
            options = overflow
        options.sort(key=lambda x: x[1])
        return options

    def _options_by_model(self, surviving_nodes):
        """{model_id: options} for every model; see _node_options_for_model."""
        return {
            mid: self._node_options_for_model(mid, surviving_nodes)
            for mid in self.model_configs
        }

    def _pick_node(self, model_id, options):
        """The node a query for `model_id` goes to: among the options that hold
        a replica, chosen at random in proportion to the traffic shares; else
        the option with the lowest expected latency."""
        shares = self._traffic_shares(model_id) or {}
        option_shares = [shares.get(n["node_id"], 0.0) for n, _ in options]
        if sum(option_shares) > 0:
            return random.choices([n for n, _ in options], weights=option_shares)[0]
        return options[0][0]

    def _choose_model(self, features, options_by_model, model_id=None):
        """Stage 2: (model_id, shadow_choice). The bandit chooses among the
        models with at least one option; with no options at all (no telemetry
        yet, cold start) it sees every model. shadow_choice is the model it
        would choose if no model were masked. A model_id fixed by the caller is
        used as both."""
        if model_id is not None:
            return model_id, model_id
        available_models = [m for m, options in options_by_model.items() if options]
        model_id = self.router_service.select_model(
            features, available_models=available_models or None
        )
        return model_id, self.router_service.shadow_choice() or model_id

    def _dispatch(self, query_text, model_id, generation_parameters, options):
        """Stage 3: enqueue the task on a node of the chosen model (see
        _pick_node), or on the tier queue when no node meets the deadline or
        no telemetry is available yet (cold start). Returns the task id, which
        the queue client assigns."""
        task = {
            "query_text": query_text,
            "selected_model": model_id,
            "generation_parameters": generation_parameters,
        }
        if options:
            node = self._pick_node(model_id, options)
            target_queue = node_queue_name(node["node_id"])
        else:
            model_tier = self.model_configs.get(model_id, {}).get("tier", "cloud")
            target_queue = tier_queue_name(model_tier)
        return self.queue_client.enqueue_task(task, queue_name=target_queue)

    async def _wait_for_result(self, task_id):
        """The worker's result for the task, or None if it does not arrive
        within result_timeout_seconds."""
        start_time = time.time()
        while time.time() - start_time < self.result_timeout_seconds:
            result = self.queue_client.get_result(task_id)
            if result:
                return result
            await asyncio.sleep(0.2)
        return None

    def _evaluate_and_update(
        self,
        result,
        reference,
        features,
        model_id,
        extraction_method,
        evaluation_metric,
    ):
        """Compute the reward of the worker's answer and update the bandit
        with it. Returns the evaluation metrics, reward included."""
        # The worker nests its measurements under "metrics" (see
        # InferenceService._run_inference).
        worker_metrics = result.get("metrics") or {}
        metrics = self.evaluation_service.evaluate_and_calculate_reward(
            response=result["response"],
            reference=reference,
            task_type=features["task_type"],
            energy_consumption=worker_metrics.get("energy_consumption", 0),
            input_tokens=worker_metrics.get("input_tokens", 0),
            output_tokens=worker_metrics.get("output_tokens", 0),
            evaluation_metric=evaluation_metric,
            extraction_method=extraction_method,
        )
        self.router_service.update(features, model_id, metrics["reward"])
        return metrics

    async def process_query(
        self,
        query_text,
        reference=None,
        metadata=None,
        model_id=None,
        wait_for_result=True,
    ):
        """Route one query through Stages 1-3 and return its task id and
        model. Unless wait_for_result is False, also wait for the worker's
        answer and add it with its metrics; with a reference, compute the
        reward and update the bandit."""
        nodes = self._filter_nodes(self.queue_client.get_all_node_telemetry())
        options_by_model = self._options_by_model(nodes)
        features = self.feature_service.extract_features(query_text, metadata)
        model_id, shadow_choice = self._choose_model(
            features, options_by_model, model_id
        )
        self.demand_counter.record(model_id, shadow_choice)
        extraction_method, evaluation_metric, generation_parameters = (
            self._dataset_settings((metadata or {}).get("dataset"))
        )
        task_id = self._dispatch(
            query_text, model_id, generation_parameters, options_by_model.get(model_id)
        )
        response = {"task_id": task_id, "model_id": model_id}
        if not wait_for_result:
            return response
        try:
            result = await self._wait_for_result(task_id)
            if result is None:
                response["error"] = "Timeout waiting for response"
            elif result.get("status") == "failed":
                # Keep the worker's message (e.g. running out of GPU memory).
                response["error"] = result.get("error") or "Inference failed"
            else:
                response["response"] = result["response"]
                # The worker's measurements: generation latency and energy,
                # token counts and model load time. They are returned with
                # every response so that clients, e.g. a load generator, can
                # log them.
                response["metrics"] = dict(result.get("metrics") or {})
                if reference:
                    response["metrics"].update(
                        self._evaluate_and_update(
                            result,
                            reference,
                            features,
                            model_id,
                            extraction_method,
                            evaluation_metric,
                        )
                    )
        except Exception as e:
            logger.error(f"Error waiting for result: {e}")
            response["error"] = str(e)
        return response
