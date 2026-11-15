"""FastAPI bootstrap for the LLM router.

Constructs the service graph (feature / router / evaluation / orchestrator /
placement controller), wires them together, and exposes the inference router
from src.api.router. The placement controller is started/stopped via the
FastAPI lifespan.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from config.manager import load_config
from src.api.router import router as inference_router
from src.queue.redis_client import RedisClient
from src.services.placement_controller import PlacementController
from src.services.demand_counter import DemandCounter
from src.services.evaluation_service import EvaluationService
from src.services.feature_service import FeatureService
from src.services.orchestrator import Orchestrator
from src.services.router_service import RouterService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_services():
    """Wire the service graph. Returns (orchestrator, placement_controller)."""
    models = load_config("models")
    features = load_config("feature_extraction")
    mab = load_config("mab")
    datasets = load_config("datasets")

    redis_client = RedisClient.from_config()

    feature_service = FeatureService(features)

    algo = mab["algorithm"]
    router_service = RouterService(
        algorithm_name=algo,
        model_ids=list(models.keys()),
        algorithm_params=mab[algo],
        context_dimension=feature_service.context_dimension,
    )

    evaluation_service = EvaluationService()

    feasibility = load_config("feasibility")
    T_max = float(feasibility["max_latency_s"])
    load_bandwidth = float(feasibility["load_bandwidth_gb_per_s"])
    min_free_gb = float(feasibility["min_free_gpu_memory_gb"])

    demand_counter = DemandCounter()
    controller = _build_controller(
        redis_client, models, demand_counter, load_bandwidth, min_free_gb
    )
    orch = Orchestrator(
        feature_service=feature_service,
        router_service=router_service,
        evaluation_service=evaluation_service,
        queue_client=redis_client,
        demand_counter=demand_counter,
        model_configs=models,
        T_max=T_max,
        dataset_configs=datasets,
        load_bandwidth_gb_per_s=load_bandwidth,
        min_free_gpu_memory_gb=min_free_gb,
        placement_controller=controller,
    )
    return orch, controller


def _build_controller(
    redis_client,
    models,
    demand_counter,
    load_bandwidth_gb_per_s,
    min_free_gpu_memory_gb,
):
    """The placement controller configured in placement.yaml, or None for
    policy "none" (nodes then load models on demand only)."""
    cfg = load_config("placement")
    policy = cfg["policy"]
    if policy == "none":
        logger.info("Placement controller disabled (policy: none)")
        return None
    signal = cfg["demand"]
    if signal not in DemandCounter.SIGNALS:
        raise ValueError(f"Unknown demand signal: {signal}")
    # The first plan uses the prior shares of queries per model, scaled to an
    # assumed total request rate.
    prior_demand = None
    if cfg["prior"]:
        shares = load_config(cfg["prior"])
        total_rps = float(cfg["prior_total_rps"])
        prior_demand = {m: float(s) * total_rps for m, s in shares.items() if m in models}
    return PlacementController(
        queue_client=redis_client,
        model_configs=models,
        demand_provider=lambda: demand_counter.consume(signal),
        policy=policy,
        prior_demand=prior_demand,
        load_bandwidth_gb_per_s=load_bandwidth_gb_per_s,
        min_free_gpu_memory_gb=min_free_gpu_memory_gb,
        planning_horizon_s=float(cfg["planning_horizon_s"]),
        T_cool=float(cfg["cooldown_s"]),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    orchestrator, controller = _build_services()
    app.state.orchestrator = orchestrator
    app.state.placement_controller = controller
    if controller:
        controller.start()
    logger.info("API ready")
    try:
        yield
    finally:
        if controller:
            await controller.stop()
        logger.info("API shutdown complete")


app = FastAPI(title="LLM Router", lifespan=lifespan)
app.include_router(inference_router)


@app.get("/health")
def health(request: Request):
    state = request.app.state
    controller = getattr(state, "placement_controller", None)
    return {
        "status": "ok",
        "orchestrator_ready": getattr(state, "orchestrator", None) is not None,
        "controller_ready": controller is not None and controller.has_plan(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("run_api:app", host="0.0.0.0", port=8000, reload=False)
