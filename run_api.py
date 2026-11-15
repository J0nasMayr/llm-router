"""FastAPI bootstrap for the LLM router.

Constructs the service graph (feature / router / evaluation / orchestrator /
allocation controller), wires them together, and exposes the inference router
from src.api.router. The allocation controller is started/stopped via the
FastAPI lifespan.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from config.manager import load_config
from src.api.router import router as inference_router
from src.queue.redis_client import RedisClient
from src.services.allocation_controller import AllocationController
from src.services.evaluation_service import EvaluationService
from src.services.feature_service import FeatureService
from src.services.orchestrator import Orchestrator
from src.services.router_service import RouterService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_api")

# Module-level singletons. router.py reads `orchestrator` lazily.
orchestrator: Optional[Orchestrator] = None
allocation_controller: Optional[AllocationController] = None


def _build_services():
    """Wire the service graph. Returns (orchestrator, allocation_controller)."""
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

    feasibility = features["feasibility"]
    T_max = float(feasibility["max_latency"])
    load_bandwidth = float(feasibility["load_bandwidth_gb_per_s"])
    orch = Orchestrator(
        feature_service=feature_service,
        router_service=router_service,
        evaluation_service=evaluation_service,
        queue_client=redis_client,
        model_configs=models,
        T_max=T_max,
        dataset_configs=datasets,
        load_bandwidth_gb_per_s=load_bandwidth,
    )

    controller = _build_controller(redis_client, models, orch, load_bandwidth)
    orch.allocation_controller = controller
    return orch, controller


def _build_controller(redis_client, models, orch, load_bandwidth_gb_per_s):
    """The allocation controller configured in allocation.yaml, or None for
    policy "none" (nodes then load models on demand only)."""
    cfg = load_config("allocation")
    policy = cfg["policy"]
    if policy == "none":
        logger.info("Allocation controller disabled (policy: none)")
        return None
    demand = cfg["demand"]
    if demand not in ("shadow", "realized"):
        raise ValueError(f"Unknown demand signal: {demand}")
    # The first plan uses the prior shares of queries per model, scaled to an
    # assumed total request rate.
    prior_demand = None
    if cfg["prior"]:
        shares = load_config(cfg["prior"])
        total_rps = float(cfg["prior_total_rps"])
        prior_demand = {m: float(s) * total_rps for m, s in shares.items() if m in models}
    return AllocationController(
        queue_client=redis_client,
        model_configs=models,
        demand_provider=(
            orch.consume_shadow_window if demand == "shadow" else orch.consume_rps_window
        ),
        policy=policy,
        prior_demand=prior_demand,
        load_bandwidth_gb_per_s=load_bandwidth_gb_per_s,
        planning_horizon_s=float(cfg["planning_horizon_s"]),
        T_cool=float(cfg["cooldown_s"]),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, allocation_controller
    orchestrator, allocation_controller = _build_services()
    if allocation_controller:
        allocation_controller.start()
    logger.info("API ready")
    try:
        yield
    finally:
        if allocation_controller:
            await allocation_controller.stop()
        logger.info("API shutdown complete")


app = FastAPI(title="LLM Router", lifespan=lifespan)
app.include_router(inference_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "orchestrator_ready": orchestrator is not None,
        "controller_ready": allocation_controller is not None
        and allocation_controller.has_plan(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("run_api:app", host="0.0.0.0", port=8000, reload=False)
