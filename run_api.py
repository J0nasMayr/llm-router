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
    general = load_config("general") or {}
    models = load_config("models") or {}
    features = load_config("feature_extraction") or {}
    mab = load_config("mab") or {}
    infrastructure = load_config("infrastructure") or {}
    datasets = (load_config("datasets") or {}).get("datasets") or {}

    # Redis lives in infrastructure.yaml (port 6380 by default — must match
    # the workers' --redis-port); general.yaml is a legacy fallback.
    redis_cfg = infrastructure.get("redis") or general.get("redis") or {}
    redis_client = RedisClient(
        host=redis_cfg.get("host", "localhost"),
        port=int(redis_cfg.get("port", 6380)),
        password=redis_cfg.get("password"),
    )

    feature_service = FeatureService(features)
    # MetricNormalizer reads config/metrics.yaml itself; its constructor takes
    # a path, not the already-loaded dict.
    evaluation_service = EvaluationService()

    # Bandit choice: prefer linucb if configured, else fall back to epsilon-greedy.
    algo = "linucb" if "linucb" in mab else next(iter(mab.keys()), "epsilon_greedy")
    algo_params = (mab.get(algo) or {}).copy()
    algo_params.pop("name", None)
    algo_params.pop("type", None)
    algo_params.pop("num_actions", None)
    router_service = RouterService(
        algorithm_name=algo,
        model_ids=list(models.keys()),
        algorithm_params=algo_params,
        # Must match the one-hot vector FeatureService actually produces
        # (task types + clusters + complexity bins), not a hardcoded guess.
        context_dimension=feature_service.context_dimension,
    )

    feasibility = features.get("feasibility") or {}
    sla_seconds = float(feasibility.get("max_latency", 60.0))
    load_bandwidth = float(feasibility.get("load_bandwidth_gb_per_s", 2.0))
    orch = Orchestrator(
        feature_service=feature_service,
        router_service=router_service,
        evaluation_service=evaluation_service,
        queue_client=redis_client,
        model_configs=models,
        sla_max_latency_seconds=sla_seconds,
        dataset_configs=datasets,
        load_bandwidth_gb_per_s=load_bandwidth,
    )

    controller = _build_controller(redis_client, models, orch, load_bandwidth)
    orch.allocation_controller = controller
    return orch, controller


def _build_controller(redis_client, models, orch, load_bandwidth_gb_per_s):
    """The allocation controller configured in allocation.yaml, or None for
    policy "none" (nodes then load models on demand only)."""
    cfg = (load_config("allocation") or {}).get("allocation") or {}
    policy = cfg.get("policy", "dynamic")
    if policy == "none":
        logger.info("Allocation controller disabled (policy: none)")
        return None
    demand = cfg.get("demand", "shadow")
    if demand not in ("shadow", "realized"):
        raise ValueError(f"Unknown demand signal: {demand}")
    # The first plan uses the prior shares of queries per model, scaled to an
    # assumed total request rate.
    prior_demand = None
    if cfg.get("prior"):
        shares = (load_config(cfg["prior"]) or {}).get("placement_prior") or {}
        total_rps = float(cfg.get("prior_total_rps", 1.0))
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
        planning_horizon_s=float(cfg.get("planning_horizon_s", 300.0)),
        T_cool=float(cfg.get("cooldown_s", 30.0)),
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
        and allocation_controller.has_eligibility(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("run_api:app", host="0.0.0.0", port=8000, reload=False)
