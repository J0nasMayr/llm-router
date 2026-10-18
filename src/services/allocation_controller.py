"""Per-model allocation controller.

Decides which models each node keeps loaded (the placement) and how each
model's traffic is split across the nodes that hold it. Adopts HAS-GPU's
Kalman + threshold + cooldown trigger; the placement is a small mixed-integer
program in the spirit of ENOVA's replica program and SageServe's placement
ILP. Nodes read their target placement from Redis and load or unload models
to follow it; the orchestrator reads the traffic shares via eligibility_for().

The controller is optional: without it, nodes load models on demand and keep
the most recently used ones.
"""

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from src.services.model_memory import (
    fits_on_node,
    load_time_on_node,
    size_on_node_gb,
)

logger = logging.getLogger(__name__)


class ScalarKalman:
    """Self-tuning EWMA-style Kalman filter on a single scalar signal.

    Update rule (A = H = 1):
        R = (1-K)·R_prev + K·R_obs
        P = (1-K)·(P_prev + Q),  K = (P_prev + Q) / (P_prev + Q + D)

    Q (process noise) and D (observation noise) are the only knobs; K
    self-adjusts. Defaults match HAS-GPU's reported settings.
    """

    def __init__(self, Q: float = 0.01, D: float = 1.0, R0: float = 0.0, P0: float = 1.0):
        self.Q = Q
        self.D = D
        self.R = R0
        self.P = P0

    def update(self, observation: float) -> float:
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.D)
        self.R = self.R + K * (observation - self.R)
        self.P = (1 - K) * P_pred
        return self.R


Pair = Tuple[str, str]


def solve_placement(
    models: List[str],
    nodes: List[str],
    demand: Dict[str, float],
    latency: Dict[Pair, float],
    load_time: Dict[Pair, float],
    size_gb: Dict[Pair, float],
    budget_gb: Dict[str, Optional[float]],
    eligible: Dict[Pair, bool],
    current: Dict[Pair, bool],
    target_utilization: float = 0.85,
    planning_horizon_s: float = 300.0,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Tuple[str, float]]]]:
    """Placement program: which models each node keeps loaded, and how each
    model's traffic is split across the nodes that hold it.

    Variables, per eligible (model m, node n) pair: y[m,n] in {0, 1}, node n
    keeps m loaded, and x[m,n] in [0, 1], the share of m's requests that n
    serves. The rest of m's requests, 1 - sum_n x[m,n], find no loaded
    replica.

    Objective, in node-seconds per second:
        sum d[m] * tau[m,n] * x[m,n]                     served requests
      + sum d[m] * (tau_avg[m] + load_avg[m]) * rest[m]   requests without a
                                                          replica: a load, then
                                                          the inference
      + sum load[m,n] / T_plan * y[m,n] * (1 - y0[m,n])   new loads, spread over
                                                          the planning horizon
    so a model is worth loading once about one request for it is expected
    within the horizon. Constraints:
        sum_n x[m,n] <= 1                    for each model
        x[m,n] <= y[m,n]                     traffic only to nodes that hold m
        sum_m size[m,n] * y[m,n] <= S[n]     memory budget of each node
        sum_m d[m] * tau[m,n] * x[m,n] <= rho   load of each node
    Loading nothing is always feasible. A tiny bonus for the current placement
    (y0) keeps plans that are equally good from replacing it.

    Returns (placement, weights): for every node, the models to keep loaded
    (highest demand first), and for every model with traffic on a replica,
    its shares [(node, share)] summing to 1. If the solver fails, the current
    placement is kept and each model's traffic is split equally over the
    nodes that hold it.
    """
    pairs = [(m, n) for m in models for n in nodes if eligible.get((m, n))]
    placement: Dict[str, List[str]] = {n: [] for n in nodes}
    if not pairs:
        return placement, {}

    by_demand = {m: -demand.get(m, 0.0) for m in models}
    P = len(pairs)
    num_vars = 2 * P  # y for every pair, then x for every pair
    keep_bonus = 1e-6

    c = np.zeros(num_vars)
    for p, (m, n) in enumerate(pairs):
        d = demand.get(m, 0.0)
        own = [q for q, (mm, _) in enumerate(pairs) if mm == m]
        tau_avg = sum(latency[pairs[q]] for q in own) / len(own)
        load_avg = sum(load_time[pairs[q]] for q in own) / len(own)
        # Serving a request here instead of leaving it without replica.
        c[P + p] = d * (latency[(m, n)] - tau_avg - load_avg)
        if current.get((m, n)):
            c[p] = -keep_bonus
        else:
            c[p] = load_time[(m, n)] / planning_horizon_s

    rows, lower, upper = [], [], []

    def add_row(coefficients, low, up):
        row = np.zeros(num_vars)
        for index, value in coefficients:
            row[index] = value
        rows.append(row)
        lower.append(low)
        upper.append(up)

    for m in models:
        own = [q for q, (mm, _) in enumerate(pairs) if mm == m]
        if own:
            add_row([(P + q, 1.0) for q in own], -np.inf, 1.0)
    for p in range(P):
        add_row([(P + p, 1.0), (p, -1.0)], -np.inf, 0.0)
    for n in nodes:
        own = [q for q, (_, nn) in enumerate(pairs) if nn == n]
        if not own:
            continue
        if budget_gb.get(n) is not None:
            add_row([(q, size_gb[pairs[q]]) for q in own], -np.inf, float(budget_gb[n]))
        add_row(
            [(P + q, demand.get(pairs[q][0], 0.0) * latency[pairs[q]]) for q in own],
            -np.inf,
            target_utilization,
        )

    integrality = np.zeros(num_vars)
    integrality[:P] = 1
    res = milp(
        c,
        constraints=LinearConstraint(np.array(rows), lower, upper),
        integrality=integrality,
        bounds=Bounds(0.0, 1.0),
        options={"time_limit": 10.0},
    )

    shares: Dict[str, List[Tuple[str, float]]] = {}
    if res.success and res.x is not None:
        y, x = res.x[:P], res.x[P:]
        for p, (m, n) in enumerate(pairs):
            if y[p] > 0.5:
                placement[n].append(m)
            if x[p] > 1e-6:
                shares.setdefault(m, []).append((n, float(x[p])))
    else:
        logger.warning(f"Placement program failed ({res.message}); keeping the current placement")
        for m, n in pairs:
            if current.get((m, n)):
                placement[n].append(m)
                shares.setdefault(m, []).append((n, 1.0))

    weights = {}
    for m, node_shares in shares.items():
        total = sum(s for _, s in node_shares)
        weights[m] = [(n, s / total) for n, s in node_shares]
    for n in placement:
        placement[n].sort(key=lambda m: by_demand[m])
    return placement, weights


class AllocationController:
    """Periodic loop: read telemetry, evaluate the trigger, and re-plan which
    models each node keeps loaded and how each model's traffic is split.

    Policy "dynamic" makes a first plan from the prior demand (if given) and
    re-plans whenever a trigger rule fires; "static" makes that first plan
    only and keeps it.
    """

    def __init__(
        self,
        queue_client,
        model_configs: Dict,
        demand_provider: Callable[[], Dict[str, float]],
        policy: str = "dynamic",
        prior_demand: Optional[Dict[str, float]] = None,
        load_bandwidth_gb_per_s: float = 2.0,
        planning_horizon_s: float = 300.0,
        theta_up: float = 0.85,
        theta_down: float = 0.5,
        T_cool: float = 30.0,
        R_min: float = 0.1,
        kappa: float = 2.0,
        min_free_gpu_memory_gb: float = 0.5,
        tick_interval: float = 5.0,
        Q: float = 0.01,
        D: float = 1.0,
        nominal_latency_s: float = 1.0,
    ):
        if policy not in ("dynamic", "static"):
            raise ValueError(f"Unknown allocation policy: {policy}")
        self.queue_client = queue_client
        self.model_configs = model_configs
        self.demand_provider = demand_provider
        self.policy = policy
        self.prior_demand = prior_demand
        self.load_bandwidth_gb_per_s = load_bandwidth_gb_per_s
        self.planning_horizon_s = planning_horizon_s
        self.theta_up = theta_up
        self.theta_down = theta_down
        self.T_cool = T_cool
        self.R_min = R_min
        self.kappa = kappa
        self.min_free_gpu_memory_gb = min_free_gpu_memory_gb
        self.tick_interval = tick_interval
        # Target placements expire unless renewed; like node telemetry, they
        # survive two missed ticks.
        self.placement_ttl_s = int(tick_interval * 3) + 1
        self.nominal_latency_s = nominal_latency_s
        self._Q, self._D = Q, D
        self._kalman: Dict[str, ScalarKalman] = {}
        # Per-(model, node) latency tracking for the deviation trigger:
        # a Kalman prediction of the pair's latency plus an exponentially
        # weighted variance of its residuals.
        self._lat_kalman: Dict[Pair, ScalarKalman] = {}
        self._lat_resvar: Dict[Pair, float] = {}
        self._lat_samples: Dict[Pair, int] = {}
        self._last_rerun: float = 0.0
        self._planned = False
        # Last plan: models each node should keep loaded, and traffic shares.
        self._targets: Dict[str, List[str]] = {}
        self._weights: Dict[str, List[Tuple[str, float]]] = {}
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    # ---------- telemetry helpers ----------

    @staticmethod
    def _node_free_memory_gb(telemetry) -> Optional[float]:
        hw = telemetry.get("hardware_state") or {}
        if "gpu_memory_free_gb" in hw:
            return float(hw["gpu_memory_free_gb"])
        total, used = hw.get("gpu_memory_total_gb"), hw.get("gpu_memory_used_gb")
        if total is not None and used is not None:
            return max(0.0, float(total) - float(used))
        return None

    @staticmethod
    def _latency_for_pair(telemetry, model_id) -> float:
        per_model = telemetry.get("latency_by_model") or {}
        if model_id in per_model:
            return float(per_model[model_id])
        return float(telemetry.get("avg_task_latency", 0.0) or 0.0)

    def _eligible_pair(self, model_id, telemetry) -> bool:
        if telemetry.get("tier") != self.model_configs.get(model_id, {}).get("tier"):
            return False
        # Same check as the routing filter: a model the node's memory budget
        # cannot hold is never placed there.
        if not fits_on_node(telemetry, model_id, self.model_configs.get(model_id)):
            return False
        free_gb = self._node_free_memory_gb(telemetry)
        if free_gb is not None and free_gb < self.min_free_gpu_memory_gb:
            return False
        return True

    def _model_capacity(self, model_id, telemetry_list) -> float:
        """Capacity ≡ sum over the model's replicas of (share × 1/latency)."""
        weights = dict(self._weights.get(model_id, []))
        if not weights:
            return 0.0
        total = 0.0
        for t in telemetry_list:
            nid = t.get("node_id")
            if nid not in weights:
                continue
            lat = self._latency_for_pair(t, model_id)
            if lat > 0:
                total += weights[nid] * (1.0 / lat)
        return total

    # ---------- trigger ----------

    def _kalman_for(self, model_id) -> ScalarKalman:
        if model_id not in self._kalman:
            self._kalman[model_id] = ScalarKalman(Q=self._Q, D=self._D)
        return self._kalman[model_id]

    def _update_latency_filters(self, telemetry_list) -> Optional[str]:
        """Feed per-(model, node) latency observations into their Kalman filters
        and residual-variance trackers. Returns a deviation reason when an
        observation exceeds its prediction by more than kappa standard
        deviations. Must run every tick so the filters stay current even when
        another rule fires first."""
        MIN_SAMPLES = 5
        EW = 0.9  # residual-variance smoothing
        reason = None
        for t in telemetry_list:
            nid = t.get("node_id")
            per_model = t.get("latency_by_model") or {}
            for m, obs in per_model.items():
                obs = float(obs)
                key = (m, nid)
                kf = self._lat_kalman.get(key)
                if kf is None:
                    kf = ScalarKalman(Q=self._Q, D=self._D, R0=obs)
                    self._lat_kalman[key] = kf
                residual = obs - kf.R
                var = self._lat_resvar.get(key, 0.0)
                n = self._lat_samples.get(key, 0)
                # Sigma floor: a perfectly stable latency history has zero
                # residual variance, which would make any deviation threshold
                # unreachable; floor at 10% of the predicted latency.
                sigma = max(var**0.5, 0.1 * max(kf.R, 1e-6))
                if (
                    reason is None
                    and n >= MIN_SAMPLES
                    and residual > self.kappa * sigma
                ):
                    reason = (
                        f"latency_deviation({m}@{nid}: obs={obs:.2f}s > "
                        f"pred={kf.R:.2f}s + {self.kappa}σ)"
                    )
                self._lat_resvar[key] = EW * var + (1 - EW) * residual**2
                self._lat_samples[key] = n + 1
                kf.update(obs)
        return reason

    def _trigger_reason(
        self, predicted_rps, capacities, telemetry_list, latency_reason=None
    ) -> Optional[str]:
        # A single cooldown gates every rule (design: react at most once per
        # T_cool, suppressing oscillation and re-planning storms).
        now = time.time()
        if now - self._last_rerun <= self.T_cool:
            return None

        # Memory-pressure (any node low)
        for t in telemetry_list:
            free_gb = self._node_free_memory_gb(t)
            if free_gb is not None and free_gb < self.min_free_gpu_memory_gb:
                return f"memory_pressure(node={t.get('node_id')}, free={free_gb:.2f}GB)"

        # Per-model scale-up: demand approaches the capacity of the model's
        # replicas, or a model without replica reaches the placement program's
        # break-even of about one request per planning horizon, from which on
        # loading it can pay off.
        break_even = 1.0 / self.planning_horizon_s
        for m, R in predicted_rps.items():
            cap = capacities.get(m, 0.0)
            if cap > 0 and R > self.theta_up * cap:
                return f"scale_up({m}: R={R:.2f} > {self.theta_up}·C={self.theta_up * cap:.2f})"
            if cap == 0 and R > break_even:
                return f"place({m}: R={R:.4f} > 1/T_plan={break_even:.4f})"

        # Per-(model, node) latency deviation (computed in _update_latency_filters)
        if latency_reason:
            return latency_reason

        # Global scale-down
        nonzero = {m: R for m, R in predicted_rps.items() if capacities.get(m, 0.0) > 0}
        if nonzero and all(R < self.theta_down * capacities[m] for m, R in nonzero.items()):
            if any(R > self.R_min for R in predicted_rps.values()):
                return "scale_down(all models below θ_down·C)"

        return None

    # ---------- placement ----------

    def _replan(self, telemetry_list, demand, reason):
        tel_by_node = {t.get("node_id"): t for t in telemetry_list if t.get("node_id")}
        nodes = list(tel_by_node)
        if not nodes:
            return
        models = list(self.model_configs.keys())
        eligible: Dict[Pair, bool] = {}
        latency: Dict[Pair, float] = {}
        load_time: Dict[Pair, float] = {}
        size_gb: Dict[Pair, float] = {}
        current: Dict[Pair, bool] = {}
        budget_gb: Dict[str, Optional[float]] = {}
        for n, t in tel_by_node.items():
            budget_gb[n] = t.get("memory_budget_gb")
            # The current placement is the last plan for the node; before
            # the first plan, the models the node holds.
            held = set(self._targets.get(n, t.get("resident_models") or []))
            for m in models:
                if not self._eligible_pair(m, t):
                    continue
                cfg = self.model_configs.get(m)
                eligible[(m, n)] = True
                lat = self._latency_for_pair(t, m)
                # Nominal latency for pairs without any history.
                latency[(m, n)] = lat if lat > 0 else self.nominal_latency_s
                load_time[(m, n)] = load_time_on_node(
                    t, m, cfg, self.load_bandwidth_gb_per_s
                )
                size_gb[(m, n)] = size_on_node_gb(t, m, cfg)
                current[(m, n)] = m in held

        placement, weights = solve_placement(
            models,
            nodes,
            demand,
            latency,
            load_time,
            size_gb,
            budget_gb,
            eligible,
            current,
            target_utilization=self.theta_up,
            planning_horizon_s=self.planning_horizon_s,
        )
        self._targets = {n: list(placement.get(n, [])) for n in nodes}
        self._weights = weights
        self._planned = True
        self._last_rerun = time.time()
        self._publish_targets()
        loaded = sum(len(v) for v in self._targets.values())
        logger.info(
            f"Placement planned ({reason}): {loaded} replicas of "
            f"{len(weights)} models on {len(nodes)} nodes"
        )

    def _publish_targets(self):
        """Write each node's target placement with an expiry. The controller
        renews it every tick, so a plan never outlives the controller: if it
        stops, the targets expire and the nodes fall back to loading models on
        demand and keeping the most recently used ones."""
        for n, models in self._targets.items():
            self.queue_client.set_node_placement(
                n, models, ttl_seconds=self.placement_ttl_s
            )

    # ---------- async loop ----------

    async def _tick(self):
        telemetry_list = self.queue_client.get_all_node_telemetry() or []
        if not telemetry_list:
            return
        observed_rps = self.demand_provider() or {}
        # Every model gets an observation each tick (0 when it received no
        # traffic), so a model's prediction decays instead of freezing at its
        # last non-zero value.
        predicted_rps = {
            m: self._kalman_for(m).update(observed_rps.get(m, 0.0))
            for m in self.model_configs
        }
        latency_reason = self._update_latency_filters(telemetry_list)
        if not self._planned:
            reason = "static_plan" if self.policy == "static" else "first_plan"
            self._replan(telemetry_list, self.prior_demand or predicted_rps, reason)
            return
        if self.policy == "static":
            self._publish_targets()
            return
        capacities = {m: self._model_capacity(m, telemetry_list) for m in self.model_configs}
        reason = self._trigger_reason(
            predicted_rps, capacities, telemetry_list, latency_reason
        )
        if reason:
            logger.info(f"Allocation trigger fired: {reason}")
            self._replan(telemetry_list, predicted_rps, reason)
        else:
            self._publish_targets()

    async def _loop(self):
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception as e:
                logger.warning(f"Allocation controller tick failed: {e}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.tick_interval)
            except asyncio.TimeoutError:
                pass

    def start(self):
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())
            logger.info("Allocation controller started")

    async def stop(self):
        self._stop.set()
        if self._task:
            try:
                await self._task
            except Exception:
                pass

    # ---------- public read API ----------

    def eligibility_for(self, model_id) -> List[Tuple[str, float]]:
        """[(node_id, share)] over the nodes that hold the model in the current
        plan. Empty for a model without replicas, which is not restricted."""
        return self._weights.get(model_id, [])

    def has_eligibility(self) -> bool:
        """True once the controller has made its first plan."""
        return self._planned

    def available_models(self) -> List[str]:
        """Models with at least one replica in the current plan."""
        return [m for m, v in self._weights.items() if v]
