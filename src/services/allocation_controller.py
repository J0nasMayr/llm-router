"""Per-model allocation controller.

Adopts HAS-GPU's Kalman + threshold + cooldown trigger and an ENOVA-style
per-model replicas+weights LP. Produces a `model_id -> [(node_id, weight)]`
eligibility map that the orchestrator consults at routing time.

The controller is optional: orchestrator works without it (cold-start path),
and falls back gracefully when the LP has not yet produced a map.
"""

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

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


def solve_replicas_weights(
    models: List[str],
    nodes: List[str],
    demand: Dict[str, float],
    latency: Dict[Tuple[str, str], float],
    eligible: Dict[Tuple[str, str], bool],
) -> Dict[str, List[Tuple[str, float]]]:
    """ENOVA-style allocation LP.

    Variables: x[m,n] = fraction of model m's traffic routed to node n.
    Constraints:
      sum_n x[m,n] = 1                              for each servable model m
      sum_m demand[m] * latency[m,n] * x[m,n] <= 1  for each node n  (no over-saturation)
      x[m,n] = 0                                    where not eligible
    Objective: minimize sum_{m,n} demand[m] * latency[m,n] * x[m,n]
               i.e. total node-busy time across the system.

    A model is servable if it has at least one eligible node. Unservable models
    get an empty entry and are left out of the program.

    Returns `model_id -> [(node_id, weight)]` (weights sum to 1 per servable model).
    On LP failure, falls back to equal weighting over each model's eligible nodes.
    """
    # A model without any eligible node (e.g. no live node of its tier) can
    # never satisfy sum_n x[m,n] = 1; keeping its row would make the whole
    # program infeasible and push every model onto the equal-weight fallback.
    out: Dict[str, List[Tuple[str, float]]] = {
        m: [] for m in models if not any(eligible.get((m, n), False) for n in nodes)
    }
    models = [m for m in models if m not in out]
    M, N = len(models), len(nodes)
    if M == 0:
        return out

    num_vars = M * N
    idx = lambda m, n: m * N + n  # noqa: E731

    # Objective coefficients: demand[m] * latency[m,n]
    c = np.zeros(num_vars)
    for m_i, m in enumerate(models):
        for n_i, n in enumerate(nodes):
            if eligible.get((m, n), False):
                c[idx(m_i, n_i)] = demand.get(m, 0.0) * max(latency.get((m, n), 0.0), 1e-9)

    # sum_n x[m,n] = 1 (per model)
    A_eq = np.zeros((M, num_vars))
    for m_i in range(M):
        for n_i in range(N):
            A_eq[m_i, idx(m_i, n_i)] = 1.0
    b_eq = np.ones(M)

    # sum_m demand[m] * latency[m,n] * x[m,n] <= 1 (per node)
    A_ub = np.zeros((N, num_vars))
    for n_i, n in enumerate(nodes):
        for m_i, m in enumerate(models):
            if eligible.get((m, n), False):
                A_ub[n_i, idx(m_i, n_i)] = demand.get(m, 0.0) * max(latency.get((m, n), 0.0), 1e-9)
    b_ub = np.ones(N)

    bounds = [
        (0.0, 1.0) if eligible.get((m, n), False) else (0.0, 0.0)
        for m in models
        for n in nodes
    ]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        x = res.x.reshape(M, N)
        for m_i, m in enumerate(models):
            weights = [
                (nodes[n_i], float(x[m_i, n_i])) for n_i in range(N) if x[m_i, n_i] > 1e-6
            ]
            out[m] = weights
        return out

    logger.warning(f"LP infeasible ({res.message}); falling back to equal weights")
    for m in models:
        elig = [n for n in nodes if eligible.get((m, n), False)]
        w = 1.0 / len(elig)
        out[m] = [(n, w) for n in elig]
    return out


class AllocationController:
    """Periodic loop: read telemetry, evaluate trigger, optionally re-solve LP.

    Holds the current eligibility map; orchestrator consults it via
    `eligibility_for(model_id)`.
    """

    def __init__(
        self,
        queue_client,
        model_configs: Dict,
        rps_provider: Callable[[], Dict[str, float]],
        theta_up: float = 0.85,
        theta_down: float = 0.5,
        T_cool: float = 30.0,
        R_min: float = 0.1,
        kappa: float = 2.0,
        min_free_gpu_memory_gb: float = 0.5,
        tick_interval: float = 5.0,
        Q: float = 0.01,
        D: float = 1.0,
    ):
        self.queue_client = queue_client
        self.model_configs = model_configs
        self.rps_provider = rps_provider
        self.theta_up = theta_up
        self.theta_down = theta_down
        self.T_cool = T_cool
        self.R_min = R_min
        self.kappa = kappa
        self.min_free_gpu_memory_gb = min_free_gpu_memory_gb
        self.tick_interval = tick_interval
        self._Q, self._D = Q, D
        self._kalman: Dict[str, ScalarKalman] = {}
        # Per-(model, node) latency tracking for the deviation trigger:
        # a Kalman prediction of the pair's latency plus an exponentially
        # weighted variance of its residuals.
        self._lat_kalman: Dict[Tuple[str, str], ScalarKalman] = {}
        self._lat_resvar: Dict[Tuple[str, str], float] = {}
        self._lat_samples: Dict[Tuple[str, str], int] = {}
        self._last_rerun: float = 0.0
        self._eligibility: Dict[str, List[Tuple[str, float]]] = {}
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
        free_gb = self._node_free_memory_gb(telemetry)
        if free_gb is not None and free_gb < self.min_free_gpu_memory_gb:
            return False
        return True

    def _model_capacity(self, model_id, telemetry_list) -> float:
        """Capacity ≡ sum over current-eligible nodes of (weight × 1/latency)."""
        weights = dict(self._eligibility.get(model_id, []))
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
        # No eligibility yet — first tick, never gated
        if not self._eligibility:
            return "first_run"

        # A single cooldown gates every rule (design: react at most once per
        # T_cool, suppressing oscillation and LP re-run storms).
        now = time.time()
        if now - self._last_rerun <= self.T_cool:
            return None

        # Memory-pressure (any node low)
        for t in telemetry_list:
            free_gb = self._node_free_memory_gb(t)
            if free_gb is not None and free_gb < self.min_free_gpu_memory_gb:
                return f"memory_pressure(node={t.get('node_id')}, free={free_gb:.2f}GB)"

        # Per-model scale-up
        for m, R in predicted_rps.items():
            cap = capacities.get(m, 0.0)
            if cap > 0 and R > self.theta_up * cap:
                return f"scale_up({m}: R={R:.2f} > {self.theta_up}·C={self.theta_up * cap:.2f})"

        # Per-(model, node) latency deviation (computed in _update_latency_filters)
        if latency_reason:
            return latency_reason

        # Global scale-down
        nonzero = {m: R for m, R in predicted_rps.items() if capacities.get(m, 0.0) > 0}
        if nonzero and all(R < self.theta_down * capacities[m] for m, R in nonzero.items()):
            if any(R > self.R_min for R in predicted_rps.values()):
                return "scale_down(all models below θ_down·C)"

        return None

    # ---------- LP actuator ----------

    def _rerun_lp(self, telemetry_list, predicted_rps):
        models = list(self.model_configs.keys())
        nodes = [t.get("node_id") for t in telemetry_list if t.get("node_id")]
        if not nodes:
            return

        tel_by_node = {t.get("node_id"): t for t in telemetry_list}
        eligible: Dict[Tuple[str, str], bool] = {}
        latency: Dict[Tuple[str, str], float] = {}
        for m in models:
            for n in nodes:
                t = tel_by_node[n]
                if self._eligible_pair(m, t):
                    eligible[(m, n)] = True
                    lat = self._latency_for_pair(t, m)
                    # Cold-start: assume nominal 1s if a node hasn't run this model yet
                    latency[(m, n)] = lat if lat > 0 else 1.0
                else:
                    eligible[(m, n)] = False

        # Demand: predicted RPS, with a tiny floor so models the bandit hasn't
        # tried yet still get capacity reserved.
        demand = {m: max(predicted_rps.get(m, 0.0), 1e-3) for m in models}
        self._eligibility = solve_replicas_weights(models, nodes, demand, latency, eligible)
        self._last_rerun = time.time()
        nonempty = sum(1 for v in self._eligibility.values() if v)
        logger.info(f"Allocation LP solved: {nonempty}/{len(models)} models have eligible nodes")

    # ---------- async loop ----------

    async def _tick(self):
        telemetry_list = self.queue_client.get_all_node_telemetry() or []
        if not telemetry_list:
            return
        observed_rps = self.rps_provider() or {}
        # Every model gets an observation each tick (0 when it received no
        # traffic), so a model's prediction decays instead of freezing at its
        # last non-zero value.
        predicted_rps = {
            m: self._kalman_for(m).update(observed_rps.get(m, 0.0))
            for m in self.model_configs
        }
        capacities = {m: self._model_capacity(m, telemetry_list) for m in self.model_configs}
        latency_reason = self._update_latency_filters(telemetry_list)
        reason = self._trigger_reason(
            predicted_rps, capacities, telemetry_list, latency_reason
        )
        if reason:
            logger.info(f"Allocation trigger fired: {reason}")
            self._rerun_lp(telemetry_list, predicted_rps)

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
        """Return [(node_id, weight)] for the model. Empty when no entry yet."""
        return self._eligibility.get(model_id, [])

    def has_eligibility(self) -> bool:
        """True once the LP has produced at least one non-empty entry."""
        return any(self._eligibility.values())

    def available_models(self) -> List[str]:
        """Models that have at least one eligible node in the current map."""
        return [m for m, v in self._eligibility.items() if v]
