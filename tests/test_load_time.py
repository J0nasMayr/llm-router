"""Model load time in telemetry and in the feasibility estimate (step 3, P0).

Run from the repository root: ./venv/bin/python -m unittest discover -s tests -v
"""

import time
import unittest
from unittest import mock

from src.services.inference_service import InferenceService
from src.services.orchestrator import Orchestrator

MODELS = {
    "small": {"name": "org/small", "tier": "edge", "parameter_count": 1},
    "big": {"name": "org/big", "tier": "edge", "parameter_count": 8},
}


def make_orchestrator(sla=30.0):
    return Orchestrator(
        feature_service=None,
        router_service=None,
        evaluation_service=None,
        queue_client=None,
        model_configs=MODELS,
        sla_max_latency_seconds=sla,
        load_bandwidth_gb_per_s=2.0,
    )


def node(**fields):
    telemetry = {
        "node_id": "n1",
        "tier": "edge",
        "queue_name": "q1",
        "queue_length": 0,
        "avg_task_latency": 1.0,
        "latency_by_model": {"small": 1.0, "big": 2.0},
    }
    telemetry.update(fields)
    return telemetry


class ExpectedLatencyTest(unittest.TestCase):
    # Size-based estimates with 2 GB/s: small = 2*1/2 = 1 s, big = 2*8/2 = 8 s.

    def test_loaded_model_adds_no_load_time(self):
        t = node(resident_models=["big"], queue_length=2)
        self.assertAlmostEqual(make_orchestrator()._expected_latency(t, "big"), 3 * 2.0)

    def test_missing_model_adds_estimated_load_time(self):
        t = node(resident_models=["small"])
        self.assertAlmostEqual(make_orchestrator()._expected_latency(t, "big"), 8.0 + 2.0)

    def test_measured_load_time_replaces_estimate(self):
        t = node(resident_models=[], load_time_by_model={"big": 5.0})
        self.assertAlmostEqual(make_orchestrator()._expected_latency(t, "big"), 5.0 + 2.0)

    def test_load_in_progress(self):
        orch = make_orchestrator()
        t = node(resident_models=[], loading={"model_id": "big", "elapsed_s": 3.0})
        # Same model: only the rest of the running load (8 - 3).
        self.assertAlmostEqual(orch._expected_latency(t, "big"), 5.0 + 2.0)
        # Other model: rest of the running load, then its own load.
        self.assertAlmostEqual(orch._expected_latency(t, "small"), 5.0 + 1.0 + 1.0)

    def test_worker_without_model_state_is_unchanged(self):
        t = node(queue_length=1)
        self.assertAlmostEqual(make_orchestrator()._expected_latency(t, "big"), 2 * 2.0)

    def test_load_time_can_make_a_node_infeasible(self):
        loaded = node(node_id="a", queue_name="qa", resident_models=["big"])
        empty = node(node_id="b", queue_name="qb", resident_models=[])
        options = make_orchestrator(sla=9.0)._node_options_for_model("big", [loaded, empty])
        self.assertEqual([n["node_id"] for n, _ in options], ["a"])


class WorkerLoadTrackingTest(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch("src.services.inference_service.EnergyMeasurement"),
            mock.patch("src.services.inference_service.AutoTokenizer"),
            mock.patch("src.services.inference_service.AutoModelForCausalLM"),
            mock.patch.object(InferenceService, "_warm_up_model"),
        ]
        mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        self.model_loader = mocks[2].from_pretrained
        self.model_loader.side_effect = self._fake_load
        self.redis = mock.MagicMock()
        self.redis.get_queue_length.return_value = 0
        self.worker = InferenceService(
            self.redis, {}, MODELS, tier="edge", node_id="e1"
        )
        self.during_load = None
        self.fail_next_load = False

    def _fake_load(self, model_name, **kwargs):
        self.during_load = self.worker._model_state_snapshot()
        if self.fail_next_load:
            raise RuntimeError("load failed")
        time.sleep(0.02)
        return mock.MagicMock(name=model_name)

    def test_load_is_timed_and_reported(self):
        _, _, seconds = self.worker._load_model("small")
        self.assertGreater(seconds, 0.0)
        resident, loading, load_times = self.worker._model_state_snapshot()
        self.assertEqual(resident, ["small"])
        self.assertIsNone(loading)
        self.assertAlmostEqual(load_times["small"], seconds)
        # Already loaded: no new load.
        _, _, seconds = self.worker._load_model("small")
        self.assertEqual(seconds, 0.0)
        self.assertEqual(self.model_loader.call_count, 1)

    def test_state_during_a_swap(self):
        self.worker._load_model("small")
        self.worker._load_model("big")
        resident, loading, _ = self.during_load
        self.assertEqual(resident, [])  # the old model is already unloaded
        self.assertEqual(loading["model_id"], "big")

    def test_failed_load_leaves_no_stale_model(self):
        self.worker._load_model("small")
        self.fail_next_load = True
        with self.assertRaises(RuntimeError):
            self.worker._load_model("big")
        resident, loading, load_times = self.worker._model_state_snapshot()
        self.assertEqual(resident, [])
        self.assertIsNone(loading)
        self.assertNotIn("big", load_times)
        # The unloaded model is loaded again instead of returning None.
        self.fail_next_load = False
        model, _, seconds = self.worker._load_model("small")
        self.assertIsNotNone(model)
        self.assertGreater(seconds, 0.0)

    def test_telemetry_contains_model_state(self):
        self.worker._load_model("big")
        self.worker._publish_telemetry()
        payload = self.redis.set_node_telemetry.call_args[0][1]
        self.assertEqual(payload["resident_models"], ["big"])
        self.assertIsNone(payload["loading"])
        self.assertIn("big", payload["load_time_by_model"])


if __name__ == "__main__":
    unittest.main()
