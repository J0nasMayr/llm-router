"""Model load time in telemetry and in the feasibility estimate.

Run from the repository root: ./venv/bin/python -m unittest discover -s tests -v
"""

import unittest

from src.services.orchestrator import Orchestrator
from tests.worker_fixture import MODELS, FakeWorkerTestCase


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
    # Size-based estimates with 2 GB/s: small = 2 GB / 2 = 1 s, big = 16 / 2 = 8 s.

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

    def test_load_in_progress_while_model_is_held(self):
        t = node(resident_models=["small"], loading={"model_id": "big", "elapsed_s": 3.0})
        # The node already holds `small`: wait for the running load only.
        self.assertAlmostEqual(make_orchestrator()._expected_latency(t, "small"), 5.0 + 1.0)

    def test_worker_without_model_state_is_unchanged(self):
        t = node(queue_length=1)
        self.assertAlmostEqual(make_orchestrator()._expected_latency(t, "big"), 2 * 2.0)

    def test_load_time_can_make_a_node_infeasible(self):
        loaded = node(node_id="a", queue_name="qa", resident_models=["big"])
        empty = node(node_id="b", queue_name="qb", resident_models=[])
        options = make_orchestrator(sla=9.0)._node_options_for_model("big", [loaded, empty])
        self.assertEqual([n["node_id"] for n, _ in options], ["a"])


class WorkerLoadTimeTest(FakeWorkerTestCase):
    def test_load_is_timed_and_reported(self):
        worker = self.make_worker()
        _, _, seconds = worker._load_model("small")
        self.assertGreater(seconds, 0.0)
        state = worker._model_state_snapshot()
        self.assertEqual(state["resident_models"], ["small"])
        self.assertIsNone(state["loading"])
        self.assertAlmostEqual(state["load_time_by_model"]["small"], seconds)
        # Already loaded: no new load.
        _, _, seconds = worker._load_model("small")
        self.assertEqual(seconds, 0.0)
        self.assertEqual(self.model_loader.call_count, 1)

    def test_state_during_a_swap(self):
        worker = self.make_worker(max_models=1)
        worker._load_model("small")
        worker._load_model("big")
        # The old model is unloaded before the new one loads.
        self.assertEqual(self.during_load["resident_models"], [])
        self.assertEqual(self.during_load["loading"]["model_id"], "big")

    def test_failed_load_is_not_recorded(self):
        worker = self.make_worker(max_models=1)
        worker._load_model("small")
        self.fail_next_load = True
        with self.assertRaises(RuntimeError):
            worker._load_model("big")
        state = worker._model_state_snapshot()
        self.assertEqual(state["resident_models"], [])
        self.assertIsNone(state["loading"])
        self.assertNotIn("big", state["load_time_by_model"])
        # The unloaded model is loaded again on its next use.
        self.fail_next_load = False
        model, _, seconds = worker._load_model("small")
        self.assertIsNotNone(model)
        self.assertGreater(seconds, 0.0)

    def test_telemetry_contains_model_state(self):
        worker = self.make_worker(memory_budget_gb=20.0)
        worker._load_model("big")
        worker._publish_telemetry()
        payload = self.redis.set_node_telemetry.call_args[0][1]
        self.assertEqual(payload["resident_models"], ["big"])
        self.assertIsNone(payload["loading"])
        self.assertIn("big", payload["load_time_by_model"])
        self.assertEqual(payload["memory_budget_gb"], 20.0)
        self.assertIn("model_memory_gb", payload)


if __name__ == "__main__":
    unittest.main()
