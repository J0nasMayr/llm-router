"""Several models per node: memory budget, least-recently-used unloading, and
the fit checks of the routing filter and the allocation controller.

Run from the repository root: ./venv/bin/python -m unittest discover -s tests -v
"""

import asyncio
import unittest
import weakref

from src.services.allocation_controller import AllocationController
from tests.test_load_time import make_orchestrator, node
from tests.worker_fixture import MODELS, FakeWorkerTestCase


class WorkerResidencyTest(FakeWorkerTestCase):
    # Estimated sizes: small 2 GB, mid 4 GB, big 16 GB.

    def test_keeps_several_models_within_budget(self):
        worker = self.make_worker(memory_budget_gb=20.0)
        worker._load_model("small")
        worker._load_model("big")
        self.assertEqual(worker._model_state_snapshot()["resident_models"], ["small", "big"])
        _, _, seconds = worker._load_model("small")
        self.assertEqual(seconds, 0.0)
        self.assertEqual(self.model_loader.call_count, 2)

    def test_unloads_least_recently_used_model(self):
        worker = self.make_worker(memory_budget_gb=20.0)
        worker._load_model("small")
        worker._load_model("mid")
        worker._load_model("small")  # small is now used more recently than mid
        worker._load_model("big")  # 2 + 4 + 16 > 20: mid must go
        self.assertEqual(worker._model_state_snapshot()["resident_models"], ["small", "big"])

    def test_max_models_one_keeps_a_single_model(self):
        worker = self.make_worker(max_models=1)
        worker._load_model("small")
        worker._load_model("mid")
        self.assertEqual(worker._model_state_snapshot()["resident_models"], ["mid"])

    def test_unloaded_model_is_released_before_the_next_load(self):
        worker = self.make_worker(max_models=1)
        model, _, _ = worker._load_model("small")
        unloaded = weakref.ref(model)
        del model
        alive_during_next_load = []
        load = self.model_loader.side_effect

        def check_then_load(name, **kwargs):
            alive_during_next_load.append(unloaded() is not None)
            return load(name, **kwargs)

        self.model_loader.side_effect = check_then_load
        worker._load_model("mid")
        self.assertEqual(alive_during_next_load, [False])

    def test_model_larger_than_budget_loads_alone(self):
        worker = self.make_worker(memory_budget_gb=10.0)
        worker._load_model("small")
        with self.assertLogs("worker", level="WARNING"):
            worker._load_model("big")
        self.assertEqual(worker._model_state_snapshot()["resident_models"], ["big"])


class FitCheckTest(unittest.TestCase):
    def test_filter_skips_node_too_small_for_the_model(self):
        orch = make_orchestrator()
        small_gpu = node(resident_models=[], memory_budget_gb=10.0)
        self.assertEqual(orch._node_options_for_model("big", [small_gpu]), [])
        self.assertEqual(len(orch._node_options_for_model("small", [small_gpu])), 1)

    def test_measured_size_replaces_estimate(self):
        orch = make_orchestrator()
        t = node(resident_models=[], memory_budget_gb=10.0, model_memory_gb={"big": 9.5})
        self.assertEqual(len(orch._node_options_for_model("big", [t])), 1)

    def test_node_without_budget_is_not_restricted(self):
        options = make_orchestrator()._node_options_for_model("big", [node(resident_models=[])])
        self.assertEqual(len(options), 1)

    def test_controller_gives_no_share_where_the_model_does_not_fit(self):
        async def build():
            return AllocationController(
                queue_client=None, model_configs=MODELS, rps_provider=dict
            )

        controller = asyncio.run(build())
        small_gpu = node(memory_budget_gb=10.0)
        self.assertFalse(controller._eligible_pair("big", small_gpu))
        self.assertTrue(controller._eligible_pair("small", small_gpu))


if __name__ == "__main__":
    unittest.main()
