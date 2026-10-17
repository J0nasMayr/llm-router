"""What process_query returns to the client: the worker's measurements with
every response, the reward when a reference is given, and the worker's own
error message for a failed task.

Run from the repository root: ./venv/bin/python -m unittest discover -s tests -v
"""

import asyncio
import unittest
from unittest import mock

from src.services.orchestrator import Orchestrator
from tests.worker_fixture import MODELS

WORKER_METRICS = {
    "latency": 1.0,
    "energy_consumption": 10.0,
    "input_tokens": 5,
    "output_tokens": 2,
    "load_time": 3.5,
}


class ProcessQueryResponseTest(unittest.TestCase):
    def setUp(self):
        self.queue = mock.MagicMock()
        self.queue.get_all_node_telemetry.return_value = []
        self.queue.enqueue_task.return_value = "task-1"
        self.queue.get_result.return_value = {
            "response": "42",
            "status": "completed",
            "metrics": dict(WORKER_METRICS),
        }
        self.router = mock.MagicMock()
        self.router.select_model.return_value = "small"
        self.evaluation = mock.MagicMock()
        self.evaluation.evaluate_and_calculate_reward.return_value = {
            "accuracy": 1.0,
            "reward": 0.7,
        }
        features = mock.MagicMock()
        features.extract_features.return_value = {"task_type": "gsm8k"}
        self.orch = Orchestrator(
            feature_service=features,
            router_service=self.router,
            evaluation_service=self.evaluation,
            queue_client=self.queue,
            model_configs=MODELS,
        )

    def query(self, **kwargs):
        return asyncio.run(self.orch.process_query("What is 6 * 7?", **kwargs))

    def test_worker_metrics_without_reference(self):
        metrics = self.query()["metrics"]
        self.assertEqual(metrics["load_time"], 3.5)
        self.assertNotIn("reward", metrics)
        self.router.update.assert_not_called()

    def test_worker_metrics_and_reward_with_reference(self):
        metrics = self.query(reference="42")["metrics"]
        self.assertEqual(metrics["load_time"], 3.5)
        self.assertEqual(metrics["reward"], 0.7)
        self.router.update.assert_called_once()

    def test_failed_task_keeps_the_worker_error(self):
        self.queue.get_result.return_value = {
            "status": "failed",
            "error": "CUDA out of memory",
            "selected_model": "small",
        }
        response = self.query(reference="42")
        self.assertEqual(response["error"], "CUDA out of memory")
        self.router.update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
