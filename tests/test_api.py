"""What a client sees from /inference and /health. The services are stubs,
so neither Redis nor a model is needed.

Run from the repository root: ./venv/bin/python -m unittest discover -s tests -v
"""

import unittest
from unittest import mock

from fastapi.testclient import TestClient

import run_api


class StubOrchestrator:
    def __init__(self, result=None, exception=None):
        self.result = result
        self.exception = exception

    async def process_query(self, **kwargs):
        if self.exception is not None:
            raise self.exception
        return self.result


class StubController:
    def start(self):
        pass

    async def stop(self):
        pass

    def has_plan(self):
        return True


class ApiTest(unittest.TestCase):
    def client(self, orchestrator, controller=None):
        """A client of the real app, whose lifespan gets the stub services."""
        patcher = mock.patch.object(
            run_api, "_build_services", return_value=(orchestrator, controller)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        client = TestClient(run_api.app)
        client.__enter__()  # runs the lifespan
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_result_is_returned(self):
        result = {"task_id": "t1", "model_id": "small", "response": "42", "metrics": {}}
        response = self.client(StubOrchestrator(result)).post(
            "/inference", json={"query": "What is 6 * 7?"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "42")

    def test_orchestrator_error_reaches_the_client_unchanged(self):
        result = {"task_id": "t1", "model_id": "small", "error": "Timeout waiting for response"}
        response = self.client(StubOrchestrator(result)).post(
            "/inference", json={"query": "q"}
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Timeout waiting for response")

    def test_exception_becomes_a_500_with_its_message(self):
        client = self.client(StubOrchestrator(exception=RuntimeError("bandit failed")))
        with self.assertLogs(level="ERROR") as logs:
            response = client.post("/inference", json={"query": "q"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "bandit failed")
        self.assertIn("Inference failed", logs.output[0])

    def test_health_reports_both_services(self):
        health = self.client(StubOrchestrator(), StubController()).get("/health").json()
        self.assertEqual(
            health, {"status": "ok", "orchestrator_ready": True, "controller_ready": True}
        )

    def test_health_without_controller(self):
        health = self.client(StubOrchestrator()).get("/health").json()
        self.assertFalse(health["controller_ready"])


if __name__ == "__main__":
    unittest.main()
