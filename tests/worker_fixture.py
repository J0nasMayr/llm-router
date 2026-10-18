"""A fake inference worker for the tests: model loading is replaced by a stub,
so no GPU and no model download is needed."""

import time
import unittest
from unittest import mock

from src.services.inference_service import InferenceService

# Estimated bf16 sizes: small 2 GB, mid 4 GB, big 16 GB.
MODELS = {
    "small": {"name": "org/small", "tier": "edge", "parameter_count": 1},
    "mid": {"name": "org/mid", "tier": "edge", "parameter_count": 2},
    "big": {"name": "org/big", "tier": "edge", "parameter_count": 8},
}


class FakeWorkerTestCase(unittest.TestCase):
    """Builds workers whose model loads take a few milliseconds and can be
    made to fail. `during_load` holds the worker's model state as seen while
    the last load ran."""

    def setUp(self):
        patches = [
            mock.patch("src.services.inference_service.EnergyMeasurement"),
            mock.patch("src.services.inference_service.AutoTokenizer"),
            mock.patch("src.services.inference_service.AutoModelForCausalLM"),
            # A plain function, not a mock: a mock would keep references to
            # the models it was called with and keep them alive.
            mock.patch.object(
                InferenceService, "_warm_up_model", lambda *args, **kwargs: None
            ),
        ]
        mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        self.model_loader = mocks[2].from_pretrained
        self.model_loader.side_effect = self._fake_load
        self.redis = mock.MagicMock()
        self.redis.get_queue_length.return_value = 0
        self.worker = None
        self.during_load = None
        self.fail_next_load = False

    def make_worker(self, redis_client=None, node_id="e1", **kwargs):
        self.worker = InferenceService(
            redis_client or self.redis, {}, MODELS, tier="edge", node_id=node_id, **kwargs
        )
        return self.worker

    def _fake_load(self, model_name, **kwargs):
        self.during_load = self.worker._model_state_snapshot()
        if self.fail_next_load:
            raise RuntimeError("load failed")
        time.sleep(0.02)
        return mock.MagicMock(name=model_name)
