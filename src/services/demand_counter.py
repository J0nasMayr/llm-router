import time
from typing import Dict


class DemandCounter:
    """Per-model arrival counts that the orchestrator records and the
    placement controller reads as its demand signal: "realized" counts the
    model each query was routed to, "shadow" the model the bandit would have
    chosen if every model were available. Shadow demand is the default signal:
    unlike realized demand, it also counts models the filter currently masks,
    e.g. because no node holds them."""

    SIGNALS = ("realized", "shadow")

    def __init__(self):
        now = time.time()
        self._windows = {signal: ({}, now) for signal in self.SIGNALS}

    def record(self, routed_model, preferred_model):
        """Count one query in both signals."""
        self._count("realized", routed_model)
        self._count("shadow", preferred_model)

    def consume(self, signal) -> Dict[str, float]:
        """Per-model arrival rate (req/s) of `signal` since its last
        consumption; that window is then reset."""
        counters, started = self._windows[signal]
        now = time.time()
        self._windows[signal] = ({}, now)
        elapsed = max(now - started, 1e-6)
        return {m: c / elapsed for m, c in counters.items()}

    def _count(self, signal, model_id):
        counters, _ = self._windows[signal]
        counters[model_id] = counters.get(model_id, 0) + 1
