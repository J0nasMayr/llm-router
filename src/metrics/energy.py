import time

from zeus.monitor import ZeusMonitor


class ZeusMeasurement:
    """GPU energy measurement using Zeus Python API."""

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.monitor = None
        self.window_name = f"inference_{time.time()}"
        self.start_time = None

    def start(self):
        """Start Zeus power measurement for GPU."""
        self.start_time = time.time()

        try:
            self.monitor = ZeusMonitor(gpu_indices=[self.device_id])
            self.monitor.begin_window(self.window_name)
            return True
        except Exception as e:
            print(f"Zeus start failed: {e}")
            self.monitor = None
            return False

    def stop(self):
        """Stop Zeus measurement and return GPU energy used (joules)."""
        if self.start_time is None or self.monitor is None:
            raise RuntimeError(
                "Zeus energy measurement failed: monitor not initialized or start time missing"
            )

        try:
            measurement = self.monitor.end_window(self.window_name)
            return measurement.total_energy
        except Exception as e:
            raise RuntimeError(f"Zeus energy measurement failed during stop: {e}")


class EnergyMetrics:
    """Calculate energy-related metrics."""

    def calculate_per_token(self, total_energy, input_tokens, output_tokens):
        """Calculate energy per token."""
        total_tokens = input_tokens + output_tokens
        if total_tokens == 0:
            return 0.0

        return total_energy / total_tokens
