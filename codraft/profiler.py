import time
import psutil
from codraft.types import ProfilerMetrics

class ResourceProfiler:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time: float = 0.0
        self.mem_start_bytes: int = 0
        self.peak_mem_mb: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def start(self):
        self.start_time = time.perf_counter()
        mem_info = self.process.memory_info()
        self.mem_start_bytes = mem_info.rss
        self.peak_mem_mb = self.mem_start_bytes / (1024 * 1024)
        psutil.cpu_percent(interval=None)

    def record_tokens(self, in_tokens: int, out_tokens: int):
        self.input_tokens += in_tokens
        self.output_tokens += out_tokens

    def update_peak_memory(self):
        current_mb = self.process.memory_info().rss / (1024 * 1024)
        if current_mb > self.peak_mem_mb:
            self.peak_mem_mb = current_mb

    def stop(self) -> ProfilerMetrics:
        duration = time.perf_counter() - self.start_time
        self.update_peak_memory()
        mem_end_mb = self.process.memory_info().rss / (1024 * 1024)
        delta_mb = max(0.0, mem_end_mb - (self.mem_start_bytes / (1024 * 1024)))
        cpu_pct = psutil.cpu_percent(interval=None)

        estimated_joules = round(65.0 * duration * (max(10.0, cpu_pct) / 100.0), 2)

        return ProfilerMetrics(
            duration_sec=round(duration, 3),
            peak_ram_mb=round(self.peak_mem_mb, 2),
            delta_ram_mb=round(delta_mb, 2),
            cpu_utilization_pct=round(cpu_pct, 1),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            estimated_energy_joules=estimated_joules
        )
