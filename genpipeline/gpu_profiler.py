"""
GPU Profiling utilities for identifying bottlenecks.
"""

import torch
import time
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class GPProfiler:
    """GPU profiler to identify memory vs compute bottlenecks."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.events = {}
        self.timings = {}

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code block."""
        if not torch.cuda.is_available():
            yield
            return

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        yield
        end_event.record()
        torch.cuda.synchronize()

        elapsed = start_event.elapsed_time(end_event)
        self.timings[name] = elapsed

    def get_summary(self) -> Dict[str, float]:
        """Get timing summary in milliseconds."""
        return self.timings

    def print_summary(self):
        """Print timing summary sorted by duration."""
        if not self.timings:
            print("No profiling data")
            return

        total = sum(self.timings.values())
        print(f"\n{'=' * 50}")
        print("GPU Profiling Summary (ms)")
        print(f"{'=' * 50}")
        for name, ms in sorted(self.timings.items(), key=lambda x: -x[1]):
            pct = 100 * ms / total if total > 0 else 0
            print(f"  {name:30s}: {ms:8.2f} ms ({pct:5.1f}%)")
        print(f"{'=' * 50}")
        print(f"  {'TOTAL':30s}: {total:8.2f} ms")


def profile_model_forward(model, input_tensor, n_warmup=10, n_runs=100) -> Dict:
    """
    Profile a model's forward pass to identify bottlenecks.

    Returns:
        dict with keys: total_time_ms, op_times, memory_used_mb, gpu_utilization
    """
    device = input_tensor.device
    model = model.eval()

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(input_tensor)

    torch.cuda.synchronize()

    # Profile with PyTorch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(n_runs):
            with torch.no_grad():
                _ = model(input_tensor)

    torch.cuda.synchronize()

    # Extract timing data - use key_averages() for proper timing
    cuda_time = prof.key_averages()

    op_times = {}
    for evt in cuda_time:
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            # Use cuda_time_total or self_cpu_time_total
            cuda_t = getattr(evt, "cuda_time_total", None) or getattr(
                evt, "self_cuda_time_total", 0
            )
            if cuda_t:
                op_times[evt.key] = cuda_t / 1000  # Convert to ms

    # Calculate totals
    total_time = sum(op_times.values()) / n_runs

    # Memory info
    mem_allocated = torch.cuda.memory_allocated(device) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(device) / 1024**2

    return {
        "total_time_ms": total_time,
        "op_times": op_times,
        "memory_used_mb": mem_allocated,
        "memory_reserved_mb": mem_reserved,
    }


def analyze_memory_bandwidth(model, input_tensor, n_runs=50) -> Dict:
    """
    Analyze if model is memory-bandwidth bound or compute bound.

    Returns:
        dict with keys: is_memory_bound, estimated_bandwidth_gbps, utilization_pct
    """
    device = input_tensor.device

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Measure time
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = 1000 * (end - start) / n_runs

    # Estimate memory traffic
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    input_size = input_tensor.numel() * input_tensor.element_size()
    output_size = input_tensor.numel() * input_tensor.element_size()  # Approximate

    # For VAE: input + params read + output write
    memory_traffic_bytes = input_size + total_params + output_size
    memory_traffic_gb = memory_traffic_bytes / 1024**3

    # GPU peak bandwidth (RTX 5080 ~ 1000 GB/s for HBM3e)
    peak_bandwidth_gbps = 1000  # Approximate for Blackwell

    # Actual bandwidth used
    bandwidth_gbps = memory_traffic_gb / (avg_time_ms / 1000)

    utilization = 100 * bandwidth_gbps / peak_bandwidth_gbps

    return {
        "avg_time_ms": avg_time_ms,
        "memory_traffic_gb": memory_traffic_gb,
        "estimated_bandwidth_gbps": bandwidth_gbps,
        "peak_bandwidth_gbps": peak_bandwidth_gbps,
        "utilization_pct": utilization,
        "is_memory_bound": utilization < 50,
    }


def count_dequant_ops(model, input_tensor) -> int:
    """Count potential dequantization operations in the model."""
    # This is a simplified check - real quantization would need INT8 models
    count = 0
    for name, module in model.named_modules():
        # Check for precision transitions
        if "float" in str(type(module)).lower():
            count += 1
    return count


def find_reshape_operations(model, input_tensor) -> List[str]:
    """Find all reshape/permute/view operations that may be inefficient."""
    inefficient_ops = []

    def hook_fn(module, input, output):
        op_name = type(module).__name__
        if hasattr(output, "shape") and any(s == 1 for s in output.shape):
            # Check for squeeze/unsqueeze patterns
            if "squeeze" in op_name.lower() or "unsqueeze" in op_name.lower():
                inefficient_ops.append(f"{op_name}: {output.shape}")

    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    return inefficient_ops


def benchmark_batch_sizes(model, input_shape, batch_sizes=[1, 2, 4, 8, 16, 32, 64]):
    """Benchmark different batch sizes to find optimal."""
    results = []

    for bs in batch_sizes:
        x = torch.randn(bs, *input_shape).cuda()

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

        # Benchmark
        n_runs = 20
        start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

        time_ms = 1000 * (end - start) / n_runs
        throughput = bs / (time_ms / 1000)  # samples/sec

        results.append(
            {
                "batch_size": bs,
                "time_ms": time_ms,
                "throughput": throughput,
            }
        )

        print(
            f"  batch={bs:3d}: {time_ms:7.2f} ms/batch, {throughput:7.0f} samples/sec"
        )

    return results


if __name__ == "__main__":
    # Quick test
    from genpipeline.vae_design_model import DesignVAE

    model = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).cuda().eval()
    x = torch.randn(8, 1, 64, 64, 64).cuda()

    print("Profiling VAE forward pass...")
    results = profile_model_forward(model, x, n_runs=20)

    print(f"\nTotal time: {results['total_time_ms']:.2f} ms")
    print(f"Memory used: {results['memory_used_mb']:.1f} MB")

    print("\nTop 10 operations by time:")
    for i, (op, t) in enumerate(
        sorted(results["op_times"].items(), key=lambda x: -x[1])[:10]
    ):
        print(f"  {op}: {t:.2f} ms")

    print("\nBatch size benchmark:")
    benchmark_batch_sizes(model, (1, 64, 64, 64))
