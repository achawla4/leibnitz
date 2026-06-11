"""Small performance benchmark for FFT and filtering helpers."""

from __future__ import annotations

import time

try:
    from .fft_tools import magnitude_spectrum
    from .filters import moving_average
    from .utils import generate_multitone
except ImportError:
    from fft_tools import magnitude_spectrum
    from filters import moving_average
    from utils import generate_multitone


def time_call(func, *args, repeats: int = 10, **kwargs) -> float:
    """Return average runtime in seconds."""
    start = time.perf_counter()
    for _ in range(repeats):
        func(*args, **kwargs)
    return (time.perf_counter() - start) / repeats


def run_benchmark(sample_rate: int = 48_000, duration: float = 2.0, repeats: int = 20) -> dict[str, float]:
    """Benchmark representative FFT and FIR workloads."""
    _, signal = generate_multitone([440, 1000, 3000], sample_rate, duration, noise_std=0.02, seed=11)
    return {
        "magnitude_spectrum_seconds": time_call(magnitude_spectrum, signal, sample_rate, repeats=repeats),
        "moving_average_seconds": time_call(moving_average, signal, 101, repeats=repeats),
    }


def main() -> None:
    """Print benchmark results."""
    for name, seconds in run_benchmark().items():
        print(f"{name}: {seconds:.6f}")


if __name__ == "__main__":
    main()
