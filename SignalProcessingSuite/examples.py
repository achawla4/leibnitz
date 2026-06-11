"""Ready-to-run examples for the signal processing suite."""

from __future__ import annotations

try:
    from .fft_tools import dominant_frequency
    from .filters import filter_signal
    from .utils import generate_multitone
    from .wavelet import denoise
except ImportError:
    from fft_tools import dominant_frequency
    from filters import filter_signal
    from utils import generate_multitone
    from wavelet import denoise


def fft_demo() -> float:
    """Generate a two-tone signal and return its dominant frequency."""
    _, signal = generate_multitone([50, 120], sample_rate=1000, duration=1.0, amplitudes=[1.0, 0.4])
    return dominant_frequency(signal, sample_rate=1000, min_frequency=1)


def filter_demo():
    """Low-pass filter a noisy multitone signal."""
    t, signal = generate_multitone([20, 200], sample_rate=1000, duration=1.0, amplitudes=[1.0, 0.5], noise_std=0.1, seed=7)
    filtered = filter_signal(signal, sample_rate=1000, cutoff=80, kind="lowpass", method="fir")
    return t, signal, filtered


def wavelet_demo():
    """Denoise a noisy signal with wavelet thresholding."""
    t, signal = generate_multitone([15, 70], sample_rate=500, duration=1.0, noise_std=0.3, seed=3)
    cleaned = denoise(signal)
    return t, signal, cleaned


def main() -> None:
    """Run all demos and print compact results."""
    print(f"Dominant FFT frequency: {fft_demo():.2f} Hz")
    _, raw, filtered = filter_demo()
    print(f"Filter demo samples: raw={len(raw)}, filtered={len(filtered)}")
    _, noisy, cleaned = wavelet_demo()
    print(f"Wavelet demo samples: noisy={len(noisy)}, cleaned={len(cleaned)}")


if __name__ == "__main__":
    main()
