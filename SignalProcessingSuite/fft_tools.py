"""FFT, inverse FFT, and spectral analysis helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from .utils import validate_1d_signal
except ImportError:
    from utils import validate_1d_signal


def get_window(name: str, size: int) -> np.ndarray:
    """Return a common analysis window."""
    if size <= 0:
        raise ValueError("size must be positive")
    key = name.lower()
    if key in {"rect", "rectangular", "boxcar", "none"}:
        return np.ones(size)
    if key in {"hann", "hanning"}:
        return np.hanning(size)
    if key == "hamming":
        return np.hamming(size)
    if key == "blackman":
        return np.blackman(size)
    if key == "bartlett":
        return np.bartlett(size)
    raise ValueError("unsupported window: use rectangular, hann, hamming, blackman, or bartlett")


def apply_window(signal: Iterable[float], window: str | np.ndarray = "hann") -> np.ndarray:
    """Apply a named or explicit window to a signal."""
    data = validate_1d_signal(signal)
    weights = get_window(window, data.size) if isinstance(window, str) else np.asarray(window, dtype=float)
    if weights.shape != data.shape:
        raise ValueError("window must have the same shape as signal")
    return data * weights


def fft(signal: Iterable[float], sample_rate: float, window: str | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute a one-sided real FFT and return frequencies and complex bins."""
    data = validate_1d_signal(signal)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if window is not None:
        data = apply_window(data, window)
    return np.fft.rfftfreq(data.size, d=1.0 / sample_rate), np.fft.rfft(data)


def inverse_fft(spectrum: Iterable[complex], n_samples: int | None = None) -> np.ndarray:
    """Reconstruct a real signal from a one-sided FFT spectrum."""
    bins = np.asarray(spectrum, dtype=complex)
    if bins.ndim != 1 or bins.size == 0:
        raise ValueError("spectrum must be a non-empty one-dimensional array")
    return np.fft.irfft(bins, n=n_samples)


def magnitude_spectrum(
    signal: Iterable[float],
    sample_rate: float,
    window: str | np.ndarray | None = "hann",
    db: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency bins and amplitude spectrum."""
    data = validate_1d_signal(signal)
    freqs, spectrum = fft(data, sample_rate, window)
    magnitude = np.abs(spectrum) / data.size
    if data.size > 1:
        magnitude[1:-1] *= 2.0
    if db:
        magnitude = 20.0 * np.log10(np.maximum(magnitude, np.finfo(float).eps))
    return freqs, magnitude


def power_spectrum(
    signal: Iterable[float],
    sample_rate: float,
    window: str | np.ndarray | None = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency bins and power spectrum."""
    freqs, magnitude = magnitude_spectrum(signal, sample_rate, window, db=False)
    return freqs, magnitude**2


def dominant_frequency(signal: Iterable[float], sample_rate: float, min_frequency: float = 0.0) -> float:
    """Estimate the strongest frequency component in a signal."""
    freqs, magnitude = magnitude_spectrum(signal, sample_rate)
    mask = freqs >= min_frequency
    if not np.any(mask):
        raise ValueError("min_frequency is above the available frequency range")
    return float(freqs[mask][np.argmax(magnitude[mask])])


def spectral_centroid(signal: Iterable[float], sample_rate: float) -> float:
    """Compute the spectral centroid in Hz."""
    freqs, magnitude = magnitude_spectrum(signal, sample_rate)
    total = np.sum(magnitude)
    return 0.0 if total == 0 else float(np.sum(freqs * magnitude) / total)


def short_time_fft(
    signal: Iterable[float],
    sample_rate: float,
    frame_size: int,
    hop_size: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a simple short-time FFT matrix."""
    data = validate_1d_signal(signal)
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if frame_size > data.size:
        raise ValueError("frame_size must be no larger than signal length")
    frames = []
    for start in range(0, data.size - frame_size + 1, hop_size):
        frames.append(np.fft.rfft(apply_window(data[start : start + frame_size], window)))
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    times = np.arange(len(frames)) * hop_size / sample_rate
    return freqs, times, np.asarray(frames).T
