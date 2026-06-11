"""Common helpers for signal generation and preprocessing."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple logging format for examples and CLI use."""
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def validate_1d_signal(signal: Iterable[float], name: str = "signal") -> np.ndarray:
    """Return *signal* as a finite one-dimensional float array."""
    data = np.asarray(signal, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if data.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return data


def normalize(signal: Iterable[float], mode: str = "peak") -> np.ndarray:
    """Normalize a signal by peak amplitude, RMS, or z-score."""
    data = validate_1d_signal(signal)
    if mode == "peak":
        scale = np.max(np.abs(data))
        return data if scale == 0 else data / scale
    if mode == "rms":
        scale = np.sqrt(np.mean(data**2))
        return data if scale == 0 else data / scale
    if mode == "zscore":
        std = np.std(data)
        return data - np.mean(data) if std == 0 else (data - np.mean(data)) / std
    raise ValueError("mode must be one of: peak, rms, zscore")


def time_vector(duration: float, sample_rate: float, endpoint: bool = False) -> np.ndarray:
    """Create a time vector for a duration and sample rate."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    count = int(round(duration * sample_rate))
    if count <= 0:
        raise ValueError("duration and sample_rate produce no samples")
    return np.linspace(0.0, duration, count, endpoint=endpoint)


def generate_sine(
    frequency: float,
    sample_rate: float,
    duration: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a sine wave and its time vector."""
    if frequency < 0:
        raise ValueError("frequency must be non-negative")
    t = time_vector(duration, sample_rate)
    return t, amplitude * np.sin(2.0 * np.pi * frequency * t + phase)


def generate_multitone(
    frequencies: Iterable[float],
    sample_rate: float,
    duration: float,
    amplitudes: Iterable[float] | None = None,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a sum of sine tones with optional Gaussian noise."""
    freqs = np.asarray(list(frequencies), dtype=float)
    if freqs.size == 0:
        raise ValueError("frequencies must not be empty")
    amps = np.ones_like(freqs) if amplitudes is None else np.asarray(list(amplitudes), dtype=float)
    if amps.shape != freqs.shape:
        raise ValueError("amplitudes must match frequencies")
    t = time_vector(duration, sample_rate)
    signal = np.zeros_like(t)
    for freq, amp in zip(freqs, amps):
        if freq < 0:
            raise ValueError("frequencies must be non-negative")
        signal += amp * np.sin(2.0 * np.pi * freq * t)
    if noise_std > 0:
        signal += np.random.default_rng(seed).normal(0.0, noise_std, size=t.size)
    return t, signal


def resample_signal(signal: Iterable[float], original_rate: float, target_rate: float) -> np.ndarray:
    """Resample a signal using SciPy when available, otherwise linear interpolation."""
    data = validate_1d_signal(signal)
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    target_length = int(round(data.size * target_rate / original_rate))
    if target_length <= 0:
        raise ValueError("target_rate produces no samples")
    try:
        from scipy.signal import resample

        return resample(data, target_length)
    except ImportError:
        logger.warning("SciPy not installed; using linear interpolation for resampling")
        old_x = np.linspace(0.0, 1.0, data.size)
        new_x = np.linspace(0.0, 1.0, target_length)
        return np.interp(new_x, old_x, data)


def add_noise(signal: Iterable[float], noise_std: float, seed: int | None = None) -> np.ndarray:
    """Add Gaussian noise to a signal."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    data = validate_1d_signal(signal)
    return data + np.random.default_rng(seed).normal(0.0, noise_std, size=data.size)
