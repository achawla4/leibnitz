"""Plotting helpers for time, frequency, spectrogram, and wavelet views."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from .fft_tools import magnitude_spectrum, short_time_fft
    from .utils import validate_1d_signal
except ImportError:
    from fft_tools import magnitude_spectrum, short_time_fft
    from utils import validate_1d_signal


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Matplotlib is required for visualization") from exc
    return plt


def plot_time(signal: Iterable[float], sample_rate: float, ax=None, title: str = "Time Domain"):
    """Plot amplitude over time and return the Matplotlib axis."""
    data = validate_1d_signal(signal)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    times = np.arange(data.size) / sample_rate
    ax.plot(times, data)
    ax.set(title=title, xlabel="Time (s)", ylabel="Amplitude")
    return ax


def plot_frequency(signal: Iterable[float], sample_rate: float, ax=None, db: bool = False, title: str = "Frequency Domain"):
    """Plot a one-sided magnitude spectrum and return the Matplotlib axis."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    freqs, magnitude = magnitude_spectrum(signal, sample_rate, db=db)
    ax.plot(freqs, magnitude)
    ax.set(title=title, xlabel="Frequency (Hz)", ylabel="Magnitude (dB)" if db else "Magnitude")
    return ax

def plot_ifft(signal: Iterable[float], sample_rate: float, ax=None, db: bool = False, title: str = "Time Domain IFFT"):
    """Plot a timedomain ifft signal and return the Matplotlib axis."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    timesamples = np.real(np.fft.ifft(signal))
    ax.plot(np.arange(len(timesamples)), timesamples)
    ax.set(title=title, xlabel="Time (s)", ylabel="Amplitude")
    return ax



def plot_spectrogram(
    signal: Iterable[float],
    sample_rate: float,
    frame_size: int = 1024,
    hop_size: int = 256,
    ax=None,
):
    """Plot a short-time FFT spectrogram and return the Matplotlib axis."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    freqs, times, stft = short_time_fft(signal, sample_rate, frame_size, hop_size)
    image = 20.0 * np.log10(np.maximum(np.abs(stft), np.finfo(float).eps))
    mesh = ax.pcolormesh(times, freqs, image, shading="auto")
    ax.set(title="Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
    ax.figure.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    return ax


def plot_wavelet_coefficients(coeffs: list[np.ndarray], ax=None):
    """Plot wavelet coefficient arrays as stacked lines."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    offset = 0.0
    for index, coeff in enumerate(coeffs):
        arr = np.asarray(coeff, dtype=float)
        scale = np.max(np.abs(arr)) or 1.0
        ax.plot(arr / scale + offset, label=f"Level {index}")
        offset += 2.0
    ax.set(title="Wavelet Coefficients", xlabel="Coefficient Index", ylabel="Normalized Level")
    ax.legend(loc="upper right")
    return ax


def show() -> None:
    """Show all pending Matplotlib figures."""
    _pyplot().show()
