"""FIR/IIR filter design and application utilities."""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

try:
    from .utils import validate_1d_signal
except ImportError:
    from utils import validate_1d_signal

FilterKind = Literal["lowpass", "highpass", "bandpass", "bandstop"]


def _normalize_cutoff(cutoff: float | tuple[float, float], sample_rate: float) -> float | list[float]:
    nyquist = sample_rate / 2.0
    if isinstance(cutoff, tuple):
        values = [float(c) / nyquist for c in cutoff]
        if len(values) != 2 or not 0 < values[0] < values[1] < 1:
            raise ValueError("band cutoff must be two increasing values below Nyquist")
        return values
    value = float(cutoff) / nyquist
    if not 0 < value < 1:
        raise ValueError("cutoff must be between 0 and Nyquist")
    return value


def design_fir(
    cutoff: float | tuple[float, float],
    sample_rate: float,
    numtaps: int = 101,
    kind: FilterKind = "lowpass",
    window: str = "hamming",
) -> np.ndarray:
    """Design an FIR filter using SciPy or a windowed-sinc fallback."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if numtaps <= 1:
        raise ValueError("numtaps must be greater than 1")
    try:
        from scipy.signal import firwin

        return firwin(numtaps, cutoff, fs=sample_rate, pass_zero=kind, window=window)
    except ImportError:
        if isinstance(cutoff, tuple) or kind not in {"lowpass", "highpass"}:
            raise ImportError("SciPy is required for band FIR filters")
        n = np.arange(numtaps) - (numtaps - 1) / 2.0
        fc = float(cutoff) / sample_rate
        taps = 2.0 * fc * np.sinc(2.0 * fc * n)
        taps *= np.hamming(numtaps) if window == "hamming" else np.ones(numtaps)
        taps /= np.sum(taps)
        if kind == "highpass":
            taps = -taps
            taps[(numtaps - 1) // 2] += 1.0
        return taps


def design_butterworth(
    cutoff: float | tuple[float, float],
    sample_rate: float,
    order: int = 4,
    kind: FilterKind = "lowpass",
) -> tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth IIR filter. Requires SciPy."""
    if order <= 0:
        raise ValueError("order must be positive")
    try:
        from scipy.signal import butter
    except ImportError as exc:
        raise ImportError("SciPy is required for IIR filter design") from exc
    return butter(order, _normalize_cutoff(cutoff, sample_rate), btype=kind)


def design_chebyshev(
    cutoff: float | tuple[float, float],
    sample_rate: float,
    order: int = 4,
    ripple_db: float = 1.0,
    kind: FilterKind = "lowpass",
) -> tuple[np.ndarray, np.ndarray]:
    """Design a Chebyshev type-I IIR filter. Requires SciPy."""
    if order <= 0 or ripple_db <= 0:
        raise ValueError("order and ripple_db must be positive")
    try:
        from scipy.signal import cheby1
    except ImportError as exc:
        raise ImportError("SciPy is required for Chebyshev filter design") from exc
    return cheby1(order, ripple_db, _normalize_cutoff(cutoff, sample_rate), btype=kind)


def apply_fir(signal: Iterable[float], taps: Iterable[float], mode: str = "same") -> np.ndarray:
    """Apply FIR taps with convolution."""
    data = validate_1d_signal(signal)
    coeffs = validate_1d_signal(taps, "taps")
    return np.convolve(data, coeffs, mode=mode)


def apply_iir(signal: Iterable[float], b: Iterable[float], a: Iterable[float], zero_phase: bool = False) -> np.ndarray:
    """Apply an IIR filter using SciPy's lfilter or filtfilt."""
    data = validate_1d_signal(signal)
    try:
        from scipy.signal import filtfilt, lfilter
    except ImportError as exc:
        raise ImportError("SciPy is required for IIR filtering") from exc
    return filtfilt(b, a, data) if zero_phase else lfilter(b, a, data)


def moving_average(signal: Iterable[float], window_size: int) -> np.ndarray:
    """Apply a simple moving-average FIR filter."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    taps = np.ones(window_size) / window_size
    return apply_fir(signal, taps)


def filter_signal(
    signal: Iterable[float],
    sample_rate: float,
    cutoff: float | tuple[float, float],
    kind: FilterKind = "lowpass",
    method: Literal["butter", "cheby1", "fir"] = "butter",
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Design and apply a common filter in one call."""
    if method == "fir":
        taps = design_fir(cutoff, sample_rate, numtaps=max(3, order * 25 + 1), kind=kind)
        return apply_fir(signal, taps)
    if method == "butter":
        b, a = design_butterworth(cutoff, sample_rate, order=order, kind=kind)
    elif method == "cheby1":
        b, a = design_chebyshev(cutoff, sample_rate, order=order, kind=kind)
    else:
        raise ValueError("method must be one of: butter, cheby1, fir")
    return apply_iir(signal, b, a, zero_phase=zero_phase)
