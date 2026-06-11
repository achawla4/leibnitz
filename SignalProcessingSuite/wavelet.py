"""Wavelet transforms, denoising, and compression helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from .utils import validate_1d_signal
except ImportError:
    from utils import validate_1d_signal


def haar_dwt(signal: Iterable[float], levels: int = 1) -> list[np.ndarray]:
    """Compute a simple multilevel Haar discrete wavelet transform."""
    data = validate_1d_signal(signal)
    if levels <= 0:
        raise ValueError("levels must be positive")
    coeffs: list[np.ndarray] = []
    approx = data.copy()
    for _ in range(levels):
        if approx.size < 2:
            break
        if approx.size % 2:
            approx = np.pad(approx, (0, 1), mode="edge")
        even = approx[0::2]
        odd = approx[1::2]
        coeffs.insert(0, (even - odd) / np.sqrt(2.0))
        approx = (even + odd) / np.sqrt(2.0)
    coeffs.insert(0, approx)
    return coeffs


def haar_idwt(coeffs: list[np.ndarray], original_length: int | None = None) -> np.ndarray:
    """Reconstruct a signal from coefficients returned by haar_dwt."""
    if len(coeffs) < 2:
        raise ValueError("coeffs must contain approximation and at least one detail array")
    approx = np.asarray(coeffs[0], dtype=float)
    for detail in coeffs[1:]:
        detail = np.asarray(detail, dtype=float)
        if approx.shape != detail.shape:
            size = min(approx.size, detail.size)
            approx, detail = approx[:size], detail[:size]
        even = (approx + detail) / np.sqrt(2.0)
        odd = (approx - detail) / np.sqrt(2.0)
        reconstructed = np.empty(even.size + odd.size, dtype=float)
        reconstructed[0::2] = even
        reconstructed[1::2] = odd
        approx = reconstructed
    return approx[:original_length] if original_length else approx


def dwt(signal: Iterable[float], wavelet: str = "db4", levels: int | None = None) -> list[np.ndarray]:
    """Compute a discrete wavelet transform using PyWavelets, with Haar fallback."""
    data = validate_1d_signal(signal)
    try:
        import pywt

        max_level = pywt.dwt_max_level(data.size, pywt.Wavelet(wavelet).dec_len)
        level = max_level if levels is None else min(levels, max_level)
        return pywt.wavedec(data, wavelet, level=level)
    except ImportError:
        return haar_dwt(data, levels or 1)


def idwt(coeffs: list[np.ndarray], wavelet: str = "db4", original_length: int | None = None) -> np.ndarray:
    """Inverse discrete wavelet transform."""
    try:
        import pywt

        reconstructed = pywt.waverec(coeffs, wavelet)
        return reconstructed[:original_length] if original_length else reconstructed
    except ImportError:
        return haar_idwt(coeffs, original_length)


def cwt(
    signal: Iterable[float],
    scales: Iterable[float],
    wavelet: str = "morl",
    sampling_period: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a continuous wavelet transform. Requires PyWavelets."""
    data = validate_1d_signal(signal)
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("PyWavelets is required for continuous wavelet transforms") from exc
    return pywt.cwt(data, np.asarray(list(scales), dtype=float), wavelet, sampling_period=sampling_period)


def threshold_coefficients(coeffs: list[np.ndarray], threshold: float, mode: str = "soft") -> list[np.ndarray]:
    """Threshold detail coefficients while preserving the approximation band."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    output = [np.asarray(coeffs[0], dtype=float)]
    for detail in coeffs[1:]:
        arr = np.asarray(detail, dtype=float)
        if mode == "soft":
            output.append(np.sign(arr) * np.maximum(np.abs(arr) - threshold, 0.0))
        elif mode == "hard":
            output.append(arr * (np.abs(arr) >= threshold))
        else:
            raise ValueError("mode must be soft or hard")
    return output


def denoise(signal: Iterable[float], wavelet: str = "db4", levels: int | None = None, threshold: float | None = None) -> np.ndarray:
    """Denoise a signal with wavelet thresholding."""
    data = validate_1d_signal(signal)
    coeffs = dwt(data, wavelet=wavelet, levels=levels)
    if threshold is None:
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if detail.size else 0.0
        threshold = sigma * np.sqrt(2.0 * np.log(data.size))
    return idwt(threshold_coefficients(coeffs, threshold), wavelet=wavelet, original_length=data.size)


def compress(signal: Iterable[float], keep_ratio: float = 0.1, wavelet: str = "db4") -> tuple[np.ndarray, list[np.ndarray]]:
    """Keep the largest wavelet coefficients and reconstruct the signal."""
    data = validate_1d_signal(signal)
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in the range (0, 1]")
    coeffs = dwt(data, wavelet=wavelet)
    flat = np.concatenate([np.ravel(c) for c in coeffs])
    cutoff = np.quantile(np.abs(flat), 1.0 - keep_ratio)
    compressed = [np.where(np.abs(c) >= cutoff, c, 0.0) for c in coeffs]
    return idwt(compressed, wavelet=wavelet, original_length=data.size), compressed
