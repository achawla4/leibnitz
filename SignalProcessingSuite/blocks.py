"""Composable processing blocks for Leibnitz pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable

import numpy as np

try:
    from .fft_tools import magnitude_spectrum
    from .filters import filter_signal
    from .wavelet import dwt
except ImportError:
    from fft_tools import magnitude_spectrum
    from filters import filter_signal
    from wavelet import dwt


@dataclass
class BlockRunResult:
    """Result returned by a processing block."""

    result: dict[str, Any]
    output_signal: np.ndarray | None = None
    plot_data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ProcessingBlock:
    """Base class for registry-backed signal-processing blocks."""

    id = ""
    name = ""
    category = "analysis"
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    default_params: dict[str, Any] = {}

    def validate(self, params: dict[str, Any], sample_rate: float) -> dict[str, Any]:
        merged = dict(self.default_params)
        merged.update(params or {})
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return merged

    def run(self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> BlockRunResult:
        raise NotImplementedError


class FFTBlock(ProcessingBlock):
    id = "fft"
    name = "FFT Analysis"
    category = "frequency"
    input_schema = {"signal": "1d-array", "sample_rate": "positive-float"}
    output_schema = {"frequencies": "array", "magnitude": "array"}
    default_params = {"fft_window": "none"}

    def run(self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> BlockRunResult:
        params = self.validate(params, sample_rate)
        window_type = params.get("fft_window", "none")
        if window_type == "none" or not window_type:
            window_type = None

        freqs, magnitude = magnitude_spectrum(signal, sample_rate=sample_rate, window=window_type)
        return BlockRunResult(
            result={
                "frequencies": freqs.tolist()[:500],
                "magnitude": magnitude.tolist()[:500],
            },
            plot_data={"frequencies": freqs, "magnitude": magnitude},
            metadata={"window": window_type or "none"},
        )


class FilterBlock(ProcessingBlock):
    id = "filter"
    name = "Filter"
    category = "conditioning"
    input_schema = {"signal": "1d-array", "sample_rate": "positive-float"}
    output_schema = {"filtered": "array"}
    default_params = {
        "filter_kind": "lowpass",
        "filter_method": "butter",
        "filter_order": 4,
        "filter_cutoff": 100.0,
    }

    def validate(self, params: dict[str, Any], sample_rate: float) -> dict[str, Any]:
        merged = super().validate(params, sample_rate)
        merged["filter_kind"] = merged.get("filter_kind") or "lowpass"
        merged["filter_method"] = merged.get("filter_method") or "butter"
        try:
            merged["filter_order"] = int(merged.get("filter_order", 4))
        except (TypeError, ValueError):
            merged["filter_order"] = 4
        if merged["filter_order"] <= 0:
            raise ValueError("filter_order must be positive")
        merged["cutoff"] = self._parse_cutoff(merged.get("filter_cutoff"), merged["filter_kind"], sample_rate)
        return merged

    def _parse_cutoff(self, raw_cutoff: Any, filter_kind: str, sample_rate: float) -> float | tuple[float, float]:
        nyquist = sample_rate / 2.0
        if filter_kind in ("bandpass", "bandstop"):
            if isinstance(raw_cutoff, (list, tuple)) and len(raw_cutoff) == 2:
                cutoff = (float(raw_cutoff[0]), float(raw_cutoff[1]))
            elif isinstance(raw_cutoff, str) and "," in raw_cutoff:
                parts = raw_cutoff.split(",", 1)
                cutoff = (float(parts[0].strip()), float(parts[1].strip()))
            else:
                cutoff = (0.1 * nyquist, 0.5 * nyquist)
            if not 0 < cutoff[0] < cutoff[1] < nyquist:
                raise ValueError("band cutoff must contain two increasing values below Nyquist")
            return cutoff

        try:
            cutoff_value = float(raw_cutoff) if raw_cutoff is not None else 100.0
        except (TypeError, ValueError):
            cutoff_value = 100.0
        if not 0 < cutoff_value < nyquist:
            raise ValueError("filter_cutoff must be between 0 and Nyquist")
        return cutoff_value

    def run(self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> BlockRunResult:
        params = self.validate(params, sample_rate)
        filtered = filter_signal(
            signal,
            sample_rate=sample_rate,
            cutoff=params["cutoff"],
            kind=params["filter_kind"],
            method=params["filter_method"],
            order=params["filter_order"],
        )
        return BlockRunResult(
            result={"filtered": filtered.tolist()[:1000]},
            output_signal=filtered,
            plot_data={"filtered": filtered},
            metadata={
                "kind": params["filter_kind"],
                "method": params["filter_method"],
                "order": params["filter_order"],
                "cutoff": params["cutoff"],
            },
        )


class WaveletBlock(ProcessingBlock):
    id = "wavelet"
    name = "Wavelet Transform"
    category = "time-frequency"
    input_schema = {"signal": "1d-array", "sample_rate": "positive-float"}
    output_schema = {"coefficients_summary": "array"}
    default_params = {"wavelet_type": "db4", "wavelet_levels": 3}

    def validate(self, params: dict[str, Any], sample_rate: float) -> dict[str, Any]:
        merged = super().validate(params, sample_rate)
        merged["wavelet_type"] = merged.get("wavelet_type") or "db4"
        try:
            merged["wavelet_levels"] = int(merged.get("wavelet_levels", 3))
        except (TypeError, ValueError):
            merged["wavelet_levels"] = 3
        if merged["wavelet_levels"] <= 0:
            raise ValueError("wavelet_levels must be positive")
        return merged

    def run(self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> BlockRunResult:
        params = self.validate(params, sample_rate)
        coeffs = dwt(signal, wavelet=params["wavelet_type"], levels=params["wavelet_levels"])
        summary = [
            {
                "band": "Approximation" if index == 0 else f"Detail Level {len(coeffs) - index}",
                "size": len(coeff),
                "mean": float(np.mean(coeff)),
                "std": float(np.std(coeff)),
                "max": float(np.max(coeff)),
                "min": float(np.min(coeff)),
            }
            for index, coeff in enumerate(coeffs)
        ]
        return BlockRunResult(
            result={
                "wavelet": params["wavelet_type"],
                "levels": len(coeffs) - 1,
                "coefficients_summary": summary,
            },
            plot_data={"_coeffs_obj": coeffs},
            metadata={"wavelet": params["wavelet_type"], "requested_levels": params["wavelet_levels"]},
        )


BLOCK_REGISTRY: dict[str, ProcessingBlock] = {
    block.id: block
    for block in (
        FFTBlock(),
        FilterBlock(),
        WaveletBlock(),
    )
}


def get_block(block_id: str) -> ProcessingBlock:
    try:
        return BLOCK_REGISTRY[block_id]
    except KeyError as exc:
        raise ValueError(f"unknown processing block: {block_id}") from exc


def list_blocks() -> list[dict[str, Any]]:
    return [
        {
            "id": block.id,
            "name": block.name,
            "category": block.category,
            "input_schema": block.input_schema,
            "output_schema": block.output_schema,
            "default_params": block.default_params,
        }
        for block in BLOCK_REGISTRY.values()
    ]


def timed_run(block: ProcessingBlock, signal: Iterable[float], sample_rate: float, params: dict[str, Any]) -> tuple[BlockRunResult, float]:
    start = perf_counter()
    run_result = block.run(signal, sample_rate, params)
    elapsed_ms = (perf_counter() - start) * 1000.0
    return run_result, elapsed_ms
