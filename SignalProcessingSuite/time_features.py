"""Time-domain feature extraction block for Leibnitz pipelines."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import stats

try:
    from .utils import validate_1d_signal
    from .blocks import ProcessingBlock, BlockRunResult
except ImportError:
    from utils import validate_1d_signal
    from blocks import ProcessingBlock, BlockRunResult


class TimeFeatureBlock(ProcessingBlock):
    """Extract statistical and temporal features from 1D signals."""

    id = "time_features"
    name = "Time-Domain Features"
    category = "feature-extraction"
    input_schema = {"signal": "1d-array", "sample_rate": "positive-float"}
    output_schema = {
        "mean": "float",
        "median": "float",
        "std": "float",
        "variance": "float",
        "rms": "float",
        "peak_to_peak": "float",
        "min": "float",
        "max": "float",
        "skewness": "float",
        "kurtosis": "float",
        "zero_crossing_rate": "float",
        "crest_factor": "float",
    }
    default_params = {}

    def validate(self, params: dict[str, Any], sample_rate: float) -> dict[str, Any]:
        """Validate parameters. TimeFeatureBlock has no configurable params."""
        merged = super().validate(params, sample_rate)
        return merged

    def run(
        self, signal: Iterable[float], sample_rate: float, params: dict[str, Any]
    ) -> BlockRunResult:
        """
        Extract time-domain features from signal.

        Args:
            signal: 1D signal array
            sample_rate: Sample rate in Hz (required by interface, not used for features)
            params: Block parameters (unused for this block)

        Returns:
            BlockRunResult with feature dict in result, no output_signal.
        """
        params = self.validate(params, sample_rate)

        # Validate and convert signal
        data = validate_1d_signal(signal, name="signal")
        warnings = []

        # Compute features
        features = {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "variance": float(np.var(data)),
            "rms": float(np.sqrt(np.mean(data**2))),
            "peak_to_peak": float(np.max(data) - np.min(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data)),
            "zero_crossing_rate": float(self._zero_crossing_rate(data)),
            "crest_factor": float(self._crest_factor(data)),
        }

        # Quality checks
        if features["std"] == 0.0:
            warnings.append("Signal is flat (zero variance). Features may not be meaningful.")
        if features["peak_to_peak"] / (features["rms"] + 1e-10) > 10:
            warnings.append("Signal has very high crest factor. May contain clipping or outliers.")
        if np.any(np.abs(data) > 1e6):
            warnings.append("Signal contains very large values. Consider scaling.")

        return BlockRunResult(
            result=features,
            output_signal=None,  # Features don't modify the signal
            plot_data={},  # Visualization handled separately
            warnings=warnings,
            metadata={
                "sample_rate": sample_rate,
                "sample_count": len(data),
                "duration_seconds": len(data) / sample_rate,
            },
        )

    @staticmethod
    def _zero_crossing_rate(signal: np.ndarray) -> float:
        """
        Compute zero-crossing rate: fraction of samples where sign changes.

        Returns:
            ZCR in [0, 1], where 1 means alternating positive/negative.
        """
        if signal.size < 2:
            return 0.0
        sign_changes = np.abs(np.diff(np.sign(signal)))
        return float(np.mean(sign_changes) / 2.0)

    @staticmethod
    def _crest_factor(signal: np.ndarray) -> float:
        """
        Compute crest factor: peak amplitude / RMS.

        Returns:
            Ratio >= 1. For sine wave, ~1.41. For impulse, can be >> 1.
        """
        rms = np.sqrt(np.mean(signal**2))
        if rms == 0:
            return 0.0
        return float(np.max(np.abs(signal)) / rms)
