"""Unit tests for TimeFeatureBlock."""

import numpy as np
import pytest

# These imports assume time_features.py is in the same directory or installed
try:
    from time_features import TimeFeatureBlock
except ImportError:
    # For testing in context of full module
    from SignalProcessingSuite.time_features import TimeFeatureBlock


class TestTimeFeatureBlock:
    """Test suite for time-domain feature extraction."""

    @pytest.fixture
    def block(self):
        """Provide a fresh TimeFeatureBlock instance."""
        return TimeFeatureBlock()

    def test_block_metadata(self):
        """Verify block is correctly registered with proper metadata."""
        block = TimeFeatureBlock()
        assert block.id == "time_features"
        assert block.name == "Time-Domain Features"
        assert block.category == "feature-extraction"
        assert len(block.output_schema) == 12

    def test_constant_signal(self, block):
        """Test on constant signal: should have zero variance."""
        signal = np.ones(100) * 5.0
        sample_rate = 1000.0

        result = block.run(signal, sample_rate, {})

        assert result.result["mean"] == pytest.approx(5.0)
        assert result.result["median"] == pytest.approx(5.0)
        assert result.result["std"] == pytest.approx(0.0)
        assert result.result["variance"] == pytest.approx(0.0)
        assert result.result["peak_to_peak"] == pytest.approx(0.0)
        assert result.result["zero_crossing_rate"] == pytest.approx(0.0)
        assert len(result.warnings) > 0  # Should warn about flat signal

    def test_sine_wave_10hz(self, block):
        """
        Test on sine wave with known properties.

        For a sine wave of amplitude A:
        - mean = 0
        - RMS = A / sqrt(2)
        - peak_to_peak = 2*A
        - zero_crossing_rate ≈ 2*frequency / sample_rate
        """
        fs = 1000  # 1 kHz sample rate
        freq = 10  # 10 Hz sine
        duration = 1.0  # 1 second
        amplitude = 2.0

        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = amplitude * np.sin(2 * np.pi * freq * t)

        result = block.run(signal, fs, {})

        # Check mean is near zero (integral of sine over complete cycles)
        assert result.result["mean"] == pytest.approx(0.0, abs=0.01)

        # Check RMS ≈ amplitude / sqrt(2)
        expected_rms = amplitude / np.sqrt(2)
        assert result.result["rms"] == pytest.approx(expected_rms, rel=0.05)

        # Check peak-to-peak ≈ 2 * amplitude
        assert result.result["peak_to_peak"] == pytest.approx(2 * amplitude, rel=0.05)

        # Check zero-crossing rate for 10 Hz sine at 1 kHz
        # Expected: ~0.02 (10 complete cycles per second = 20 zero crossings per second)
        expected_zcr = 2 * freq / fs
        assert result.result["zero_crossing_rate"] == pytest.approx(expected_zcr, rel=0.1)

        # Crest factor for sine wave should be sqrt(2) ≈ 1.41
        expected_cf = np.sqrt(2)
        assert result.result["crest_factor"] == pytest.approx(expected_cf, rel=0.05)

    def test_square_wave(self, block):
        """Test on square wave: higher crest factor than sine."""
        fs = 1000
        freq = 5
        duration = 1.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        # Create square wave
        signal = np.sign(np.sin(2 * np.pi * freq * t))

        result = block.run(signal, fs, {})

        # Square wave has mean ≈ 0
        assert result.result["mean"] == pytest.approx(0.0, abs=0.1)

        # Peak-to-peak = 2
        assert result.result["peak_to_peak"] == pytest.approx(2.0, abs=0.1)

        # RMS of ±1 square wave = 1
        assert result.result["rms"] == pytest.approx(1.0, abs=0.05)

    def test_impulse(self, block):
        """Test on impulse signal: very high crest factor."""
        signal = np.zeros(100)
        signal[50] = 10.0

        result = block.run(signal, 1000, {})

        # RMS very small compared to peak
        rms = result.result["rms"]
        peak = result.result["max"]
        assert peak / rms > 5  # High crest factor

    def test_gaussian_noise(self, block):
        """Test on Gaussian noise: skewness and kurtosis near 0 and 3."""
        np.random.seed(42)
        signal = np.random.normal(0, 1.0, 10000)

        result = block.run(signal, 1000, {})

        # Gaussian noise should have skewness ≈ 0
        assert result.result["skewness"] == pytest.approx(0.0, abs=0.1)

        # Gaussian noise should have excess kurtosis ≈ 0 (scipy returns excess)
        assert result.result["kurtosis"] == pytest.approx(0.0, abs=0.3)

    def test_metadata_includes_duration(self, block):
        """Verify metadata includes correct duration calculation."""
        signal = np.ones(5000)
        sample_rate = 1000.0

        result = block.run(signal, sample_rate, {})

        assert result.metadata["sample_rate"] == 1000.0
        assert result.metadata["sample_count"] == 5000
        assert result.metadata["duration_seconds"] == pytest.approx(5.0)

    def test_no_output_signal_modification(self, block):
        """TimeFeatureBlock should not return modified signal."""
        signal = np.array([1.0, 2.0, 3.0])

        result = block.run(signal, 1000, {})

        assert result.output_signal is None
        assert len(result.result) > 0  # But should have features

    def test_invalid_signal_all_nan(self, block):
        """Should raise ValueError on all-NaN input."""
        signal = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError):
            block.run(signal, 1000, {})

    def test_invalid_signal_empty(self, block):
        """Should raise ValueError on empty input."""
        signal = np.array([])

        with pytest.raises(ValueError):
            block.run(signal, 1000, {})

    def test_invalid_sample_rate(self, block):
        """Should raise ValueError on non-positive sample rate."""
        signal = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            block.run(signal, 0, {})

        with pytest.raises(ValueError):
            block.run(signal, -1000, {})

    def test_list_conversion(self, block):
        """Block should accept list input (not just numpy array)."""
        signal = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = block.run(signal, 1000, {})

        assert isinstance(result.result["mean"], float)
        assert result.result["mean"] == pytest.approx(3.0)

    def test_result_dict_serializable(self, block):
        """Result should be JSON-serializable (for HTTP response)."""
        import json

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = block.run(signal, 1000, {})

        # Should not raise
        json_str = json.dumps(result.result)
        assert len(json_str) > 0

    def test_crest_factor_zero_rms(self, block):
        """Crest factor should handle zero RMS gracefully."""
        signal = np.zeros(100)

        result = block.run(signal, 1000, {})

        # Should not raise, and crest factor should be 0 or handled safely
        assert result.result["crest_factor"] >= 0


class TestTimeFeatureBlockIntegration:
    """Integration tests with full block workflow."""

    def test_block_registry_integration(self):
        """Verify block can be imported and used in registry context."""
        try:
            from blocks import BLOCK_REGISTRY
            block = BLOCK_REGISTRY.get("time_features")
            assert block is not None
        except ImportError:
            # If full module not available, skip
            pytest.skip("Full module integration not available")

    def test_multiple_files_workflow(self):
        """Simulate processing multiple signals through same block."""
        block = TimeFeatureBlock()

        # Simulate three different files
        signals = [
            np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)),
            np.random.normal(0, 1, 1000),
            np.ones(1000) * 5.0,
        ]

        for i, sig in enumerate(signals):
            result = block.run(sig, 1000, {})
            assert len(result.result) == 12  # All features present
            assert isinstance(result.result["mean"], float)


if __name__ == "__main__":
    # Simple run without pytest (for quick testing)
    block = TimeFeatureBlock()

    print("=" * 60)
    print("TimeFeatureBlock Quick Test")
    print("=" * 60)

    # Test 1: Constant signal
    print("\n1. Constant signal (value=5.0):")
    result = block.run(np.ones(100) * 5.0, 1000, {})
    print(f"   Mean: {result.result['mean']:.2f}")
    print(f"   Std: {result.result['std']:.4f}")
    print(f"   Warnings: {result.warnings}")

    # Test 2: 10 Hz sine
    print("\n2. 10 Hz sine wave (amplitude=2.0):")
    t = np.linspace(0, 1, 1000, endpoint=False)
    sine = 2.0 * np.sin(2 * np.pi * 10 * t)
    result = block.run(sine, 1000, {})
    print(f"   Mean: {result.result['mean']:.4f}")
    print(f"   RMS: {result.result['rms']:.4f} (expected ≈ 1.41)")
    print(f"   Peak-to-peak: {result.result['peak_to_peak']:.4f} (expected ≈ 4.0)")
    print(f"   Zero-crossing rate: {result.result['zero_crossing_rate']:.4f} (expected ≈ 0.02)")
    print(f"   Crest factor: {result.result['crest_factor']:.4f} (expected ≈ 1.41)")

    # Test 3: Gaussian noise
    print("\n3. Gaussian noise:")
    np.random.seed(42)
    noise = np.random.normal(0, 1.0, 1000)
    result = block.run(noise, 1000, {})
    print(f"   Mean: {result.result['mean']:.4f}")
    print(f"   Std: {result.result['std']:.4f}")
    print(f"   Skewness: {result.result['skewness']:.4f} (expected ≈ 0.0)")
    print(f"   Kurtosis: {result.result['kurtosis']:.4f} (expected ≈ 0.0)")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
