# Python Signal Processing Suite

A small, modular signal-processing toolkit with helpers for FFT analysis, filtering, wavelets, streaming workflows, plotting, examples, benchmarking, and a command-line interface.

## Files

- `fft_tools.py`: FFT, inverse FFT, magnitude/power spectra, spectral centroid, dominant frequency, and short-time FFT.
- `filters.py`: FIR/IIR filter design and application, Butterworth and Chebyshev helpers, moving average, and one-call filtering.
- `wavelet.py`: Haar fallback transforms plus PyWavelets-backed DWT/CWT, denoising, and compression.
- `realtime_processing.py`: ring buffer, chunking, sliding windows, and streaming processor helpers.
- `utils.py`: signal validation, normalization, resampling, sine/multitone generation, noise, and logging setup.
- `visualization.py`: Matplotlib plots for time-domain, frequency-domain, spectrogram, and wavelet coefficient views.
- `examples.py`: runnable FFT, filtering, and wavelet demonstrations.
- `benchmark.py`: simple runtime benchmark for FFT and filtering.
- `cli.py`: command-line utilities for tone generation, dominant-frequency estimation, and filtering.
- `__init__.py`: high-level package exports.

Per-module unit tests are intentionally not included.

## Requirements

Base functionality requires:

```bash
pip install numpy
```

Optional functionality:

```bash
pip install scipy matplotlib PyWavelets
```

SciPy is required for IIR filters and advanced FIR filter design. Matplotlib is required for visualization. PyWavelets is required for continuous wavelet transforms and non-Haar discrete wavelets.

## Quick Start

```python
from utils import generate_multitone
from fft_tools import dominant_frequency
from filters import filter_signal

sample_rate = 1000
t, signal = generate_multitone([50, 120], sample_rate, duration=1.0, amplitudes=[1.0, 0.4])

print(dominant_frequency(signal, sample_rate, min_frequency=1))

filtered = filter_signal(signal, sample_rate, cutoff=80, kind="lowpass", method="butter")
```

## Examples

Run all demos:

```bash
python examples.py
```

Run the benchmark:

```bash
python benchmark.py
```

## CLI

Generate a tone:

```bash
python cli.py tone tone.csv --freq 50 --freq 120 --sample-rate 1000 --duration 1
```

Estimate dominant frequency:

```bash
python cli.py dominant tone.csv --sample-rate 1000 --min-frequency 1
```

Filter a signal:

```bash
python cli.py filter tone.csv filtered.csv --sample-rate 1000 --cutoff 80 --kind lowpass --method butter
```

## Notes

The modules are plain Python files in the project root, so they can be imported directly when running scripts from this folder. If you later move them into a package directory, keep `__init__.py` with the modules and update imports accordingly.
