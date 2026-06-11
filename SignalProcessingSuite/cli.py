"""Command-line tools for quick signal processing tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .fft_tools import dominant_frequency
    from .filters import filter_signal
    from .utils import generate_multitone
except ImportError:
    from fft_tools import dominant_frequency
    from filters import filter_signal
    from utils import generate_multitone


def _load_signal(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    if data.ndim == 2:
        data = data[:, -1]
    return np.asarray(data, dtype=float)


def _save_signal(path: Path, signal: np.ndarray) -> None:
    np.savetxt(path, signal, delimiter="," if path.suffix.lower() == ".csv" else " ")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Signal Processing Suite CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tone = subparsers.add_parser("tone", help="Generate a multitone signal")
    tone.add_argument("output", type=Path)
    tone.add_argument("--freq", type=float, action="append", required=True, help="Tone frequency in Hz; repeatable")
    tone.add_argument("--sample-rate", type=float, default=1000.0)
    tone.add_argument("--duration", type=float, default=1.0)
    tone.add_argument("--noise-std", type=float, default=0.0)

    dom = subparsers.add_parser("dominant", help="Estimate dominant frequency")
    dom.add_argument("input", type=Path)
    dom.add_argument("--sample-rate", type=float, required=True)
    dom.add_argument("--min-frequency", type=float, default=0.0)

    filt = subparsers.add_parser("filter", help="Apply a simple filter")
    filt.add_argument("input", type=Path)
    filt.add_argument("output", type=Path)
    filt.add_argument("--sample-rate", type=float, required=True)
    filt.add_argument("--cutoff", type=float, required=True)
    filt.add_argument("--kind", choices=["lowpass", "highpass"], default="lowpass")
    filt.add_argument("--method", choices=["butter", "cheby1", "fir"], default="butter")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    if args.command == "tone":
        _, signal = generate_multitone(args.freq, args.sample_rate, args.duration, noise_std=args.noise_std)
        _save_signal(args.output, signal)
    elif args.command == "dominant":
        signal = _load_signal(args.input)
        print(dominant_frequency(signal, args.sample_rate, args.min_frequency))
    elif args.command == "filter":
        signal = _load_signal(args.input)
        filtered = filter_signal(signal, args.sample_rate, args.cutoff, kind=args.kind, method=args.method)
        _save_signal(args.output, filtered)


if __name__ == "__main__":
    main()
