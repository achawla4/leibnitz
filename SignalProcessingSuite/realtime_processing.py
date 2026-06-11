"""Streaming signal-processing primitives."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Iterator

import numpy as np

try:
    from .utils import validate_1d_signal
except ImportError:
    from utils import validate_1d_signal


class RingBuffer:
    """Fixed-size numeric ring buffer for streaming samples."""

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self._buffer: deque[float] = deque(maxlen=size)

    def append(self, samples: float | Iterable[float]) -> None:
        """Append one sample or many samples."""
        if np.isscalar(samples):
            self._buffer.append(float(samples))
        else:
            self._buffer.extend(float(x) for x in samples)

    def clear(self) -> None:
        """Clear buffered samples."""
        self._buffer.clear()

    def is_full(self) -> bool:
        """Return True when the buffer contains size samples."""
        return len(self._buffer) == self.size

    def to_array(self, pad: bool = False) -> np.ndarray:
        """Return buffer contents, optionally left-padding with zeros."""
        data = np.asarray(self._buffer, dtype=float)
        if pad and data.size < self.size:
            data = np.pad(data, (self.size - data.size, 0))
        return data


def sliding_windows(signal: Iterable[float], window_size: int, hop_size: int) -> Iterator[np.ndarray]:
    """Yield overlapping windows from a finite signal."""
    data = validate_1d_signal(signal)
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive")
    for start in range(0, data.size - window_size + 1, hop_size):
        yield data[start : start + window_size]


def chunk_stream(samples: Iterable[float], chunk_size: int) -> Iterator[np.ndarray]:
    """Yield fixed-size chunks from any iterable of samples."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunk: list[float] = []
    for sample in samples:
        chunk.append(float(sample))
        if len(chunk) == chunk_size:
            yield np.asarray(chunk, dtype=float)
            chunk = []
    if chunk:
        yield np.asarray(chunk, dtype=float)


def stream_process(
    samples: Iterable[float],
    processor: Callable[[np.ndarray], np.ndarray | float],
    window_size: int,
    hop_size: int,
) -> Iterator[np.ndarray | float]:
    """Apply a processor to rolling windows from a stream."""
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive")
    buffer = RingBuffer(window_size)
    pending = 0
    for sample in samples:
        buffer.append(float(sample))
        pending += 1
        if buffer.is_full() and pending >= hop_size:
            yield processor(buffer.to_array())
            pending = 0


def realtime_fft_processor(sample_rate: float, min_frequency: float = 0.0) -> Callable[[np.ndarray], float]:
    """Create a processor that emits dominant frequency for each window."""
    try:
        from .fft_tools import dominant_frequency
    except ImportError:
        from fft_tools import dominant_frequency

    def process(window: np.ndarray) -> float:
        return dominant_frequency(window, sample_rate, min_frequency=min_frequency)

    return process
