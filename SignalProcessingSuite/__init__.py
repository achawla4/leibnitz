"""Python Signal Processing Suite."""

try:
    from .fft_tools import (
        apply_window,
        dominant_frequency,
        fft,
        inverse_fft,
        magnitude_spectrum,
        power_spectrum,
        short_time_fft,
        spectral_centroid,
    )
    from .filters import (
        apply_fir,
        apply_iir,
        design_butterworth,
        design_chebyshev,
        design_fir,
        filter_signal,
        moving_average,
    )
    from .realtime_processing import RingBuffer, chunk_stream, realtime_fft_processor, sliding_windows, stream_process
    from .utils import add_noise, generate_multitone, generate_sine, normalize, resample_signal, time_vector
    from .wavelet import compress, cwt, denoise, dwt, haar_dwt, haar_idwt, idwt
except ImportError:
    from fft_tools import (
        apply_window,
        dominant_frequency,
        fft,
        inverse_fft,
        magnitude_spectrum,
        power_spectrum,
        short_time_fft,
        spectral_centroid,
    )
    from filters import (
        apply_fir,
        apply_iir,
        design_butterworth,
        design_chebyshev,
        design_fir,
        filter_signal,
        moving_average,
    )
    from realtime_processing import RingBuffer, chunk_stream, realtime_fft_processor, sliding_windows, stream_process
    from utils import add_noise, generate_multitone, generate_sine, normalize, resample_signal, time_vector
    from wavelet import compress, cwt, denoise, dwt, haar_dwt, haar_idwt, idwt

__all__ = [
    "RingBuffer",
    "add_noise",
    "apply_fir",
    "apply_iir",
    "apply_window",
    "chunk_stream",
    "compress",
    "cwt",
    "denoise",
    "design_butterworth",
    "design_chebyshev",
    "design_fir",
    "dominant_frequency",
    "dwt",
    "fft",
    "filter_signal",
    "generate_multitone",
    "generate_sine",
    "haar_dwt",
    "haar_idwt",
    "idwt",
    "inverse_fft",
    "magnitude_spectrum",
    "moving_average",
    "normalize",
    "power_spectrum",
    "realtime_fft_processor",
    "resample_signal",
    "short_time_fft",
    "sliding_windows",
    "spectral_centroid",
    "stream_process",
    "time_vector",
]
