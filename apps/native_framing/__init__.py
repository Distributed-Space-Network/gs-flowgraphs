"""Engine-independent native satellite framing contracts.

The package is deliberately importable without GNU Radio.  Protocol profiles
compose repository-owned streaming decoders and expose one result shape to the
live and post-pass applications.
"""

from .afsk import AfskConfig, AfskDecodedFrame, AfskSymbols, decode_afsk_profile, demodulate_afsk
from .fsk_audio import (
    FskAudioConfig,
    FskAudioDecodedFrame,
    FskAudioSymbols,
    decode_fsk_audio_mm_profile,
    decode_fsk_audio_profile,
    demodulate_fsk_audio,
    demodulate_fsk_audio_mm,
)
from .psk import BpskConfig, BpskSymbols, demodulate_bpsk, manchester_sync_symbols
from .psk_audio import (
    BpskAudioConfig,
    BpskAudioDecodedFrame,
    BpskAudioReplay,
    decode_bpsk_audio_profile,
)
from .registry import advertised_profiles, build_decoder, resolve_profile
from .sample_clock import SampleClock, convert_offset, select_channel_rate
from .types import (
    DecodeDisposition,
    FrameResult,
    FramingProfile,
    IntegrityStatus,
    Polarity,
    SymbolInput,
)
from .viterbi import (
    CONVENTIONS,
    ConvolutionalCode,
    StreamingViterbiDecoder,
    ViterbiResult,
    decode_hypotheses,
)

__all__ = [
    "DecodeDisposition",
    "AfskConfig",
    "AfskDecodedFrame",
    "AfskSymbols",
    "BpskConfig",
    "BpskAudioConfig",
    "BpskAudioDecodedFrame",
    "BpskAudioReplay",
    "BpskSymbols",
    "CONVENTIONS",
    "ConvolutionalCode",
    "FrameResult",
    "FramingProfile",
    "FskAudioConfig",
    "FskAudioDecodedFrame",
    "FskAudioSymbols",
    "IntegrityStatus",
    "Polarity",
    "SampleClock",
    "SymbolInput",
    "StreamingViterbiDecoder",
    "ViterbiResult",
    "advertised_profiles",
    "build_decoder",
    "convert_offset",
    "decode_afsk_profile",
    "decode_bpsk_audio_profile",
    "decode_fsk_audio_profile",
    "decode_fsk_audio_mm_profile",
    "demodulate_afsk",
    "demodulate_fsk_audio",
    "demodulate_fsk_audio_mm",
    "demodulate_bpsk",
    "manchester_sync_symbols",
    "resolve_profile",
    "select_channel_rate",
    "decode_hypotheses",
]
