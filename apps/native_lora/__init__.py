"""Engine-independent LoRa receive primitives.

The package is intentionally separate from the advertised satellite framing
registry until a mission profile and independent capture satisfy NF-FRM-030.
"""

from .framing import (
    LoRaDecodeResult,
    LoRaFrameError,
    LoRaHeader,
    LoRaIntegrityError,
    LoRaPhyConfig,
    decode_lora_symbols,
    deinterleave_block,
    gray_map_symbol,
    hamming_decode_codeword,
    lora_payload_crc,
)
from .modem import (
    LoRaDemodulatedSymbols,
    LoRaModemConfig,
    LoRaSyncBuffer,
    LoRaSyncError,
    LoRaSyncResult,
    build_upchirp,
    demodulate_payload_bins,
    find_lora_sync,
    normalize_framing_symbols,
)
from .profiles import (
    LoRaCatalogSnapshot,
    LoRaMissionProfile,
    LoRaProfileError,
    catalog_sha256,
    load_catalog_snapshot,
    normalize_sync_word,
    resolve_lora_profile,
)

__all__ = [
    "LoRaDecodeResult",
    "LoRaFrameError",
    "LoRaHeader",
    "LoRaIntegrityError",
    "LoRaPhyConfig",
    "decode_lora_symbols",
    "deinterleave_block",
    "gray_map_symbol",
    "hamming_decode_codeword",
    "lora_payload_crc",
    "LoRaDemodulatedSymbols",
    "LoRaModemConfig",
    "LoRaSyncBuffer",
    "LoRaSyncError",
    "LoRaSyncResult",
    "build_upchirp",
    "demodulate_payload_bins",
    "find_lora_sync",
    "normalize_framing_symbols",
    "LoRaCatalogSnapshot",
    "LoRaMissionProfile",
    "LoRaProfileError",
    "catalog_sha256",
    "load_catalog_snapshot",
    "normalize_sync_word",
    "resolve_lora_profile",
]
