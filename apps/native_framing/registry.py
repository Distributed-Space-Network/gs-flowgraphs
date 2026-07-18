"""Declarative registry for the 28 advertised gr-satellites framing labels.

Registration is intentionally separate from runtime availability.  Every
advertised label has exactly one disposition, while only profiles with an
actual repository-owned decoder factory can be built.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from native_framing.profiles.ao40 import build_ao40_uncoded
from native_framing.profiles.ao40_fec import build_ao40_fec, build_ao40_fec_short
from native_framing.profiles.ax25 import build_ax25, build_ax25_g3ruh
from native_framing.profiles.ax100 import (
    build_ax100_asm_golay,
    build_ax100_mode5,
    build_ax100_mode6,
)
from native_framing.profiles.cc11xx import build_aalto1, build_reaktor
from native_framing.profiles.ccsds import (
    build_ccsds_concatenated,
    build_ccsds_rs,
    build_ccsds_uncoded,
)
from native_framing.profiles.fx25 import build_fx25
from native_framing.profiles.geoscan import build_geoscan
from native_framing.profiles.grizu import build_grizu
from native_framing.profiles.mobitex import build_mobitex, build_mobitex_nx
from native_framing.profiles.ngham import build_ngham, build_ngham_no_rs
from native_framing.profiles.openlst import CAPTURE_SIZE as CAPTURE_SIZE_OPENLST
from native_framing.profiles.openlst import build_openlst
from native_framing.profiles.sanosat import build_sanosat
from native_framing.profiles.smogp import build_smogp_signalling
from native_framing.profiles.smogp_ra import build_smogp_ra
from native_framing.profiles.snet import CAPTURE_SIZE as CAPTURE_SIZE_SNET
from native_framing.profiles.snet import build_snet
from native_framing.profiles.tt64 import build_tt64
from native_framing.profiles.u482c import build_u482c
from native_framing.profiles.usp import build_usp
from native_framing.types import (
    DecodeDisposition,
    FramingProfile,
    ParameterSpec,
    Polarity,
    StreamingDecoder,
    SymbolInput,
)

_BOTH_POLARITIES = (Polarity.NORMAL, Polarity.INVERTED)
_MAX_AX25_RETAINED = 4 * (8192 * 10 + 64 + 17)


class ProfileRegistry:
    def __init__(self, profiles: Iterable[FramingProfile]) -> None:
        by_canonical: dict[str, FramingProfile] = {}
        by_alias: dict[str, FramingProfile] = {}
        by_advertised: dict[str, FramingProfile] = {}
        for profile in profiles:
            if profile.canonical in by_canonical:
                raise ValueError(f"duplicate canonical profile: {profile.canonical}")
            by_canonical[profile.canonical] = profile
            if profile.advertised_label in by_advertised:
                raise ValueError(f"duplicate advertised label: {profile.advertised_label}")
            by_advertised[profile.advertised_label] = profile
            for alias in (profile.canonical, profile.advertised_label, *profile.aliases):
                key = alias.casefold()
                prior = by_alias.get(key)
                if prior is not None and prior is not profile:
                    raise ValueError(
                        f"alias {alias!r} is shared by {prior.canonical} and {profile.canonical}"
                    )
                by_alias[key] = profile
        self._by_canonical = by_canonical
        self._by_alias = by_alias
        self._by_advertised = by_advertised

    @property
    def profiles(self) -> tuple[FramingProfile, ...]:
        return tuple(self._by_canonical.values())

    @property
    def advertised(self) -> Mapping[str, FramingProfile]:
        return dict(self._by_advertised)

    def resolve(self, label: object) -> FramingProfile | None:
        key = str(label or "").strip().casefold()
        return self._by_alias.get(key) if key else None

    def build(
        self, label: object, parameters: Mapping[str, object] | None = None
    ) -> StreamingDecoder:
        profile = self.resolve(label)
        if profile is None:
            raise KeyError(f"unknown framing profile: {label!r}")
        if profile.decoder_factory is None:
            raise RuntimeError(
                f"framing profile {profile.advertised_label!r} is {profile.disposition.value}; "
                "no native decoder is available"
            )
        validated = profile.validate_parameters(parameters)
        decoder = profile.decoder_factory(validated)
        if not isinstance(decoder, StreamingDecoder):
            raise TypeError(f"decoder factory for {profile.canonical} violated the stream contract")
        if decoder.max_retained_symbols > profile.max_retained_symbols:
            raise ValueError(f"decoder bound exceeds profile bound for {profile.canonical}")
        return decoder


def _planned(
    canonical: str,
    label: str,
    symbol_input: SymbolInput,
    transforms: tuple[str, ...],
    *,
    aliases: tuple[str, ...] = (),
) -> FramingProfile:
    return FramingProfile(
        canonical=canonical,
        advertised_label=label,
        aliases=aliases,
        disposition=DecodeDisposition.PLANNED,
        symbol_input=symbol_input,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=1,
        sync_policy="specified by the corresponding NF-FRM ledger row before activation",
        frame_length_policy="bounded profile-specific policy required before activation",
        transforms=transforms,
        integrity_policy="profile-specific integrity gate required before activation",
        output_semantics="profile-specific payload contract required before activation",
        live_supported=False,
        post_pass_supported=False,
    )


_AX25_PARAMETER = {
    "max_frame_bytes": ParameterSpec(int, default=4096, minimum=18, maximum=8192),
}

_PROFILES = (
    FramingProfile(
        canonical="ax25",
        advertised_label="AX.25",
        aliases=("AX25", "APRS"),
        disposition=DecodeDisposition.EXISTING_PENDING_PARITY,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=_MAX_AX25_RETAINED,
        sync_policy="HDLC 0x7e flags with exact matching",
        frame_length_policy="18..max_frame_bytes octets including FCS",
        transforms=("optional G3RUH descramble", "NRZI decode", "HDLC bit unstuff"),
        integrity_policy="AX.25/X.25 FCS; address sanity is separate station-policy metadata",
        output_semantics="AX.25 body with FCS removed",
        live_supported=True,
        post_pass_supported=True,
        parameters=_AX25_PARAMETER,
        decoder_factory=build_ax25,
    ),
    FramingProfile(
        canonical="ax25_g3ruh",
        advertised_label="AX.25 G3RUH",
        aliases=("AX25 G3RUH", "G3RUH AX.25"),
        disposition=DecodeDisposition.EXISTING_PENDING_PARITY,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=_MAX_AX25_RETAINED,
        sync_policy="HDLC 0x7e flags after G3RUH/NRZI",
        frame_length_policy="18..max_frame_bytes octets including FCS",
        transforms=("G3RUH descramble", "NRZI decode", "HDLC bit unstuff"),
        integrity_policy="AX.25/X.25 FCS; address sanity is separate station-policy metadata",
        output_semantics="AX.25 body with FCS removed",
        live_supported=True,
        post_pass_supported=True,
        parameters=_AX25_PARAMETER,
        decoder_factory=build_ax25_g3ruh,
    ),
    FramingProfile(
        canonical="ax100_mode5",
        advertised_label="AX100 Mode 5",
        aliases=("AX.100 Mode 5",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=258 * 8 + 31,
        sync_policy="0x930b51de, threshold 0..32 (default 4)",
        frame_length_policy="fixed 258-byte capture; Golay header declares RS span 33..255",
        transforms=("Golay(24,12) header", "optional CCSDS randomizer", "shortened RS"),
        integrity_policy="Golay bounded correction and conventional-basis CCSDS RS decode",
        output_semantics="payload with Golay header and 32 RS parity symbols removed",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "scrambler": ParameterSpec(str, default="CCSDS", choices=("CCSDS", "none")),
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
        },
        decoder_factory=build_ax100_mode5,
    ),
    FramingProfile(
        canonical="ax100_mode6",
        advertised_label="AX100 Mode 6",
        aliases=("AX.100 Mode 6", "AX100 RS"),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=256 * 8 + 31,
        sync_policy="0x930b51de after self-synchronizing descramble; threshold 0..32 (default 4)",
        frame_length_policy="fixed 256-byte capture; byte 0 declares a shortened RS span 34..255",
        transforms=("multiplicative descrambler mask 0x21 seed 0 length 16", "shortened RS"),
        integrity_policy="conventional-basis CCSDS RS decode; no additional CRC",
        output_semantics="declared payload with length byte and 32 RS parity symbols removed",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
        },
        decoder_factory=build_ax100_mode6,
    ),
    FramingProfile(
        canonical="ax100_asm_golay",
        advertised_label="AX100 ASM+Golay",
        aliases=("AX.100 ASM+Golay", "AX100 ASM"),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=258 * 8 + 31,
        sync_policy="0x930b51de, threshold 0..32 (default 4)",
        frame_length_policy="fixed 258-byte capture; Golay header declares RS span 33..255",
        transforms=("Golay(24,12) header", "optional CCSDS randomizer", "shortened RS"),
        integrity_policy="Golay bounded correction and conventional-basis CCSDS RS decode",
        output_semantics="payload with Golay header and 32 RS parity symbols removed",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "scrambler": ParameterSpec(str, default="CCSDS", choices=("CCSDS", "none")),
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
        },
        decoder_factory=build_ax100_asm_golay,
    ),
    FramingProfile(
        canonical="usp",
        advertised_label="USP",
        aliases=("Unified SPUTNIX Protocol",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.SOFT_SYMBOLS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=4144 + 63,
        sync_policy="64-bit soft sync, normalized-distance threshold 0..64 (default 13)",
        frame_length_policy=(
            "PLS code selects 48 or 223 RS data bytes from fixed 4144-symbol capture"
        ),
        transforms=("PLS crop", "CCSDS Viterbi", "CCSDS derandomizer", "dual-basis RS"),
        integrity_policy="PLS correlation must be unambiguous and dual-basis RS must decode",
        output_semantics="little-endian length-delimited AX.25 bytes after four-byte USP header",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=13, minimum=0, maximum=64),
        },
        decoder_factory=build_usp,
    ),
    FramingProfile(
        canonical="mobitex",
        advertised_label="Mobitex",
        aliases=("Mobitex classic",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.SOFT_SYMBOLS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=963 * 8 + 15,
        sync_policy="0x5765 soft sync, normalized-distance threshold 0..16 (default 3)",
        frame_length_policy=(
            "three-byte control/FEC header; corrected second control byte selects "
            "1..32 fixed 30-byte coded data blocks"
        ),
        transforms=("GNU additive derandomizer", "12x20 permutation", "Mobitex FEC"),
        integrity_policy="control FEC plus per-block big-endian CRC-16/X-25",
        output_semantics="TUBIX20 master frame with invalid-block bitmap and quality markers",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=3, minimum=0, maximum=16),
        },
        decoder_factory=build_mobitex,
    ),
    FramingProfile(
        canonical="mobitex_nx",
        advertised_label="Mobitex-NX",
        aliases=("Mobitex NX",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.SOFT_SYMBOLS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=971 * 8 + 15,
        sync_policy="0x0ef0 soft sync, normalized-distance threshold 0..16 (default 3)",
        frame_length_policy="control FEC selects 1..32 fixed 30-byte coded data blocks",
        transforms=("GNU additive derandomizer", "12x20 permutation", "Mobitex FEC"),
        integrity_policy="callsign gate plus per-block big-endian CRC-16/X-25",
        output_semantics="TUBIX20 master frame with invalid-block bitmap and quality markers",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "variant": ParameterSpec(
                str, default="default", choices=("default", "BEESAT-1", "BEESAT-9")
            ),
            "callsign": ParameterSpec(str, default=""),
            # The upstream default is conditional: 12 with a configured
            # callsign, otherwise 2. Preserve omission for the builder.
            "callsign_threshold": ParameterSpec(int, minimum=0, maximum=64),
            "sync_threshold": ParameterSpec(int, default=3, minimum=0, maximum=16),
        },
        decoder_factory=build_mobitex_nx,
    ),
    FramingProfile(
        canonical="geoscan",
        advertised_label="GEOSCAN",
        aliases=("GeoScan",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=258 * 8 + 31,
        sync_policy="0x930b51de, Hamming threshold 0..32 (default 4)",
        frame_length_policy="fixed 3..258 bytes after sync (default 66)",
        transforms=("PN9 x^9+x^5+1 seed 0x1ff",),
        integrity_policy="CRC-16/CC11XX, big-endian wire field",
        output_semantics="fixed packet with two CRC bytes removed",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "frame_size": ParameterSpec(int, default=66, minimum=3, maximum=258),
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
        },
        decoder_factory=build_geoscan,
    ),
    FramingProfile(
        canonical="ao40_fec",
        advertised_label="AO-40 FEC",
        aliases=("AO40 FEC",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.SOFT_SYMBOLS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=5200 - 1,
        sync_policy="65-bit sync distributed every 80 symbols; threshold 0..25 (default 8)",
        frame_length_policy="fixed 80x65 matrix; 5132 coded symbols after 65-symbol skip",
        transforms=("matrix deinterleave", "CCSDS Viterbi", "CCSDS randomizer", "RS"),
        integrity_policy="two-way conventional-basis CCSDS RS; optional preserved CRC-16/ARC",
        output_semantics="256-byte AO-40 frame; optional CRC is validated and preserved",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=8, minimum=0, maximum=25),
            "crc": ParameterSpec(bool, default=False),
        },
        decoder_factory=build_ao40_fec,
    ),
    FramingProfile(
        canonical="ao40_fec_short",
        advertised_label="AO-40 FEC short",
        aliases=("AO40 FEC short",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.SOFT_SYMBOLS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=2652 - 1,
        sync_policy="52-bit sync distributed every 51 symbols; threshold 0..25 (default 8)",
        frame_length_policy="fixed 51x52 matrix; 2572 coded symbols after 80-symbol skip",
        transforms=("matrix deinterleave", "CCSDS Viterbi", "CCSDS randomizer", "RS"),
        integrity_policy="one-way conventional-basis CCSDS RS; optional preserved CRC-16/ARC",
        output_semantics="128-byte AO-40 short frame; optional CRC is validated and preserved",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=8, minimum=0, maximum=25),
            "crc": ParameterSpec(bool, default=False),
        },
        decoder_factory=build_ao40_fec_short,
    ),
    FramingProfile(
        canonical="ao40_uncoded",
        advertised_label="AO-40 uncoded",
        aliases=("AO40 uncoded",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=514 * 8 + 31,
        sync_policy="0x3915ed30, Hamming threshold 0..32 (default 3)",
        frame_length_policy="fixed 514 bytes after sync",
        transforms=(),
        integrity_policy="CRC-16/CCITT-FALSE, big-endian wire field",
        output_semantics="512-byte frame with CRC removed",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=3, minimum=0, maximum=32),
        },
        decoder_factory=build_ao40_uncoded,
    ),
    FramingProfile(
        canonical="ccsds_concatenated",
        advertised_label="CCSDS Concatenated",
        aliases=("CCSDS Conv+RS",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.SOFT_SYMBOLS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=40000,
        sync_policy="CCSDS ASM after dual convolutional pair-phase hypotheses",
        frame_length_policy="configured transfer-frame size plus optional 32 RS bytes per path",
        transforms=(
            "K=7 rate-1/2 Viterbi",
            "optional differential decode",
            "optional CCSDS derandomizer",
            "optional RS",
        ),
        integrity_policy="optional conventional or CCSDS dual-basis RS; otherwise not present",
        output_semantics="configured transfer-frame bytes with convolutional and RS parity removed",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "frame_size": ParameterSpec(int, default=223, minimum=1, maximum=1784),
            "rs_enabled": ParameterSpec(bool, default=True),
            "rs_basis": ParameterSpec(str, default="dual", choices=("conventional", "dual")),
            "rs_interleaving": ParameterSpec(int, default=1, minimum=1, maximum=8),
            "scrambler": ParameterSpec(str, default="CCSDS", choices=("CCSDS", "none")),
            "precoding": ParameterSpec(
                str, default="none", choices=("none", "differential")
            ),
            "convolutional": ParameterSpec(
                str,
                default="CCSDS",
                choices=(
                    "CCSDS",
                    "NASA-DSN",
                    "CCSDS uninverted",
                    "NASA-DSN uninverted",
                ),
            ),
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
            "viterbi_traceback": ParameterSpec(int, default=80, minimum=6, maximum=256),
        },
        decoder_factory=build_ccsds_concatenated,
    ),
    FramingProfile(
        canonical="ccsds_reed_solomon",
        advertised_label="CCSDS Reed-Solomon",
        aliases=("CCSDS RS",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=2040 * 8 + 31,
        sync_policy="CCSDS ASM 0x1acffc1d, Hamming threshold 0..32 (default 4)",
        frame_length_policy="configured data size plus 32 parity bytes per interleave path",
        transforms=(
            "optional differential decode",
            "optional CCSDS derandomizer",
            "shortened/interleaved RS",
        ),
        integrity_policy="conventional or CCSDS dual-basis RS decode",
        output_semantics="configured transfer-frame bytes with RS parity removed",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "frame_size": ParameterSpec(int, default=223, minimum=1, maximum=1784),
            "rs_basis": ParameterSpec(str, default="dual", choices=("conventional", "dual")),
            "rs_interleaving": ParameterSpec(int, default=1, minimum=1, maximum=8),
            "scrambler": ParameterSpec(str, default="CCSDS", choices=("CCSDS", "none")),
            "precoding": ParameterSpec(
                str, default="none", choices=("none", "differential")
            ),
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
        },
        decoder_factory=build_ccsds_rs,
    ),
    FramingProfile(
        canonical="ccsds_uncoded",
        advertised_label="CCSDS Uncoded",
        aliases=("CCSDS ASM Uncoded",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=2048 * 8 + 31,
        sync_policy="CCSDS ASM 0x1acffc1d, Hamming threshold 0..32 (default 4)",
        frame_length_policy="configured fixed frame size up to 2048 bytes",
        transforms=("optional differential decode", "optional CCSDS derandomizer"),
        integrity_policy="no integrity field; explicit-profile only and never autodetected",
        output_semantics="configured transfer-frame bytes",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "frame_size": ParameterSpec(int, default=223, minimum=1, maximum=2048),
            "rs_interleaving": ParameterSpec(int, default=1, minimum=1, maximum=8),
            "scrambler": ParameterSpec(str, default="CCSDS", choices=("CCSDS", "none")),
            "precoding": ParameterSpec(
                str, default="none", choices=("none", "differential")
            ),
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
        },
        decoder_factory=build_ccsds_uncoded,
    ),
    FramingProfile(
        canonical="ngham",
        advertised_label="NGHam",
        aliases=(),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=258 * 8 + 31,
        sync_policy="0x5de62a7e, Hamming threshold 0..32 (default 4)",
        frame_length_policy="24-bit protected size tag selects a 47..255-byte RS codeword",
        transforms=("NGHam size tag", "CCSDS derandomizer", "shortened RS16/RS32"),
        integrity_policy="NGHam RS correction followed by CRC-16/X-25, big-endian field",
        output_semantics="NGHam packet with RS parity, padding, and CRC removed; header retained",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
            "tag_threshold": ParameterSpec(int, default=6, minimum=0, maximum=6),
        },
        decoder_factory=build_ngham,
    ),
    FramingProfile(
        canonical="ngham_no_rs",
        advertised_label="NGHam no Reed Solomon",
        aliases=("NGHam no RS",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=258 * 8 + 31,
        sync_policy="0x5de62a7e, Hamming threshold 0..32 (default 4)",
        frame_length_policy="24-bit protected size tag selects a 47..255-byte uncoded slot",
        transforms=("NGHam size tag", "CCSDS derandomizer", "padding removal"),
        integrity_policy="CRC-16/X-25, big-endian wire field",
        output_semantics="NGHam packet with padding and CRC removed; header byte retained",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
            "tag_threshold": ParameterSpec(int, default=6, minimum=0, maximum=6),
        },
        decoder_factory=build_ngham_no_rs,
    ),
    FramingProfile(
        canonical="u482c",
        advertised_label="U482C",
        aliases=("GOMspace U482C",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=258 * 8 + 31,
        sync_policy="0xc3aa6655, threshold 0..32 (default 4)",
        frame_length_policy="fixed 258-byte capture; Golay header declares encoded span 1..255",
        transforms=(
            "Golay(24,12) header",
            "optional NASA-DSN-uninverted Viterbi",
            "optional CCSDS randomizer",
            "optional shortened RS",
        ),
        integrity_policy="RS when header-enabled; otherwise explicit NOT_PRESENT integrity",
        output_semantics="decoded U482C payload after header-selected stages",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
        },
        decoder_factory=build_u482c,
    ),
    FramingProfile(
        canonical="snet",
        advertised_label="SNET",
        aliases=("S-NET",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=CAPTURE_SIZE_SNET * 8 + 31,
        sync_policy="0x04cf5fc8, Hamming threshold 0..15 (default 4)",
        frame_length_policy="fixed 512-byte capture; BCH header declares bounded PDU length",
        transforms=("BCH(15,5,7) header", "optional BCH(15,k,d) payload"),
        integrity_policy="CRC-5 header and CRC-13 payload; buggy compatibility explicit",
        output_semantics="decoded PDU bytes with SNET source/header metadata",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=15),
            "buggy_crc": ParameterSpec(bool, default=False),
        },
        decoder_factory=build_snet,
    ),
    FramingProfile(
        canonical="openlst",
        advertised_label="OpenLST",
        aliases=("Open LST",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=CAPTURE_SIZE_OPENLST * 8 + 31,
        sync_policy="0xd391d391, Hamming threshold 0..32 (default 4)",
        frame_length_policy="fixed 520-byte FEC capture; decoded byte 0 declares 3..256 bytes",
        transforms=("CC1110 DN504 FEC/interleave", "PN9 whitening"),
        integrity_policy="CRC-16 polynomial 0x8005, init 0xffff, little-endian wire field",
        output_semantics="declared OpenLST frame with length byte and CRC removed",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
        },
        decoder_factory=build_openlst,
    ),
    FramingProfile(
        canonical="reaktor_hello_world",
        advertised_label="Reaktor Hello World",
        aliases=("Reaktor",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=258 * 8 + 31,
        sync_policy="0x352e352e, Hamming threshold 0..32 (default 4)",
        frame_length_policy="capture 258 bytes then apply bounded CC11xx length byte",
        transforms=("PN9 x^9+x^5+1 seed 0x1ff", "head 3 / tail 1 crop"),
        integrity_policy="CRC-16/CC11XX, big-endian wire field",
        output_semantics="application payload after length/header/tail and CRC removal",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
        },
        decoder_factory=build_reaktor,
    ),
    FramingProfile(
        canonical="smogp_ra",
        advertised_label="SMOG-P RA",
        aliases=("SMOGP RA",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.SOFT_SYMBOLS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=514 * 8 + 15,
        sync_policy="0x2dd4, soft Hamming threshold 0..7 (default exact)",
        frame_length_policy="frame_size 128 or 256 selects 260 or 514 RA wire bytes",
        transforms=("byte-bit reversal", "40-pass punctured repeat-accumulate decode"),
        integrity_policy="no SMOG-P CRC; upstream 0.35 RA recode-distance gate",
        output_semantics="128- or 256-byte SMOG-P frame with NOT_PRESENT integrity",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "frame_size": ParameterSpec(int, default=128, choices=(128, 256)),
            "sync_threshold": ParameterSpec(int, default=0, minimum=0, maximum=7),
            "error_threshold": ParameterSpec(
                float, default=0.35, minimum=0.0, maximum=0.35
            ),
        },
        decoder_factory=build_smogp_ra,
    ),
    FramingProfile(
        canonical="smogp_signalling",
        advertised_label="SMOG-P Signalling",
        aliases=("SMOGP Signalling",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=2 * (64 * 8 + 63),
        sync_policy="64-bit RX sync and optional new-protocol TX-observation sync; threshold 0..64",
        frame_length_policy="fixed 64 bytes after sync",
        transforms=("fixed extraction",),
        integrity_policy="no integrity field; explicit-profile only and never autodetected",
        output_semantics="64-byte signalling payload",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=8, minimum=0, maximum=64),
            "new_protocol": ParameterSpec(bool, default=False),
        },
        decoder_factory=build_smogp_signalling,
    ),
    FramingProfile(
        canonical="fx25_nrzi",
        advertised_label="FX.25 NRZI",
        aliases=("Astrocast FX.25 NRZ-I", "Astrocast FX.25 NRZ", "FX25 NRZI"),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=255 * 8 + 63,
        sync_policy="Astrocast 64-bit tag, threshold 0..31 (default 8), after optional NRZI",
        frame_length_policy="fixed 255-byte reflected dual-basis RS codeword",
        transforms=("optional NRZI", "byte reflection", "dual-basis RS"),
        integrity_policy="RS decode followed by inner AX.25/X.25 FCS between 0x7e flags",
        output_semantics="inner Astrocast payload with flags and FCS removed",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=8, minimum=0, maximum=31),
            "nrzi": ParameterSpec(bool, default=True),
        },
        decoder_factory=build_fx25,
    ),
    FramingProfile(
        canonical="tt64",
        advertised_label="TT-64",
        aliases=("TT64",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=64 * 8 + 15,
        sync_policy="0x2dd4, Hamming threshold 0..16 (default 1)",
        frame_length_policy="fixed 64-byte shortened RS codeword",
        transforms=("shortened RS(64,48) over GF(0x11d), FCR 1",),
        integrity_policy="RS correction followed by CRC-16/ARC, little-endian wire field",
        output_semantics="46-byte payload with RS parity and CRC removed",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=1, minimum=0, maximum=16),
        },
        decoder_factory=build_tt64,
    ),
    FramingProfile(
        canonical="sanosat",
        advertised_label="SanoSat",
        aliases=("SanoSat-1",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=135 * 8 + 15,
        sync_policy="ORION Space 0x2dd4 LSB-first, received as 0xb42b; exact by default",
        frame_length_policy="capture 135 bytes; declared length + 5 must fit capture",
        transforms=("validate/remove CRC1", "validate/remove CRC2", "remove length+delimiter"),
        integrity_policy="CRC-16/CCITT-FALSE CRC1 and CRC2, little-endian wire fields",
        output_semantics="SanoSat message after length, CRC1, delimiter, and CRC2 removal",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=0, minimum=0, maximum=16),
        },
        decoder_factory=build_sanosat,
    ),
    FramingProfile(
        canonical="grizu263a",
        advertised_label="Grizu-263A",
        aliases=("Grizu 263A",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=258 * 8 + 63,
        sync_policy="0x0123456789abcdef, Hamming threshold 0..64 (default 8)",
        frame_length_policy="capture 258 bytes then apply bounded SX12xx length byte",
        transforms=("reflect bytes", "PN9 seed 0x100", "reflect bytes", "head 3 / tail 1 crop"),
        integrity_policy="CRC-16/CC11XX, big-endian wire field",
        output_semantics="application payload after length/header/tail and CRC removal",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=8, minimum=0, maximum=64),
        },
        decoder_factory=build_grizu,
    ),
    FramingProfile(
        canonical="aalto1",
        advertised_label="AALTO-1",
        aliases=("Aalto 1",),
        disposition=DecodeDisposition.IN_PROGRESS,
        symbol_input=SymbolInput.HARD_BITS,
        accepted_polarities=_BOTH_POLARITIES,
        max_retained_symbols=258 * 8 + 31,
        sync_policy="0x352e352e, Hamming threshold 0..32 (default 4)",
        frame_length_policy="capture 258 bytes then apply bounded CC11xx length byte",
        transforms=("PN9 x^9+x^5+1 seed 0x1ff", "head 3 / tail 1 crop"),
        integrity_policy="CRC-16/X-25, little-endian wire field",
        output_semantics="application payload after length/header/tail and CRC removal",
        live_supported=False,
        post_pass_supported=False,
        parameters={
            "sync_threshold": ParameterSpec(int, default=4, minimum=0, maximum=32),
        },
        decoder_factory=build_aalto1,
    ),
)

REGISTRY = ProfileRegistry(_PROFILES)


def advertised_profiles() -> Mapping[str, FramingProfile]:
    return REGISTRY.advertised


def resolve_profile(label: object) -> FramingProfile | None:
    return REGISTRY.resolve(label)


def build_decoder(
    label: object, parameters: Mapping[str, object] | None = None
) -> StreamingDecoder:
    return REGISTRY.build(label, parameters)


__all__ = [
    "ProfileRegistry",
    "REGISTRY",
    "advertised_profiles",
    "build_decoder",
    "resolve_profile",
]
