"""FEC registry — forward-error-correction codecs the framing layer composes.

Phase 0 of docs/08 (universal modem + framing): a skeleton. Today each deframer encapsulates
its own coding (AX.25 G3RUH scramble, EnduroSat CRC), mirroring gr-satellites, so there is no
mandatory ``demod → fec → deframe`` middle stage yet. Tier 1/2 (docs/08) populate this with the
GNU Radio ``fec.*`` / gr-satellites codecs — CCSDS convolutional/Viterbi (``fec.decode_ccsds_27``),
Reed-Solomon (gr-satellites ``ccsds_rs``), LDPC (``fec.ldpc_decoder``), Turbo, scramblers, ASM —
which the new framings (CCSDS TM/TC/AOS/USLP, Argos) pull in as reusable building blocks.
"""
from __future__ import annotations

# Populated in Tier 1/2 (docs/08). code name → (decode, encode) callables.
_CODECS: dict[str, object] = {}


def known_codes() -> tuple[str, ...]:
    return tuple(_CODECS)
