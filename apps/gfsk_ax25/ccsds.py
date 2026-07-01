"""CCSDS TM/AOS Transfer Frame framing (docs/08 Tier 2 — framings not in gr-satellites).

The CCSDS Space Data Link backbone (commercial/government). This module implements the TM Space
Data Link Protocol (CCSDS 132.0) transfer frame end to end — build + deframe — over the CCSDS
Sync & Channel Coding chain (CCSDS 131.0): Attached Sync Marker → optional RS(255,223) → optional
pseudo-randomization → optional Frame Error Control (CRC-16). It composes the FEC primitives
(:mod:`gfsk_ax25.reedsolomon`, :mod:`fec`) and is round-trip unit-tested.

Also provides the AOS (CCSDS 732.0) primary-header parser. TC (232.0) / USLP (732.1) share the
same field-parsing style and are follow-ups (see docs/09). numpy/stdlib-only.

Channel-coding order (CCSDS 131.0), TX:  frame(+FECF) → RS encode → randomize → prepend ASM.
RX is the exact inverse:                  find ASM → derandomize → RS decode → FECF check → parse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gfsk_ax25.crc import crc16_ccitt_false
from gfsk_ax25.reedsolomon import RS_NSYM_255_223, RSCodec

ASM_CCSDS = 0x1ACFFC1D
_ASM_BYTES = ASM_CCSDS.to_bytes(4, "big")
_TM_PRIMARY_HDR_LEN = 6
RS_FRAME_LEN = 223          # RS(255,223): a 223-byte transfer frame → 255-byte codeblock
_rs = RSCodec(RS_NSYM_255_223)


# ── TM primary header ────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TMHeader:
    version: int
    spacecraft_id: int
    virtual_channel_id: int
    ocf_flag: int
    master_channel_frame_count: int
    virtual_channel_frame_count: int
    secondary_header_flag: int
    sync_flag: int
    first_header_pointer: int


def _pack_tm_header(h: TMHeader) -> bytes:
    w0 = (h.version & 0x3) << 14 | (h.spacecraft_id & 0x3FF) << 4 \
        | (h.virtual_channel_id & 0x7) << 1 | (h.ocf_flag & 0x1)
    w2 = (h.secondary_header_flag & 0x1) << 15 | (h.sync_flag & 0x1) << 14 \
        | (0 & 0x1) << 13 | (0 & 0x3) << 11 | (h.first_header_pointer & 0x7FF)
    return bytes([
        (w0 >> 8) & 0xFF, w0 & 0xFF,
        h.master_channel_frame_count & 0xFF, h.virtual_channel_frame_count & 0xFF,
        (w2 >> 8) & 0xFF, w2 & 0xFF,
    ])


def parse_tm_primary_header(frame: bytes) -> TMHeader:
    """Parse the 6-byte TM transfer-frame primary header (CCSDS 132.0)."""
    if len(frame) < _TM_PRIMARY_HDR_LEN:
        raise ValueError("frame shorter than the 6-byte TM primary header")
    w0 = (frame[0] << 8) | frame[1]
    w2 = (frame[4] << 8) | frame[5]
    return TMHeader(
        version=(w0 >> 14) & 0x3,
        spacecraft_id=(w0 >> 4) & 0x3FF,
        virtual_channel_id=(w0 >> 1) & 0x7,
        ocf_flag=w0 & 0x1,
        master_channel_frame_count=frame[2],
        virtual_channel_frame_count=frame[3],
        secondary_header_flag=(w2 >> 15) & 0x1,
        sync_flag=(w2 >> 14) & 0x1,
        first_header_pointer=w2 & 0x7FF,
    )


# ── AOS primary header (CCSDS 732.0) ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AOSHeader:
    version: int
    spacecraft_id: int
    virtual_channel_id: int
    virtual_channel_frame_count: int
    replay_flag: int


def parse_aos_primary_header(frame: bytes) -> AOSHeader:
    """Parse the 6-byte AOS transfer-frame primary header (no Frame Header Error Control field)."""
    if len(frame) < 6:
        raise ValueError("frame shorter than the 6-byte AOS primary header")
    w0 = (frame[0] << 8) | frame[1]
    return AOSHeader(
        version=(w0 >> 14) & 0x3,
        spacecraft_id=(w0 >> 6) & 0xFF,
        virtual_channel_id=w0 & 0x3F,
        virtual_channel_frame_count=(frame[2] << 16) | (frame[3] << 8) | frame[4],
        replay_flag=(frame[5] >> 7) & 0x1,
    )


# ── TM transfer frame: build + deframe (full channel-coding chain) ───────────────────────────
def _fecf(frame_wo_crc: bytes) -> bytes:
    return crc16_ccitt_false(frame_wo_crc).to_bytes(2, "big")


def build_tm_frame(
    header: TMHeader, data: bytes, *,
    frame_len: int = RS_FRAME_LEN, randomize: bool = True, rs: bool = True, fecf: bool = True,
) -> np.ndarray:
    """Build a transmitted TM frame as a hard-bit array: primary header + data (+FECF) padded to
    ``frame_len``, then RS-encoded, randomized, and ASM-prefixed per CCSDS 131.0. Inverse of
    :func:`deframe_tm` — used for round-trip tests and as a reference TX framer."""
    from fec import ccsds_randomize  # noqa: PLC0415 — apps-level module

    body = bytearray(_pack_tm_header(header))
    body += bytes(data)
    fill = frame_len - len(body) - (2 if fecf else 0)
    if fill < 0:
        raise ValueError("data too long for frame_len")
    body += b"\x00" * fill
    if fecf:
        body += _fecf(bytes(body))
    frame = bytes(body)
    codeblock = _rs.encode(frame) if rs else frame
    if randomize:
        codeblock = ccsds_randomize(codeblock)
    bits = np.unpackbits(np.frombuffer(_ASM_BYTES + codeblock, dtype=np.uint8))
    return bits.astype(np.uint8)


def deframe_tm(
    bits, *,
    frame_len: int = RS_FRAME_LEN, randomize: bool = True, rs: bool = True, fecf: bool = True,
) -> list[bytes]:
    """Recover TM transfer frames from a hard-bit stream. For each ASM, take one codeblock,
    derandomize, RS-decode (drop if uncorrectable), verify the FECF CRC, and return the transfer
    frame bytes (header + data field, FECF stripped). A frame failing RS or the FECF is dropped —
    ASM + RS + CRC together make false frames negligible (unlike Argos' short sync + BCH)."""
    from fec import ccsds_derandomize, find_asm  # noqa: PLC0415

    arr = np.asarray(bits, dtype=np.uint8).ravel()
    codeblock_len = (frame_len + RS_NSYM_255_223) if rs else frame_len
    need_bits = codeblock_len * 8
    out: list[bytes] = []
    search_from = 0
    while True:
        idx = find_asm(arr[search_from:], ASM_CCSDS)
        if idx < 0:
            break
        start = search_from + idx  # find_asm returns the index just AFTER the ASM = codeblock start
        if start + need_bits > arr.size:
            break
        codeblock = bytes(np.packbits(arr[start:start + need_bits]))
        if randomize:
            codeblock = ccsds_derandomize(codeblock)
        frame = _rs.decode(codeblock) if rs else codeblock
        if frame is None:  # RS uncorrectable → spurious ASM; resume right after it
            search_from = start
            continue
        if fecf and (len(frame) < 2 or _fecf(frame[:-2]) != frame[-2:]):
            search_from = start
            continue
        out.append(bytes(frame[:-2] if fecf else frame))
        search_from = start + need_bits  # consume the whole codeblock
    return out
