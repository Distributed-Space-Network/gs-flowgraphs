"""Bounded VZLUSAT-2 CSP telemetry preview.

Adapted from ``python/telemetry/vzlusat_2.py``, ``python/telemetry/csp.py``, and
``python/adapters.py`` in gr-satellites commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

Upstream copyrights:
  Copyright 2020 jgromes <gromes.jan@gmail.com>
  Copyright 2019 Daniel Estevez <daniel@destevez.net>
  Copyright 2018 Daniel Estevez <daniel@destevez.net>

Modified for bounded, JSON-safe, exact-mission preview use on 2026-07-18.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from construct import (
    Adapter,
    BitsInteger,
    BitStruct,
    Flag,
    GreedyBytes,
    IfThenElse,
    Int8ub,
    Int16sb,
    Int16ub,
    Int32ub,
    Padding,
    Struct,
    Switch,
    this,
)

from native_telemetry.types import FrameContext, ParserPreview

NORAD_ID = 51085
FRAMING = "ax100_asm_golay"
BEACON_COMMAND = 0x56
DROP_COMMAND = 0x03
MIN_PACKET_BYTES = 5
BEACON_PACKET_BYTES = 43
DROP_HEADER_BYTES = 14
MAX_PACKET_BYTES = 65_536
MAX_RAW_PREVIEW_BYTES = 512


class _LinearAdapter(Adapter):
    def __init__(self, scale: int, subcon: object) -> None:
        self._scale = scale
        super().__init__(subcon)

    def _decode(self, obj: int, context: object, path: object) -> float:
        return obj / self._scale

    def _encode(self, obj: float, context: object, path: object) -> int:
        return int(round(obj * self._scale))


CSP_HEADER = BitStruct(
    "priority" / BitsInteger(2),
    "source" / BitsInteger(5),
    "destination" / BitsInteger(5),
    "destination_port" / BitsInteger(6),
    "source_port" / BitsInteger(6),
    "reserved" / BitsInteger(3),
    "fragmentation" / Flag,
    "hmac" / Flag,
    "xtea" / Flag,
    "rdp" / Flag,
    "crc" / Flag,
)

BEACON = Struct(
    Padding(8),
    "obc_timestamp" / Int32ub,
    "obc_boot_count" / Int32ub,
    "obc_reset_cause" / Int32ub,
    "eps_vbatt" / Int16ub,
    "eps_cursun" / Int16ub,
    "eps_cursys" / Int16ub,
    "eps_temp_bat" / Int16sb,
    "radio_temp_pa" / _LinearAdapter(10, Int16sb),
    "radio_tot_tx_count" / Int32ub,
    "radio_tot_rx_count" / Int32ub,
)

DROP = Struct(
    "flag" / Int8ub,
    "chunk" / Int32ub,
    "time" / Int32ub,
    "data" / GreedyBytes,
)

VZLUSAT2 = Struct(
    "csp_header" / CSP_HEADER,
    "command" / Int8ub,
    "payload"
    / IfThenElse(
        (this.csp_header.source == 1)
        & (this.csp_header.destination == 26)
        & (this.csp_header.source_port == 18)
        & (this.csp_header.destination_port == 18),
        Switch(
            this.command,
            {BEACON_COMMAND: BEACON, DROP_COMMAND: DROP},
            default=GreedyBytes,
        ),
        GreedyBytes,
    ),
)


def _csp_values(header: object) -> dict[str, object]:
    return {
        "priority": int(header.priority),
        "source": int(header.source),
        "destination": int(header.destination),
        "destination_port": int(header.destination_port),
        "source_port": int(header.source_port),
        "reserved": int(header.reserved),
        "fragmentation": bool(header.fragmentation),
        "hmac": bool(header.hmac),
        "xtea": bool(header.xtea),
        "rdp": bool(header.rdp),
        "crc": bool(header.crc),
    }


def _bounded_bytes(data: bytes) -> dict[str, object]:
    preview = data[:MAX_RAW_PREVIEW_BYTES]
    return {
        "data_hex": preview.hex(),
        "data_length": len(data),
        "data_truncated": len(data) > len(preview),
    }


def parse_vzlusat2(context: FrameContext) -> ParserPreview:
    """Parse one already-deframed VZLUSAT-2 payload under registry mission gates."""

    data = context.payload
    if len(data) < MIN_PACKET_BYTES:
        raise ValueError(f"VZLUSAT-2 packet is shorter than {MIN_PACKET_BYTES} bytes")
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError(f"VZLUSAT-2 packet exceeds {MAX_PACKET_BYTES} bytes")

    header = CSP_HEADER.parse(data[:4])
    header_gate = (
        header.source == 1
        and header.destination == 26
        and header.source_port == 18
        and header.destination_port == 18
    )
    command = data[4]
    if header_gate and command == BEACON_COMMAND and len(data) != BEACON_PACKET_BYTES:
        raise ValueError(
            f"VZLUSAT-2 Beacon must be exactly {BEACON_PACKET_BYTES} bytes"
        )
    if header_gate and command == DROP_COMMAND and len(data) < DROP_HEADER_BYTES:
        raise ValueError(f"VZLUSAT-2 Drop is shorter than {DROP_HEADER_BYTES} bytes")

    parsed = VZLUSAT2.parse(data)
    csp = _csp_values(parsed.csp_header)
    mission_gate = (
        csp["source"] == 1
        and csp["destination"] == 26
        and csp["source_port"] == 18
        and csp["destination_port"] == 18
    )
    values: dict[str, object] = {
        "command": int(parsed.command),
        "command_hex": f"0x{int(parsed.command):02x}",
        "csp": csp,
        "mission_csp_gate": mission_gate,
    }

    if not mission_gate or parsed.command not in {BEACON_COMMAND, DROP_COMMAND}:
        values.update({"kind": "raw", **_bounded_bytes(bytes(parsed.payload))})
        return ParserPreview(status="ok", values=values)

    if parsed.command == BEACON_COMMAND:
        payload = parsed.payload
        values.update(
            {
                "kind": "beacon",
                "telemetry": {
                    "obc_timestamp": int(payload.obc_timestamp),
                    "obc_boot_count": int(payload.obc_boot_count),
                    "obc_reset_cause": int(payload.obc_reset_cause),
                    "eps_vbatt": int(payload.eps_vbatt),
                    "eps_cursun": int(payload.eps_cursun),
                    "eps_cursys": int(payload.eps_cursys),
                    "eps_temp_bat": int(payload.eps_temp_bat),
                    "radio_temp_pa": float(payload.radio_temp_pa),
                    "radio_tot_tx_count": int(payload.radio_tot_tx_count),
                    "radio_tot_rx_count": int(payload.radio_tot_rx_count),
                },
            }
        )
        return ParserPreview(status="ok", values=values)

    payload = parsed.payload
    values.update(
        {
            "kind": "drop",
            "telemetry": {
                "flag": int(payload.flag),
                "chunk": int(payload.chunk),
                "time": int(payload.time),
                **_bounded_bytes(bytes(payload.data)),
            },
        }
    )
    return ParserPreview(status="ok", values=values)


__all__ = [
    "BEACON_COMMAND",
    "BEACON_PACKET_BYTES",
    "DROP_COMMAND",
    "DROP_HEADER_BYTES",
    "FRAMING",
    "MAX_PACKET_BYTES",
    "NORAD_ID",
    "VZLUSAT2",
    "parse_vzlusat2",
]
