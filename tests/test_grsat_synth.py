"""docs/08 Phase 1 — synthetic gr-satellites SatYAML builder.

Verifies the SatYAML we hand gr-satellites for a non-catalogued bird is well-formed
(name + int norad + transmitters, per gr-satellites' check_yaml) and maps our
``(modulation, baud, framing)`` to gr-satellites' transmitter vocabulary correctly.
"""
from __future__ import annotations

import grsat_synth
import yaml


def test_fsk_family_maps_to_FSK_with_h_half_deviation():
    text = grsat_synth.synthetic_satyaml(60527, "gmsk", 9600, "AX.25 G3RUH", 400.65e6)
    d = yaml.safe_load(text)
    assert d["norad"] == 60527 and isinstance(d["norad"], int)
    tx = d["transmitters"]["downlink"]
    assert tx["modulation"] == "FSK"
    assert tx["baudrate"] == 9600
    assert tx["framing"] == "AX.25 G3RUH"
    assert tx["deviation"] == 2400  # h=0.5 → 0.5*9600/2
    assert tx["frequency"] == 400.65e6


def test_bpsk_maps_to_BPSK_no_deviation():
    tx = yaml.safe_load(
        grsat_synth.synthetic_satyaml(60469, "bpsk", 1200, "AX.25 G3RUH", 400.65e6)
    )["transmitters"]["downlink"]
    assert tx["modulation"] == "BPSK" and "deviation" not in tx


def test_afsk_maps_with_af_carrier_and_deviation():
    tx = yaml.safe_load(
        grsat_synth.synthetic_satyaml(25544, "afsk", 1200, "AX.25", 145.825e6)
    )["transmitters"]["downlink"]
    assert tx["modulation"] == "AFSK" and tx["af_carrier"] == 1700 and tx["deviation"] == 500


def test_unsupported_modulation_returns_none():
    # gr-satellites has no QAM/OFDM/QPSK demod → our own modem handles those (Tier 2).
    assert grsat_synth.synthetic_satyaml(1, "qam16", 200000, "CCSDS Concatenated", 2.26e9) is None
    assert grsat_synth.synthetic_satyaml(1, "qpsk", 9600, "AX.25", 400e6) is None


def test_missing_framing_or_baud_returns_none():
    assert grsat_synth.synthetic_satyaml(1, "fsk", 9600, None, 400e6) is None
    assert grsat_synth.synthetic_satyaml(1, "fsk", 0, "AX.25", 400e6) is None


def test_output_has_the_fields_gr_satellites_requires():
    # check_yaml requires name + int norad + transmitters; grc_block mode needs no data block.
    d = yaml.safe_load(grsat_synth.synthetic_satyaml(60527, "fsk", 9600, "AX100 ASM+Golay", 400e6))
    assert "name" in d and isinstance(d["norad"], int) and "transmitters" in d
    assert "data" not in d  # grc_block=True reads only transmitters


def test_write_synthetic_satyaml_roundtrips_to_a_file(tmp_path):
    p = tmp_path / "synth.yml"
    out = grsat_synth.write_synthetic_satyaml(str(p), 60527, "fsk", 9600, "USP", 400.575e6)
    assert out == str(p)
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert d["transmitters"]["downlink"]["framing"] == "USP"
    # unsupported modulation writes nothing
    assert grsat_synth.write_synthetic_satyaml(str(p), 1, "ofdm", 1000, "x", 1e9) is None
