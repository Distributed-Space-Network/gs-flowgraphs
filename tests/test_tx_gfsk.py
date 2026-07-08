"""Standalone GFSK bench transmitter: frame build (endurosat + ax25), rate snap, settings, file out.

Round-trips the built IQ back through the proven receivers to prove the tool emits decodable frames;
the actual SoapySDR keying is bench-only (# pragma: no cover in tx_gfsk)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tx_gfsk

from gfsk_ax25 import ax25, endurosat, endurosat_link, gfsk

_SR = 96_000.0  # 10 samples/symbol at 9600


def _guard(iq: np.ndarray, n: int = 2000) -> np.ndarray:
    return np.concatenate([np.zeros(n, np.complex64), iq, np.zeros(n, np.complex64)])


def test_resolve_rate_snaps_to_integer_sps():
    assert tx_gfsk.resolve_rate(2_048_000, 9600) % 9600 == 0
    assert abs(tx_gfsk.resolve_rate(2_048_000, 9600) - 2_048_000) < 9600  # nearest multiple
    assert tx_gfsk.resolve_rate(96_000, 9600) == 96_000  # already integer sps
    assert tx_gfsk.resolve_rate(2_048_000, 4800) % 4800 == 0


def test_parse_settings():
    assert tx_gfsk.parse_settings("PAD=30,IAMP=0") == [("PAD", 30.0), ("IAMP", 0.0)]
    assert tx_gfsk.parse_settings("") == []
    assert tx_gfsk.parse_settings("junk,LNA=6,=bad,x=") == [("LNA", 6.0)]  # malformed skipped
    assert tx_gfsk.parse_settings("=1,PAD=-10") == [("PAD", -10.0)]


def test_build_frame_iq_endurosat_roundtrip():
    payload = b"AIRMAC-encrypted-uplink-blob"
    iq = tx_gfsk.build_frame_iq(
        payload, framing="endurosat", sample_rate=_SR, symbol_rate=9600.0, mod_index=0.5, bt=0.5
    )
    assert payload in endurosat_link.receive(_guard(iq), _SR)


def test_build_frame_iq_ax25_roundtrip():
    payload = b"CMD set-beacon 30s"
    iq = tx_gfsk.build_frame_iq(
        payload, framing="ax25", sample_rate=_SR, symbol_rate=9600.0, mod_index=0.5, bt=0.5,
        dest="ES1", src="DSN0",
    )
    dec = endurosat.StreamDecoder(
        _SR, profile=endurosat.LinkProfile(symbol_rate_hz=9600.0), recover_timing=False
    )
    dec.push(iq)
    frames = dec.flush()
    expected = ax25.encode_ui(dest="ES1", src="DSN0", info=payload)
    assert expected in frames


def test_build_frame_iq_endurosat_truncates_over_max():
    payload = bytes((np.arange(200) % 256).astype(np.uint8).tolist())  # > MAX_PAYLOAD (128)
    got = endurosat_link.receive(
        _guard(tx_gfsk.build_frame_iq(payload, framing="endurosat", sample_rate=_SR,
                                      symbol_rate=9600.0, mod_index=0.5, bt=0.5)),
        _SR,
    )
    assert payload[: endurosat_link.MAX_PAYLOAD] in got


def test_build_frame_iq_raw_is_bytes_verbatim():
    # raw = the file bytes AS-IS: MSB-first bits → GFSK, nothing added. len(iq) = bits × sps, and
    # the bits round-trip through the demod (the interior, allowing a couple of edge symbols).
    payload = bytes((np.arange(96) * 5 % 256).astype(np.uint8).tolist())  # 768 bits
    sps = int(round(_SR / 9600.0))
    iq = tx_gfsk.build_frame_iq(
        payload, framing="raw", sample_rate=_SR, symbol_rate=9600.0, mod_index=0.5, bt=0.5
    )
    assert iq.dtype == np.complex64
    assert len(iq) == len(payload) * 8 * sps  # no preamble/sync/len/CRC added

    params = gfsk.GfskParams(sample_rate_hz=_SR, symbol_rate_hz=9600.0, mod_index=0.5, bt=0.5)
    out = "".join(str(b) for b in gfsk.demodulate(iq, params, recover_timing=False).astype(int))
    bits_in = "".join(str(b) for b in np.unpackbits(np.frombuffer(payload, np.uint8)))
    assert bits_in[4:-4] in out  # interior bits recovered verbatim (edge symbols may trim)


def test_build_frame_iq_raw_lsb_bitorder():
    payload = bytes([0x80, 0x01, 0x55, 0xAA]) * 24
    iq = tx_gfsk.build_frame_iq(
        payload, framing="raw", sample_rate=_SR, symbol_rate=9600.0, mod_index=0.5, bt=0.5,
        raw_bitorder="little",
    )
    params = gfsk.GfskParams(sample_rate_hz=_SR, symbol_rate_hz=9600.0, mod_index=0.5, bt=0.5)
    out = "".join(str(b) for b in gfsk.demodulate(iq, params, recover_timing=False).astype(int))
    bits_in = "".join(str(b) for b in np.unpackbits(np.frombuffer(payload, np.uint8),
                                                     bitorder="little"))
    assert bits_in[2:-2] in out


def test_build_frame_iq_raw_zero_gap_bytes_render_silence():
    payload = b"\xAA\x7E" + (b"\x00" * 4) + b"\x55"
    sps = int(round(_SR / 9600.0))
    iq = tx_gfsk.build_frame_iq(
        payload, framing="raw", sample_rate=_SR, symbol_rate=9600.0, mod_index=0.5, bt=0.5,
        raw_zero_gap_bytes=4,
    )
    first = 2 * 8 * sps
    gap = 4 * 8 * sps
    assert len(iq) == len(payload) * 8 * sps
    assert np.max(np.abs(iq[:first])) > 0.9
    assert np.max(np.abs(iq[first:first + gap])) == 0.0
    assert np.max(np.abs(iq[first + gap:])) > 0.9


def test_build_frame_iq_raw_zero_gap_keeps_short_zero_runs_continuous():
    payload = b"\xAA\x00\x00\x7E\x55"  # short zero run is part of the raw frame, not a gap
    plain = tx_gfsk.build_frame_iq(
        payload, framing="raw", sample_rate=_SR, symbol_rate=9600.0, mod_index=0.5, bt=0.5,
    )
    gap_mode = tx_gfsk.build_frame_iq(
        payload, framing="raw", sample_rate=_SR, symbol_rate=9600.0, mod_index=0.5, bt=0.5,
        raw_zero_gap_bytes=4,
    )
    np.testing.assert_array_equal(gap_mode, plain)


def test_xtrx_narrow_bw_lifted_unless_explicitly_allowed():
    args = SimpleNamespace(soapy_tx_device="driver=xtrx", bw=25_000, allow_narrow_bw=False)
    assert tx_gfsk._resolve_tx_bw(args, 2_112_000.0) == tx_gfsk.XTRX_MIN_TX_BW_HZ
    args.allow_narrow_bw = True
    assert tx_gfsk._resolve_tx_bw(args, 2_112_000.0) == 25_000
    args.soapy_tx_device = "driver=rtlsdr"
    args.allow_narrow_bw = False
    assert tx_gfsk._resolve_tx_bw(args, 2_112_000.0) == 25_000


def test_tx_chunk_defaults_below_or_at_stream_mtu():
    assert tx_gfsk._resolve_tx_chunk(0, 4096) == tx_gfsk._TX_CHUNK
    assert tx_gfsk._resolve_tx_chunk(0, 512) == 512
    assert tx_gfsk._resolve_tx_chunk(256, 4096) == 256
    assert tx_gfsk._resolve_tx_chunk(8192, 4096) == 4096


def test_tx_stream_channels_resolves_auto_default_and_explicit():
    args = SimpleNamespace(tx_stream_channels="auto", soapy_tx_device="driver=xtrx")
    assert tx_gfsk._resolve_tx_stream_channels(args, 960_000.0) == "explicit"
    assert tx_gfsk._resolve_tx_stream_channels(args, 2_044_800.0) == "default"
    args.soapy_tx_device = "driver=lime"
    assert tx_gfsk._resolve_tx_stream_channels(args, 960_000.0) == "default"
    args.tx_stream_channels = "default"
    assert tx_gfsk._resolve_tx_stream_channels(args) == "default"
    args.tx_stream_channels = "explicit"
    assert tx_gfsk._resolve_tx_stream_channels(args) == "explicit"
    args.tx_stream_channels = "bad"
    assert tx_gfsk._resolve_tx_stream_channels(args) == "default"


def test_tx_write_call_auto_is_bounded_for_xtrx():
    args = SimpleNamespace(soapy_tx_device="driver=xtrx", tx_write_call="auto")
    assert tx_gfsk._resolve_tx_write_call(args) == "simple"
    args.soapy_tx_device = "driver=lime"
    assert tx_gfsk._resolve_tx_write_call(args) == "simple"
    args.tx_write_call = "simple"
    assert tx_gfsk._resolve_tx_write_call(args) == "simple"
    args.tx_write_call = "bad"
    assert tx_gfsk._resolve_tx_write_call(args) == "simple"


def test_tx_activate_auto_and_modes():
    args = SimpleNamespace(soapy_tx_device="driver=xtrx", tx_activate_elems="auto")
    assert tx_gfsk._resolve_tx_activate_mode(args) == "0"
    args.soapy_tx_device = "driver=lime"
    assert tx_gfsk._resolve_tx_activate_mode(args) == "0"
    args.tx_activate_elems = "burst"
    assert tx_gfsk._resolve_tx_activate_mode(args) == "burst"

    kwargs = {"burst_samples": 1000, "mtu": 4096, "repeat": 3}
    assert tx_gfsk._resolve_activate_elems("0", **kwargs) == 0
    assert tx_gfsk._resolve_activate_elems("mtu", **kwargs) == 4096
    assert tx_gfsk._resolve_activate_elems("burst", **kwargs) == 3000
    assert tx_gfsk._resolve_activate_elems("2048", **kwargs) == 2048
    assert tx_gfsk._resolve_activate_elems("-1", **kwargs) == 0
    assert tx_gfsk._resolve_activate_elems("bad", **kwargs) == 0


def test_tx_format_resolver_and_cs16_conversion():
    args = SimpleNamespace(tx_format="auto", soapy_tx_device="driver=xtrx")
    assert tx_gfsk._resolve_tx_format(args, 480_000.0) == "cf32"
    assert tx_gfsk._resolve_tx_format(args, 960_000.0) == "cs16"
    assert tx_gfsk._resolve_tx_format(args, 2_044_800.0) == "cs16"
    args.soapy_tx_device = "driver=lime"
    assert tx_gfsk._resolve_tx_format(args, 960_000.0) == "cf32"
    args.tx_format = "cf32"
    assert tx_gfsk._resolve_tx_format(args) == "cf32"
    args.tx_format = "cs16"
    assert tx_gfsk._resolve_tx_format(args) == "cs16"
    args.tx_format = "bad"
    assert tx_gfsk._resolve_tx_format(args) == "cf32"

    iq = np.array([1 + 0j, -1 - 1j, 0.5 - 0.5j, 2 + 2j], dtype=np.complex64)
    out = tx_gfsk._iq_to_cs16(iq, scale=0.5)
    assert out.dtype == np.int16
    assert out.shape == (8,)
    assert out[0] == round(0.5 * tx_gfsk._TX_CS16_PEAK)
    assert out[1] == 0
    assert out[2] == round(-0.5 * tx_gfsk._TX_CS16_PEAK)
    assert out[3] == round(-0.5 * tx_gfsk._TX_CS16_PEAK)
    assert out[6] == round(1.0 * tx_gfsk._TX_CS16_PEAK)


def test_transmit_cs16_uses_flat_buffer_but_complex_sample_count(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "SoapySDR",
        SimpleNamespace(
            SOAPY_SDR_TX=1,
            SOAPY_SDR_CF32="CF32",
            SOAPY_SDR_CS16="CS16",
            SOAPY_SDR_END_BURST=2,
            SOAPY_SDR_HAS_TIME=4,
            SOAPY_SDR_TIMEOUT=-1,
        ),
    )

    class Dev:
        def __init__(self):
            self.writes = []

        def setupStream(self, *args):
            assert args == (1, "CS16")
            return "stream"

        def getStreamMTU(self, _stream):
            return 4

        def activateStream(self, *_args):
            return 0

        def writeStream(self, stream, buffs, num_elems):
            assert stream == "stream"
            block = buffs[0]
            self.writes.append((block.copy(), num_elems))
            return SimpleNamespace(ret=num_elems, flags=0)

        def readStreamStatus(self, *_args, **_kwargs):
            return SimpleNamespace(ret=0, flags=0)

        def deactivateStream(self, *_args):
            pass

        def closeStream(self, *_args):
            pass

    dev = Dev()
    tx_gfsk._transmit(
        dev, 0, np.ones(4, dtype=np.complex64),
        repeat=1, gap_s=0.0, sample_rate=1_000_000.0,
        tx_chunk=4, write_call="simple", write_timeout_us=250_000,
        copy_chunks=False, pace=False, write_sleep_us=0,
        stream_channels="default", activate_elems_mode="0",
        tx_format="cs16", tx_scale=1.0,
        tx_time_mode="none", tx_time_lead_ms=50.0,
    )
    assert len(dev.writes) == 1
    block, num_elems = dev.writes[0]
    assert block.dtype == np.int16
    assert block.shape == (8,)
    assert num_elems == 4


def test_tx_time_mode_resolver():
    args = SimpleNamespace(tx_time_mode="none", soapy_tx_device="", allow_xtrx_timed_tx=False)
    assert tx_gfsk._resolve_tx_time_mode(args) == "none"
    args.tx_time_mode = "hw"
    assert tx_gfsk._resolve_tx_time_mode(args) == "hw"
    args.tx_time_mode = "reset"
    assert tx_gfsk._resolve_tx_time_mode(args) == "reset"
    args.tx_time_mode = "bad"
    assert tx_gfsk._resolve_tx_time_mode(args) == "none"


def test_tx_time_mode_disables_xtrx_timed_tx_unless_forced():
    args = SimpleNamespace(
        tx_time_mode="reset", soapy_tx_device="driver=xtrx", allow_xtrx_timed_tx=False,
    )
    assert tx_gfsk._resolve_tx_time_mode(args) == "none"
    args.tx_time_mode = "hw"
    assert tx_gfsk._resolve_tx_time_mode(args) == "none"
    args.allow_xtrx_timed_tx = True
    assert tx_gfsk._resolve_tx_time_mode(args) == "hw"


def test_xtrx_overall_tx_gain_disabled_unless_forced():
    args = SimpleNamespace(
        soapy_tx_device="driver=xtrx",
        gain=30,
        allow_xtrx_overall_gain=False,
    )
    assert not tx_gfsk._use_overall_tx_gain(args)
    args.allow_xtrx_overall_gain = True
    assert tx_gfsk._use_overall_tx_gain(args)
    args.soapy_tx_device = "driver=lime"
    args.allow_xtrx_overall_gain = False
    assert tx_gfsk._use_overall_tx_gain(args)
    args.gain = None
    assert not tx_gfsk._use_overall_tx_gain(args)


def test_configure_tx_skips_noop_named_gain(monkeypatch):
    monkeypatch.setitem(sys.modules, "SoapySDR", SimpleNamespace(SOAPY_SDR_TX=1))

    class Dev:
        def __init__(self):
            self.set_gain_calls = []

        def setSampleRate(self, *_args):
            pass

        def setFrequency(self, *_args):
            pass

        def setBandwidth(self, *_args):
            pass

        def setGainMode(self, *_args):
            pass

        def listGains(self, *_args):
            return ["PAD"]

        def getGainRange(self, *_args):
            return "range"

        def getGain(self, *_args):
            name = _args[2] if len(_args) > 2 else None
            return 0.0 if name == "PAD" else 52.0

        def setGain(self, *args):
            self.set_gain_calls.append(args)

    args = SimpleNamespace(
        soapy_tx_device="driver=xtrx", tx_freq=402_500_000, bw=800_000,
        allow_narrow_bw=False, antenna="", gain=None, other_settings="PAD=0", ppm=0.0,
    )
    dev = Dev()
    tx_gfsk._configure_tx(dev, args, 0, 2_044_800.0)
    assert dev.set_gain_calls == []


def test_transmit_timed_first_write_uses_post_activate_hardware_time(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "SoapySDR",
        SimpleNamespace(
            SOAPY_SDR_TX=1,
            SOAPY_SDR_CF32="CF32",
            SOAPY_SDR_END_BURST=2,
            SOAPY_SDR_HAS_TIME=4,
            SOAPY_SDR_TIMEOUT=-1,
        ),
    )

    class Dev:
        def __init__(self):
            self.hw_time = 0
            self.calls = []

        def setupStream(self, *args):
            self.calls.append(("setup", args))
            return "stream"

        def getStreamMTU(self, _stream):
            return 4

        def setHardwareTime(self, time_ns, *_args):
            self.calls.append(("set_time", time_ns))
            self.hw_time = time_ns

        def activateStream(self, *args):
            self.calls.append(("activate", args))
            self.hw_time = 123_000

        def getHardwareTime(self, *_args):
            return self.hw_time

        def writeStream(self, *args):
            self.calls.append(("write", args))
            return SimpleNamespace(ret=args[2], flags=args[3])

        def readStreamStatus(self, *_args, **_kwargs):
            return SimpleNamespace(ret=0, flags=0)

        def deactivateStream(self, *_args):
            pass

        def closeStream(self, *_args):
            pass

    dev = Dev()
    tx_gfsk._transmit(
        dev, 0, np.ones(4, dtype=np.complex64),
        repeat=1, gap_s=0.0, sample_rate=1_000_000.0,
        tx_chunk=4, write_call="full", write_timeout_us=250_000,
        copy_chunks=False, pace=False, write_sleep_us=0,
        stream_channels="default", activate_elems_mode="0",
        tx_format="cf32", tx_scale=1.0,
        tx_time_mode="reset", tx_time_lead_ms=50.0,
    )
    write_args = next(args for name, args in dev.calls if name == "write")
    assert write_args[3] == 6  # END_BURST | HAS_TIME
    assert write_args[4] == 50_123_000


def test_write_stream_call_shapes():
    class Dev:
        def __init__(self):
            self.calls = []

        def writeStream(self, *args):
            self.calls.append(args)
            return SimpleNamespace(ret=3, flags=0)

    block = np.zeros(3, dtype=np.complex64)

    dev = Dev()
    tx_gfsk._write_stream_call(
        dev, "stream", block, 3, flags=9, time_ns=77, timeout_us=123, write_call="simple"
    )
    call = dev.calls[-1]
    assert call[0] == "stream"
    assert call[1][0] is block
    assert call[2:] == (3,)

    tx_gfsk._write_stream_call(
        dev, "stream", block, 3, flags=9, time_ns=77, timeout_us=123, write_call="flags"
    )
    call = dev.calls[-1]
    assert call[0] == "stream"
    assert call[1][0] is block
    assert call[2:] == (3, 9)

    tx_gfsk._write_stream_call(
        dev, "stream", block, 3, flags=9, time_ns=77, timeout_us=123, write_call="full"
    )
    call = dev.calls[-1]
    assert call[0] == "stream"
    assert call[1][0] is block
    assert call[2:] == (3, 9, 77, 123)


def test_release_soapy_device_prefers_close_method():
    class Dev:
        closed = False

        def close(self):
            self.closed = True

    dev = Dev()
    soapy = SimpleNamespace(Device=SimpleNamespace(unmake=lambda d: setattr(d, "unmade", True)))
    assert tx_gfsk._release_soapy_device(soapy, dev)
    assert dev.closed
    assert not hasattr(dev, "unmade")


def test_release_soapy_device_uses_unmake_fallback():
    calls = []

    class Device:
        @staticmethod
        def unmake(dev):
            calls.append(dev)

    dev = SimpleNamespace(thisown=True)
    assert tx_gfsk._release_soapy_device(SimpleNamespace(Device=Device), dev)
    assert calls == [dev]
    assert not dev.thisown


def test_raw_over_cap_refused(tmp_path):
    pf = tmp_path / "huge.bin"
    pf.write_bytes(bytes(tx_gfsk.RAW_MAX_BYTES + 1))
    rc = tx_gfsk.main([
        "--payload-file", str(pf), "--framing", "raw", "--out-file", str(tmp_path / "o.cf32"),
    ])
    assert rc == 2
    assert not (tmp_path / "o.cf32").exists()


def test_raw_1kb_renders(tmp_path):
    pf = tmp_path / "cmd.bin"
    pf.write_bytes(bytes((np.arange(1024) % 256).astype(np.uint8).tolist()))  # 1 KB, within cap
    out = tmp_path / "raw.cf32"
    rc = tx_gfsk.main([
        "--payload-file", str(pf), "--framing", "raw", "--samp-rate", "96000",
        "--out-file", str(out),
    ])
    assert rc == 0
    assert len(np.fromfile(out, dtype=np.complex64)) == 1024 * 8 * 10  # bits × sps(=10 at 96k/9600)


def test_build_frame_iq_unknown_framing_raises():
    with pytest.raises(ValueError, match="unknown framing"):
        tx_gfsk.build_frame_iq(b"x", framing="bogus", sample_rate=_SR, symbol_rate=9600.0,
                               mod_index=0.5, bt=0.5)


def test_out_file_renders_decodable_cf32(tmp_path):
    pf = tmp_path / "payload.bin"
    pf.write_bytes(b"telemetry-request-01")
    out = tmp_path / "tx.cf32"
    rc = tx_gfsk.main([
        "--payload-file", str(pf), "--framing", "endurosat",
        "--samp-rate", "96000", "--out-file", str(out),
    ])
    assert rc == 0 and out.is_file()
    iq = np.fromfile(out, dtype=np.complex64)
    assert b"telemetry-request-01" in endurosat_link.receive(_guard(iq), 96_000.0)


def test_oversize_payload_refused(tmp_path):
    # A file larger than the EnduroSat max must be REFUSED (not silently truncated + transmitted) —
    # this is exactly the ipos_pass_*.bin (9660 B) footgun.
    pf = tmp_path / "big.bin"
    pf.write_bytes(bytes(9660))
    rc = tx_gfsk.main([
        "--payload-file", str(pf), "--framing", "endurosat", "--out-file", str(tmp_path / "o.cf32"),
    ])
    assert rc == 2
    assert not (tmp_path / "o.cf32").exists()  # nothing rendered/sent


def test_oversize_payload_allow_truncate(tmp_path):
    pf = tmp_path / "big.bin"
    pf.write_bytes(bytes(9660))
    out = tmp_path / "o.cf32"
    rc = tx_gfsk.main([
        "--payload-file", str(pf), "--framing", "endurosat", "--samp-rate", "96000",
        "--allow-truncate", "--out-file", str(out),
    ])
    assert rc == 0 and out.is_file()  # knowingly sends the first MAX_PAYLOAD bytes


def test_missing_payload_file_returns_error(tmp_path):
    rc = tx_gfsk.main([
        "--payload-file", str(tmp_path / "nope.bin"), "--out-file", str(tmp_path / "o.cf32"),
    ])
    assert rc == 2


def test_transmit_needs_tx_freq(tmp_path):
    # Without --out-file, keying the SDR requires --tx-freq; absent it errors (no SoapySDR touched).
    pf = tmp_path / "p.bin"
    pf.write_bytes(b"x")
    assert tx_gfsk.main(["--payload-file", str(pf)]) == 2
