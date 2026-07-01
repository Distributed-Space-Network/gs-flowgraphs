# gs-flowgraphs

GNU Radio out-of-tree (OOT) module providing the DSP flowgraphs invoked by the
DSN ground station client (`gs-client`) as separate subprocesses.

**License: GPLv3.** This repository must remain GPL-compatible because GNU
Radio is GPLv3 and any code linking GNU Radio is a derivative work. This
repository is intentionally separated from the closed-source `gs-client`
orchestrator — they communicate over Unix domain sockets only, never linking.

## Repository status

Phase 5 — first real Python-GR flowgraphs (NBFM RX/TX) land alongside
the Phase 3/5 placeholders (``stub_rx.py``, ``stub_tx.py``). All four
share the same spawn-contract boilerplate (``_spawn_contract.py``)
and CLI interface, so the gs-client orchestrator does not distinguish
them at spawn time.

```
gs-flowgraphs/
├── README.md
├── COPYING                       # GPLv3 full text
├── CMakeLists.txt                # ``install``-only at present
├── pyproject.toml                # dev/test toolchain for the pure-Python DSP lib
├── tests/                        # pytest suite for apps/gfsk_ax25 (numpy/scipy)
└── apps/
    ├── _spawn_contract.py        # NDJSON sockets, argparse, command loop
    ├── stub_rx.py                # Python placeholder for orchestrator E2E tests
    ├── stub_tx.py                # Python placeholder for orchestrator E2E tests
    ├── amateur_fm_narrowband_rx.py   # real NBFM receive (Phase 5)
    ├── amateur_fm_narrowband_tx.py   # real NBFM transmit (Phase 5, test-tone)
    ├── cubesat_gfsk_ax25_rx.py   # 2-GFSK/AX.25 9k6 receive (dsp | gnuradio engines)
    ├── cubesat_gfsk_ax25_tx.py   # 2-GFSK/AX.25 9k6 transmit (dsp | gnuradio engines)
    ├── gnuradio_gfsk.py          # GNU Radio front-end for the cubesat apps (bench engine)
    ├── satellite_rx.py           # multi-mission gr-satellites receiver (bench)
    ├── gnuradio_satellites.py    # gr-satellites bridge -> spawn contract (bench)
    └── gfsk_ax25/                # shared, unit-tested DSP + AX.25 protocol library
```

## Multi-mission decoding (gr-satellites)

To be "prepared for everything", non-EnduroSat missions decode via **gr-satellites**
(GPLv3) — the canonical library of public satellite framers/deframers across
AFSK/FSK/GFSK/BPSK/GMSK/… with AX.25/AX.100/Mobitex/CCSDS/… framings. We *integrate*
it rather than reimplement it: `satellite_rx.py` is a spawn-contract app that runs
gr-satellites for a selected `satellite` (SatYAML id) and emits decoded frames, via
the `gnuradio_satellites.py` bridge. **Bench-pending** (needs GNU Radio +
gr-satellites + gr-soapy). Design + licensing (gr-satellites GPLv3; *not* the AGPL
gr-satnogs/satnogs-client): see `../docs/07-multi-mission-framing.md`. The EnduroSat
mission keeps its dedicated, tested `dsp` path (`cubesat_gfsk_ax25_rx.py
--framing endurosat`).

## Universal modem + framing (docs/08)

Beyond gr-satellites, the engine is a **three-registry composer** so it can demodulate + deframe
*any documented downlink* (commercial/government/amateur), not just amateur AX.25. See
[`../docs/08-universal-modem-framing-plan.md`](../docs/08-universal-modem-framing-plan.md) (plan) and
[`../docs/09-universal-modem-framing-integration-changes.md`](../docs/09-universal-modem-framing-integration-changes.md)
(the deferred backend/frontend/config changes it implies).

* **Modem registry** — [`apps/modem.py`](apps/modem.py): `modulation_spec()` classifies the full
  taxonomy — **Tier 1** FSK (2-FSK/GFSK/GMSK/MSK/CPFSK, M-FSK) · PSK (BPSK/DBPSK, QPSK/DQPSK/OQPSK,
  8-PSK) · AFSK; **Tier 2** QAM 16–256 · APSK 16/32 · OFDM · DVB-S2/S2X; **Tier 3** OOK/ASK · CW/Morse
  · NBFM/WFM/AM. `build_demod`/`build_mod` construct the chains (Tier-1 in
  [`gnuradio_gfsk.py`](apps/gnuradio_gfsk.py); Tier-2/analog in
  [`gnuradio_hirate.py`](apps/gnuradio_hirate.py)).
* **FEC registry** — [`apps/fec.py`](apps/fec.py): numpy CCSDS randomizer, CRC-16/32, ASM, and
  **Reed-Solomon RS(255,223)** ([`gfsk_ax25/reedsolomon.py`](apps/gfsk_ax25/reedsolomon.py)); Viterbi
  / LDPC / Turbo / Golay declared for the bench / gr-satellites.
* **Framing registry** — [`apps/framings.py`](apps/framings.py): local numpy deframers
  (`ax25`, `endurosat`, **`argos`** BCH(31,21) PTT, **`ccsds_tm`** CCSDS TM/AOS transfer frame,
  **`kiss`**/**`slip`** TNC), plus the whole gr-satellites vocabulary reused via a synthetic SatYAML
  ([`grsat_synth.py`](apps/grsat_synth.py)).
* **Composer** — [`apps/compose.py`](apps/compose.py): `plan_decode(rfLink)` → our-engine /
  gr-satellites / race / record-only.

The numpy codecs (RS, BCH, CCSDS, OOK, Morse, KISS/SLIP) are exhaustively unit-tested; the GNU-Radio
demod/mod chains are **bench-validation-pending** (no GNU Radio in CI).

## Cubesat 2-GFSK / AX.25 (9k6) waveform

`cubesat_gfsk_ax25_{rx,tx}.py` implement an EnduroSat-class UHF link: 2-GFSK
(h≈0.5, BT≈0.5), a 12 480 sym/s channel (~9 600 bps user, no FEC), G3RUH
scrambling + NRZI, AX.25 UI framing over HDLC, ~18.7 kHz occupied bandwidth at
401.5 MHz. The link parameters live in
[`apps/gfsk_ax25/endurosat.py`](apps/gfsk_ax25/endurosat.py).

**Two framings**, chosen by `--`/`GS_FLOWGRAPH_FRAMING` env / params `framing`
(default `ax25`):

* **`ax25`** — AX.25 UI over HDLC with G3RUH/NRZI (generic; for AX.25 satellites).
* **`endurosat`** — the EnduroSat chip packet in
  [`apps/gfsk_ax25/endurosat_link.py`](apps/gfsk_ax25/endurosat_link.py):
  `0xAA` preamble + `0x7E` sync + length + payload + CRC-16/CCITT, **9 600 sym/s**
  (dev ±2400, h=0.5, BT=0.5). This is the real Gen-2 link (matches gr-satellites'
  `endurosat_deframer`). The payload is the EnduroSat **AirMAC** frame, which is
  AES-encrypted and parsed by the closed orchestrator — this repo only
  receives/transmits the link packet. Capture analysis: `tools/iq_analyze.py`.

**Two interchangeable engines**, chosen by `--engine {dsp,gnuradio}`, the
`GS_FLOWGRAPH_ENGINE` env var, or a params-file `engine` key (default `dsp`):

* **`dsp`** — pure numpy/scipy modem in `apps/gfsk_ax25` (modulate, demodulate,
  Gardner timing recovery, scramble/NRZI/HDLC/AX.25). Needs no SDR or GNU Radio;
  IQ comes from SoapySDR (`--sdr-args driver=...`) or a `cf32` file
  (`--sdr-args file:/path.cf32`). The whole chain is unit-tested.
* **`gnuradio`** — GNU Radio front-end (`gnuradio_gfsk.py`) for the bench, which
  hands the recovered bitstream to the **same** `gfsk_ax25.framing` deframer, so
  both engines decode identically. Bench-validate as in the NBFM recipe below.

### SDR front-end parameters (gnuradio engine + `satellite_rx.py`)

GNU Radio gives the demod, not the SDR front-end, so the gr-soapy source must be
told its gain/antenna or it sits near 0 dB and hears nothing. These
`waveform_parameters` (params-file) keys configure it (`apps/_soapy.py`); if none
are given, a 30 dB manual gain is applied so the front-end is never silent:

* `sdr_antenna` (str) — antenna port, e.g. `"LNAL"`, `"RX2"`.
* `sdr_agc` (bool) — hardware AGC on/off (when on, no manual gain is forced).
* `sdr_gain_db` (number) — overall manual gain in dB.
* `sdr_gains` (table) — per-element gains, e.g. `{ LNA = 20, TIA = 6, PGA = 0 }`.

Run the DSP library tests (proves a frame survives modulate → channel with
AWGN/Doppler/timing → demodulate → deframe):

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Reserved for later phases:
* ``grc/`` — GRC-authored flowgraphs (GUI-built XML)
* ``lib/`` + ``include/`` — custom C++ blocks for performance-critical DSP

## Runtime requirements (bench / Renesas target)

* GNU Radio 3.10+ with Python bindings
* ``gr-soapy`` (a.k.a. ``gnuradio-soapy`` on Debian/Ubuntu)
* SoapySDR + at least one device module (``soapysdr-module-lms7`` for
  LimeSDR-mini, ``soapysdr-module-xtrx`` for XTRX, etc.)
* For dev without real hardware: ``soapysdr-module-loopback``
* **gr-satellites** (GPLv3) — the multi-mission decode engine (`satellite_rx.py`).
* **Optional: gr-dvbs2rx** (GPLv3) — only for **DVB-S2/S2X** downlinks (the one core modulation
  gap). Absence is handled gracefully — `gnuradio_hirate.connect_dvbs2_demod` returns None and the
  modulation is treated as build-pending. Install only on stations that service DVB-S2 birds.

Install on Debian/Ubuntu::

    sudo apt install gnuradio gr-soapy soapysdr-tools \
        soapysdr-module-lms7 soapysdr-module-loopback python3-numpy

## Install (staging, not compiling)

Nothing here is compiled — the CMake project is ``LANGUAGES NONE``, so
``make install`` just *copies* the scripts (and the shared ``gfsk_ax25/`` lib +
``_spawn_contract.py``/``_soapy.py`` helpers) into one directory. You can also run
a script straight from ``apps/``; installing only fixes the path.

```bash
mkdir build && cd build
cmake ..
sudo make install        # -> /opt/gs-flowgraphs/bin/
```

It installs to ``/opt/gs-flowgraphs/bin/`` — the path the client's
``waveforms.toml`` points at (``/opt/<pkg>`` is the app's own FHS tree). Override
with ``-DGS_APPS_INSTALL_DIR=...``; staged/packaged installs use ``make install
DESTDIR=/tmp/stage`` (lands under ``/tmp/stage/opt/gs-flowgraphs/bin``).

## Verification on Linux (Phase 6 bench prep)

The orchestrator E2E tests use the ``stub_*.py`` files via a
supervisor patch; they do not exercise GNU Radio. To verify the
real flowgraphs work, run them manually against a SoapyLoopback
device:

```bash
# Create the three TCP listeners (any ports):
nc -lk 5001 &   # control
nc -lk 5002 &   # status
nc -lk 5003 >/tmp/audio.raw &   # data

# In another shell:
python3 apps/amateur_fm_narrowband_rx.py \
    --waveform-id amateur.fm.narrowband \
    --direction rx \
    --center-freq-hz 437800000 \
    --bandwidth-hz 25000 \
    --sample-rate 2000000 \
    --sdr-driver soapy \
    --sdr-args "driver=loopback" \
    --sdr-port RX1 \
    --control-socket tcp://127.0.0.1:5001 \
    --status-socket tcp://127.0.0.1:5002 \
    --data-socket tcp://127.0.0.1:5003 \
    --output-dir /tmp/

# On the control socket, send: {"cmd":"start"}
# After observing /tmp/audio.raw growing, send: {"cmd":"stop"}
```

Same shape for the TX flowgraph; the data socket receives nothing
(TX first-cut emits a tone, not a data stream).

## Spawn contract

Each flowgraph binary obeys the spawn contract documented in
[Document A section A.7.2](../docs/02-client-behavior-spec.md). The
orchestrator passes parameters via command-line flags and a `--params-file`
protobuf, and the flowgraph emits structured events on its `--status-socket`
and (where applicable) frame data on `--data-socket`.

## Versioning

Flowgraphs are versioned independently of the orchestrator. Each binary
supports `--version` and reports a version string the orchestrator records
per pass. Version mismatches between expected (in `waveforms.toml`) and
actual cause refusal to spawn.
