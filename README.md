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
    └── gfsk_ax25/                # shared, unit-tested DSP + AX.25 protocol library
```

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

Install on Debian/Ubuntu::

    sudo apt install gnuradio gr-soapy soapysdr-tools \
        soapysdr-module-lms7 soapysdr-module-loopback python3-numpy

## Build / install

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo make install
```

This installs the Python scripts to
``${CMAKE_INSTALL_PREFIX}/opt/gs-flowgraphs/bin/`` (default
``/usr/local/opt/gs-flowgraphs/bin/``). The bench ``waveforms.toml``
points ``binary`` at this path.

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

## Build

```bash
mkdir build && cd build
cmake ..
make
sudo make install
```

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
