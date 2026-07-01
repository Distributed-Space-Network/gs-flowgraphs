# gs-flowgraphs — DSN ground-station GNU Radio out-of-tree module.
#
# Phase 5: the first real Python-GR flowgraphs (NBFM RX / TX) shipped
# in ``apps/``. They are plain ``.py`` scripts driven by gr-python +
# gr-soapy at runtime — no C++ compilation needed yet. The install
# step just stages the scripts into the bin / share directories.
#
# Phase 7+ may add custom C++ blocks (``lib/``) and GRC-authored
# flowgraphs (``grc/``); both directories are reserved.
#
# Build dependencies (runtime on the bench / Renesas board):
#   * GNU Radio 3.10+ with the Python bindings
#   * gr-soapy (SoapySDR bridge) — also via gnuradio-soapy on Debian
#   * SoapySDR + at least one device-specific module (lime / xtrx /
#     loopback for dev)
#
# License: GPLv3 (see COPYING).

cmake_minimum_required(VERSION 3.16)
project(gs-flowgraphs LANGUAGES NONE)

set(GS_FLOWGRAPHS_VERSION_MAJOR 0)
set(GS_FLOWGRAPHS_VERSION_MINOR 1)
set(GS_FLOWGRAPHS_VERSION_PATCH 0)
set(GS_FLOWGRAPHS_VERSION
    "${GS_FLOWGRAPHS_VERSION_MAJOR}.${GS_FLOWGRAPHS_VERSION_MINOR}.${GS_FLOWGRAPHS_VERSION_PATCH}")

# Install path. These scripts are NOT compiled (project LANGUAGES NONE);
# ``make install`` only stages them. ``/opt/gs-flowgraphs`` is the application's
# own tree (FHS ``/opt/<pkg>``) and matches the binary paths in the client's
# waveforms.toml, so it is an ABSOLUTE path, not derived from
# CMAKE_INSTALL_PREFIX (which would give the nonsensical /usr/local/opt/...).
# Override with -DGS_APPS_INSTALL_DIR=... ; staged/packaged installs use DESTDIR=.
set(GS_APPS_INSTALL_DIR "/opt/gs-flowgraphs/bin"
    CACHE PATH "Install dir for flowgraph scripts (matches waveforms.toml)")

# All four apps in ``apps/`` are Python scripts honoring the spawn
# contract (Document A §A.7.2):
#   stub_rx.py / stub_tx.py     — placeholders for orchestrator E2E tests
#   amateur_fm_narrowband_rx.py — real NBFM receive
#   amateur_fm_narrowband_tx.py — real NBFM transmit (test-tone first cut)
# All four import _spawn_contract.py for socket / NDJSON boilerplate.
install(
  PROGRAMS
    apps/stub_rx.py
    apps/stub_tx.py
    apps/amateur_fm_narrowband_rx.py
    apps/amateur_fm_narrowband_tx.py
    apps/cubesat_gfsk_ax25_rx.py
    apps/cubesat_gfsk_ax25_tx.py
    apps/satellite_rx.py             # multi-mission gr-satellites receiver (bench)
    apps/iq_views.py                 # post-pass PNG/CSV/SDF from a .cf32 (run by gs-client)
  DESTINATION ${GS_APPS_INSTALL_DIR}
)
install(
  FILES
    apps/_spawn_contract.py
    apps/_soapy.py                   # SoapySDR gain/antenna config, shared by the GR engines
    apps/_recorder.py                # pre-demod IQ capture (cf32/CSV/PNG), imported by the RX engines
    apps/_fallback_select.py         # fallback demod selection (imported by gnuradio_satellites)
    apps/modem.py                    # modem registry: modulation -> demod chain (docs/08)
    apps/framings.py                 # framing registry: framing -> deframe (docs/08)
    apps/fec.py                      # FEC registry skeleton (docs/08)
    apps/grsat_synth.py              # synthetic SatYAML: reuse gr-satellites deframers (docs/08 Ph1)
    apps/gnuradio_gfsk.py            # bench engine helper (imported only by --engine gnuradio)
    apps/gnuradio_satellites.py      # gr-satellites bridge (imported by satellite_rx at decode time)
  DESTINATION ${GS_APPS_INSTALL_DIR}
)
# Shared pure-Python DSP + AX.25 protocol library used by the cubesat apps
# (and by the gnuradio engine's deframer). Staged next to the app scripts so
# ``from gfsk_ax25 import ...`` resolves at runtime, like ``_spawn_contract``.
install(
  DIRECTORY apps/gfsk_ax25
  DESTINATION ${GS_APPS_INSTALL_DIR}
  FILES_MATCHING PATTERN "*.py"
)

# add_subdirectory(lib) — reserved for future C++ blocks
# add_subdirectory(grc) — reserved for GRC-authored flowgraphs

message(STATUS "gs-flowgraphs ${GS_FLOWGRAPHS_VERSION} — Python flowgraphs only (Phase 5)")
message(STATUS "  install dir: ${GS_APPS_INSTALL_DIR}")
