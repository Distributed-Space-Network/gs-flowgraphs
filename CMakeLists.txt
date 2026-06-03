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

# Install path. Bench expectation per docs/04-implementation-plan.md
# is ``/opt/gs-flowgraphs/bin/<binary>``; the path is configurable
# via -DCMAKE_INSTALL_PREFIX=...
include(GNUInstallDirs)
set(GS_APPS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/opt/gs-flowgraphs/bin"
    CACHE PATH "Install dir for flowgraph executables")

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
  DESTINATION ${GS_APPS_INSTALL_DIR}
)
install(
  FILES apps/_spawn_contract.py
  DESTINATION ${GS_APPS_INSTALL_DIR}
)

# add_subdirectory(lib) — reserved for future C++ blocks
# add_subdirectory(grc) — reserved for GRC-authored flowgraphs

message(STATUS "gs-flowgraphs ${GS_FLOWGRAPHS_VERSION} — Python flowgraphs only (Phase 5)")
message(STATUS "  install dir: ${GS_APPS_INSTALL_DIR}")
