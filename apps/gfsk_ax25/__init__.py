"""GFSK / AX.25 receive + transmit DSP and protocol library.

A self-contained, GNU-Radio-independent implementation of the physical and
link layers used by EnduroSat-class UHF cubesat radios (2-GFSK modulation,
G3RUH scrambling, NRZI, HDLC framing, AX.25 UI frames). It is pure
numpy/scipy so the whole modulate -> channel -> demodulate -> deframe chain is
unit-testable on a developer host with no SDR or GNU Radio present.

Two consumers share this library (see ``apps/cubesat_gfsk_ax25_rx.py``):

* the ``dsp`` engine — runs :func:`gfsk.demodulate` here, end to end in numpy;
* the ``gnuradio`` engine — runs the IQ->bits front-end in GNU Radio on the
  bench, then hands the recovered bitstream to :func:`framing.rxbits_to_frames`,
  so the scrambler/NRZI/HDLC/AX.25 protocol layer is identical and equally
  tested for both engines.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

__all__ = ["ax25", "endurosat", "fcs", "framing", "g3ruh", "gfsk", "hdlc"]
