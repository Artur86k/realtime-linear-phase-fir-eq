"""Headless tests for the limiter port + adaptive sound-improver (no GUI / no
audio device).

Run:  python test_sound_improver.py
Verifies:
  1. Limiter holds output below its ceiling (saturation) for loud input, and
     passes quiet input below threshold essentially unchanged.
  2. The compressor gain computer is monotone (louder -> more reduction).
  3. AnalysisEngine._step produces symmetric (linear-phase) taps and bumps the
     holder version, and reduces gain in a band that is over threshold.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import limiter as lm
import sound_improver as si
from audio_engine import CoeffHolder

FS = 48000


def _db(x):
    return 20.0 * np.log10(max(x, 1e-12))


def test_limiter_ceiling():
    # saturation -12 dB, no makeup -> ceiling ~ -12 dBFS (0.251 linear).
    lim = lm.Limiter(FS, saturation_db=-12.0, knee_db=0.0, output_gain_db=0.0,
                     lookahead_ms=1.0, warmup=False)
    t = np.arange(FS) / FS
    sig = 0.95 * np.sin(2 * np.pi * 200 * t)
    block = 1024
    out_peak = 0.0
    for i in range(0, len(sig) - block, block):
        x = np.stack([sig[i:i + block], sig[i:i + block]], axis=1)
        y = lim.process(x)
        # Ignore the very first blocks (envelope settling + lookahead priming).
        if i > 4 * block:
            out_peak = max(out_peak, float(np.max(np.abs(y))))
    # Ceiling is -12 dBFS; allow a little overshoot from hold/release.
    assert _db(out_peak) < -10.0, f"limiter overshoot: {_db(out_peak):.2f} dBFS"
    print(f"test_limiter_ceiling: peak {_db(out_peak):.2f} dBFS (ceil -12) OK")


def test_limiter_passes_quiet():
    # Quiet signal below saturation, no makeup -> ~unchanged.
    lim = lm.Limiter(FS, saturation_db=-6.0, knee_db=0.0, output_gain_db=0.0,
                     lookahead_ms=0.5, warmup=False)
    t = np.arange(FS) / FS
    amp = 0.1                                   # -20 dBFS, well below -6
    sig = amp * np.sin(2 * np.pi * 200 * t)
    block = 1024
    peak = 0.0
    for i in range(0, len(sig) - block, block):
        x = np.stack([sig[i:i + block], sig[i:i + block]], axis=1)
        y = lim.process(x)
        if i > 4 * block:
            peak = max(peak, float(np.max(np.abs(y))))
    assert abs(peak - amp) < 0.02, f"quiet signal altered: {peak:.3f} vs {amp}"
    print(f"test_limiter_passes_quiet: peak {peak:.3f} ~= {amp} OK")


def test_gain_computer_monotone():
    grs = [si._compress_gr_db(L, -30.0, 3.0, 6.0)
           for L in (-40, -30, -20, -10, 0)]
    # Below threshold -> 0; above -> increasingly negative.
    assert grs[0] == 0.0
    assert all(grs[i] >= grs[i + 1] for i in range(len(grs) - 1)), grs
    assert grs[-1] < grs[1], grs
    print(f"test_gain_computer_monotone: {[round(g, 1) for g in grs]} OK")


class _StubEngine:
    """Minimal stand-in for AudioEngine for testing the analysis step."""
    def __init__(self, fs, window):
        self.fs = fs
        self._window = window
        self.limiter = None

    def get_analysis_window(self, win):
        return self._window


def test_analysis_step():
    N, M, beta = 2047, 8192, 8.0
    win_len = 4096
    t = np.arange(win_len) / FS
    # Loud low-mid tone (200 Hz) so the band over threshold gets reduced.
    x = 0.9 * np.sin(2 * np.pi * 200 * t)
    window = np.stack([x, x], axis=1)

    holder = CoeffHolder(np.zeros(N))
    eng = _StubEngine(FS, window)
    improver = si.AnalysisEngine(eng, holder, N, M, beta, win_len=win_len,
                                 threshold_db=-40.0, ratio=4.0)
    improver.set_enabled(True)

    v0 = holder.version
    for _ in range(50):                         # let attack settle
        improver._step(window, dt=0.0125)
    b = holder.b_target

    assert holder.version > v0, "holder version not bumped"
    assert len(b) == N and len(b) % 2 == 1, "taps not odd length"
    asym = float(np.max(np.abs(b - b[::-1])))
    assert asym < 1e-9, f"taps not symmetric (asym={asym:.2e})"
    # The 200 Hz band (index 1: 120-350 Hz) should have pulled gain down.
    assert improver.gr_state[1] < -0.5, f"no reduction: {improver.gr_state}"
    print(f"test_analysis_step: symmetric taps, band1 GR "
          f"{improver.gr_state[1]:.1f} dB OK")


if __name__ == "__main__":
    test_limiter_ceiling()
    test_limiter_passes_quiet()
    test_gain_computer_monotone()
    test_analysis_step()
    print("\nAll sound-improver tests passed.")
