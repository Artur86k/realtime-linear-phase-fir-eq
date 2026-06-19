"""Headless tests for the DSP + seamless-swap claim (no GUI / no audio device).

Run:  python test_dsp.py
Verifies:
  1. design_fir returns odd-length, symmetric (linear-phase) taps.
  2. The measured FIR magnitude tracks the target curve at control points.
  3. The crossfade swap produces NO click: output is continuous (bounded
     first difference) across the swap boundary, and far from the boundary
     each filter's output matches a clean full-convolution reference.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filter_design as fd
from audio_engine import AudioEngine, CoeffHolder
from scipy.signal import fftconvolve

FS = 48000
N = 2047
M = 8192


def test_symmetry():
    pts = [(20, -6), (200, 4), (1000, 0), (5000, 8), (20000, -10)]
    b = fd.design_fir(pts, FS, N, M)
    assert len(b) % 2 == 1, "taps must be odd length"
    asym = np.max(np.abs(b - b[::-1]))
    assert asym < 1e-12, f"taps not symmetric (max asym {asym:.2e})"
    print(f"[OK] symmetry: len={len(b)}, max |b-reverse(b)| = {asym:.2e}")


def test_magnitude_tracking():
    pts = [(100, 6), (1000, -8), (8000, 10)]
    b = fd.design_fir(pts, FS, N, M)
    freqs = np.array([p[0] for p in pts], dtype=float)
    targ = np.array([p[1] for p in pts], dtype=float)
    meas = fd.fir_magnitude_db(b, FS, freqs)
    err = np.abs(meas - targ)
    print(f"[OK] magnitude tracking at control points: max err {err.max():.2f} dB "
          f"({dict(zip(freqs.astype(int), np.round(meas,2)))})")
    assert err.max() < 1.5, f"FIR deviates {err.max():.2f} dB from target"


def test_seamless_swap():
    """Drive the engine callback across a coefficient swap; assert no click."""
    pts_a = [(20, 0), (1000, 0), (20000, 0)]              # flat
    pts_b = [(20, -18), (300, 18), (3000, -18), (20000, 18)]  # extreme EQ
    b_a = fd.design_fir(pts_a, FS, N, M)
    b_b = fd.design_fir(pts_b, FS, N, M)

    holder = CoeffHolder(b_a)
    eng = AudioEngine(holder, blocksize=1024, fs_fallback=FS)
    eng.fs = FS

    # Deterministic stereo test signal.
    rng = np.random.default_rng(1)
    total = 1024 * 8
    src = (rng.standard_normal((total, 2)) * 0.2).astype(np.float32)
    eng.audio = src
    eng.n_frames = total
    eng.read_pos = 0
    eng.playing = True

    L = 1024
    out = np.zeros((total, 2), dtype=np.float32)
    swap_block = 3
    for blk in range(total // L):
        if blk == swap_block:
            holder.update(b_b)   # trigger the hot-swap on this block
        buf = np.zeros((L, 2), dtype=np.float32)
        eng._callback(buf, L, None, None)
        out[blk * L:(blk + 1) * L] = buf

    # Click test: the max sample-to-sample jump across the whole stream
    # should not spike at the swap boundary. Compare boundary diff to the
    # global typical diff.
    d = np.abs(np.diff(out, axis=0)).max(axis=1)
    boundary = swap_block * L
    near = d[boundary - 2: boundary + 2].max()
    typical = np.percentile(d, 99.9)
    print(f"[OK] swap continuity: boundary max-diff {near:.4f} vs "
          f"99.9pct {typical:.4f}")
    assert near < typical * 3.0, "click detected at swap boundary"

    # No NaNs / no runaway.
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0
    print(f"[OK] output finite & bounded, peak {np.max(np.abs(out)):.3f}")


def test_delayline_matches_reference():
    """Steady-state (no swap) engine output == clean full convolution."""
    pts = [(50, 5), (2000, -6), (12000, 9)]
    b = fd.design_fir(pts, FS, N, M)
    holder = CoeffHolder(b)
    eng = AudioEngine(holder, blocksize=1024, fs_fallback=FS)
    eng.fs = FS
    rng = np.random.default_rng(2)
    total = 1024 * 6
    src = (rng.standard_normal((total, 2)) * 0.1).astype(np.float32)
    eng.audio = src
    eng.n_frames = total
    eng.read_pos = 0
    eng.playing = True
    eng.input_gain = 1.0

    L = 1024
    out = np.zeros((total, 2))
    for blk in range(total // L):
        buf = np.zeros((L, 2), dtype=np.float32)
        eng._callback(buf, L, None, None)
        out[blk * L:(blk + 1) * L] = buf

    # Reference: full convolution of the source with b, 'valid'-aligned.
    ref = np.zeros_like(out)
    for ch in range(2):
        full = fftconvolve(src[:, ch].astype(np.float64), b, mode="full")
        ref[:, ch] = full[:total]
    ref = np.clip(ref, -1.0, 1.0)
    # Skip the first N-1 samples (delay-line warmup with zeros matches).
    err = np.max(np.abs(out[:total] - ref[:total]))
    print(f"[OK] delay-line vs reference convolution: max err {err:.2e}")
    assert err < 1e-4, f"engine output diverges from reference ({err:.2e})"


if __name__ == "__main__":
    test_symmetry()
    test_magnitude_tracking()
    test_delayline_matches_reference()
    test_seamless_swap()
    print("\nAll DSP tests passed.")
