"""Linear-phase FIR design from an EQ curve.

Pure DSP, no GUI / no audio deps. Unit-testable.

The design uses the *oversampled frequency-sampling + window* method:
curve resolution (M) is decoupled from the tap count (N), and a Kaiser
window suppresses the Gibbs ripple between control points.

The resulting taps are symmetric (Type I FIR, odd N) => exact linear phase,
constant group delay (N-1)/2.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

# ---- Defaults (mirrored in main.py constants) ----------------------------
N_TAPS = 2047       # odd -> Type I linear phase
M_DESIGN = 8192     # oversampled IFFT length (>= ~4*N)
KAISER_BETA = 8.0
FMIN, FMAX = 20.0, 20000.0
GAIN_MIN_DB, GAIN_MAX_DB = -20.0, 20.0


def interpolate_db(points, freqs):
    """Interpolate an EQ curve in (log10 f, dB) space.

    points : iterable of (freq_hz, gain_db). Need not be sorted.
    freqs  : array of frequencies (Hz) to evaluate at (must be > 0).

    Returns dB gain at each freq. The curve is held flat below the first
    point and above the last point (flat extension to DC / Nyquist).

    Default interpolant is PCHIP (monotone, no overshoot between widely
    spaced points). Catmull-Rom in (log f, dB) is the "true Bezier-chain"
    alternative that also passes through every point but can overshoot.
    """
    pts = sorted(points, key=lambda p: p[0])
    fx = np.array([p[0] for p in pts], dtype=np.float64)
    gy = np.array([p[1] for p in pts], dtype=np.float64)

    freqs = np.asarray(freqs, dtype=np.float64)
    logf = np.log10(np.maximum(freqs, 1e-6))

    if len(pts) == 1:
        return np.full_like(freqs, gy[0])

    # Collapse duplicate frequencies (PCHIP needs strictly increasing x).
    logx = np.log10(fx)
    uniq_mask = np.concatenate(([True], np.diff(logx) > 1e-12))
    logx = logx[uniq_mask]
    gy = gy[uniq_mask]

    if len(logx) == 1:
        return np.full_like(freqs, gy[0])

    interp = PchipInterpolator(logx, gy, extrapolate=False)
    out = interp(logf)
    # Flat extension outside the control-point span.
    out = np.where(logf < logx[0], gy[0], out)
    out = np.where(logf > logx[-1], gy[-1], out)
    return out


def design_fir(points, fs, N=N_TAPS, M=M_DESIGN, beta=KAISER_BETA):
    """Design a linear-phase FIR whose magnitude follows the EQ curve.

    points : iterable of (freq_hz, gain_db).
    fs     : sample rate (Hz).
    N      : number of taps (odd recommended for Type I).
    M      : oversampled design FFT length.

    Returns b : float64 ndarray of length N, symmetric (linear phase).
    """
    if N % 2 == 0:
        N += 1  # force odd for Type I symmetry / exact linear phase

    # rfft bin frequencies on the oversampled grid.
    k = np.arange(M // 2 + 1)
    f_k = k * fs / M

    db = interpolate_db(points, np.maximum(f_k, 1e-6))
    db = np.clip(db, GAIN_MIN_DB, GAIN_MAX_DB)
    H = 10.0 ** (db / 20.0)          # real, zero-phase magnitude spectrum

    h_full = np.fft.irfft(H, n=M)    # real impulse response, symmetric about 0
    h = np.fft.fftshift(h_full)      # center the symmetry at M//2

    center = M // 2
    half = N // 2
    h = h[center - half: center + half + 1]   # take center N samples

    w = np.kaiser(N, beta)
    b = (h * w).astype(np.float64)
    return b


def fir_magnitude_db(b, fs, freqs):
    """Measured magnitude (dB) of FIR taps b, sampled at `freqs` (Hz).

    Used by the GUI to overlay the actual FIR response on the target curve.
    """
    N = len(b)
    # Zero-pad for smooth frequency resolution.
    nfft = 1 << int(np.ceil(np.log2(max(N * 4, 4096))))
    H = np.fft.rfft(b, n=nfft)
    f_grid = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag = np.abs(H)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-9))
    return np.interp(freqs, f_grid, mag_db)
