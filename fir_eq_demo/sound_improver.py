"""Adaptive multiband leveler / compressor that "improves" a voice toward a
broadcast ("radio voice") target by continuously rewriting the EQ curve.

The key idea (no separate crossover filterbank needed): in a linear-phase FIR
the magnitude curve *is* the per-frequency gain, so a "band" is just a region
of that curve and a multiband compressor is per-band gain that reacts to
per-band level over time. This engine runs off the audio thread, measures the
input spectrum, computes per-band gain reduction with attack/release time
constants, adds a fixed tone curve, and hands the new taps to the existing
lock-free ``CoeffHolder`` — the audio thread's crossfade de-zippers the swap.

Because every redesigned curve shares the same N and linear phase (identical
group delay), consecutive swaps are phase-coherent: no combing during the
continuous motion. The minimum effective attack is one audio block (~21 ms at
1024/48k) — ideal for density/leveling, not for fast peak control (that is the
downstream limiter's job).

No GUI deps. ``AnalysisEngine`` owns its own worker thread.
"""

from __future__ import annotations

import threading
import time

import numpy as np

try:
    from filter_design import design_fir, FMIN, FMAX
except ImportError:                     # when imported as part of a package
    from .filter_design import design_fir, FMIN, FMAX

# ---- Band layout (voice-oriented) ----------------------------------------
# Edges in Hz -> 6 bands. Centres are geometric means of each edge pair.
BAND_EDGES = (20.0, 120.0, 350.0, 900.0, 2500.0, 6000.0, 20000.0)

# Fixed "radio voice" tone target (dB at frequency). Drives timbre; the
# compressor's per-band reduction is *added* on top of this. A strong low
# shelf stands in for a rumble high-pass; gentle low-mid warmth; presence lift
# for intelligibility; a touch of de-ess above 6 kHz.
RADIO_TARGET = (
    (20.0, -15.0), (45.0, -10.0), (80.0, -3.0), (160.0, 1.5), (300.0, 1.0),
    (1000.0, 0.0), (3000.0, 3.5), (5000.0, 2.0), (7500.0, -2.0),
    (12000.0, -1.0), (20000.0, -3.0),
)


def _band_centers(edges):
    e = np.asarray(edges, dtype=np.float64)
    return np.sqrt(e[:-1] * e[1:])


def _static_db_at(freq, target=RADIO_TARGET):
    """Linear-in-(log f, dB) sample of the fixed tone target at one frequency."""
    fx = np.array([p[0] for p in target])
    gy = np.array([p[1] for p in target])
    return float(np.interp(np.log10(freq), np.log10(fx), gy))


def _compress_gr_db(level_db, threshold_db, ratio, knee_db):
    """Downward-compressor gain computer. Returns gain reduction in dB (<= 0)."""
    over = level_db - threshold_db
    if knee_db > 0.0 and abs(over) <= knee_db / 2.0:
        # Quadratic soft knee.
        k = over + knee_db / 2.0
        return (1.0 / ratio - 1.0) * (k * k) / (2.0 * knee_db)
    if over > 0.0:
        return (1.0 / ratio - 1.0) * over
    return 0.0


class AnalysisEngine:
    """Drives the EQ curve from a live analysis of the played audio.

    engine : AudioEngine — provides ``get_analysis_window`` + ``fs`` and (when
             present) ``limiter`` to push spectral hints to.
    holder : CoeffHolder — the same lock-free handoff the GUI uses.
    """

    def __init__(self, engine, holder, n_taps, m_design, kaiser_beta,
                 update_hz=80.0, win_len=4096,
                 threshold_db=-30.0, ratio=3.0, knee_db=6.0,
                 attack_ms=30.0, release_ms=250.0, makeup_db=0.0,
                 on_curve=None):
        self.engine = engine
        self.holder = holder
        self.N = n_taps
        self.M = m_design
        self.beta = kaiser_beta

        self.update_interval = 1.0 / update_hz
        self.win_len = win_len

        # Compressor parameters (shared across bands; could be per-band).
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.knee_db = knee_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.makeup_db = makeup_db

        self.centers = _band_centers(BAND_EDGES)
        self.n_bands = len(self.centers)
        self.static_db = np.array([_static_db_at(f) for f in self.centers])
        self.static_lo = _static_db_at(FMIN)
        self.static_hi = _static_db_at(FMAX)

        # Smoothed per-band gain reduction state (dB, <= 0).
        self.gr_state = np.zeros(self.n_bands)
        self.band_level_db = np.full(self.n_bands, -120.0)

        self._win = np.hanning(win_len)
        self._win_norm = np.sum(self._win ** 2)
        self._freqs = np.fft.rfftfreq(win_len, d=1.0 / engine.fs)
        # Precompute band bin masks.
        self._band_bins = [
            np.where((self._freqs >= BAND_EDGES[i]) &
                     (self._freqs < BAND_EDGES[i + 1]))[0]
            for i in range(self.n_bands)
        ]

        self.on_curve = on_curve        # optional GUI callback(points, b)
        self.enabled = False
        self._stop = threading.Event()
        self._thread = None

    # ---- lifecycle -------------------------------------------------------
    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="sound-improver")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def set_enabled(self, on):
        if on and not self.enabled:
            self.gr_state[:] = 0.0       # start from no reduction
        self.enabled = bool(on)

    # ---- the analysis loop ----------------------------------------------
    def _run(self):
        last = time.monotonic()
        while not self._stop.is_set():
            time.sleep(self.update_interval)
            if not self.enabled:
                last = time.monotonic()
                continue

            win = self.engine.get_analysis_window(self.win_len)
            now = time.monotonic()
            dt = now - last
            last = now
            if win is None:
                continue

            self._step(win, dt)

    def _step(self, win, dt):
        # Mono detector (sum the channels), windowed power spectrum.
        x = win[:, 0] + win[:, 1]
        xw = x * self._win
        mag2 = np.abs(np.fft.rfft(xw)) ** 2

        # Per-band level (dBFS-ish; thresholds are tunable against this scale).
        eps = 1e-12
        for b, bins in enumerate(self._band_bins):
            if bins.size:
                power = 2.0 * np.sum(mag2[bins]) / (self.win_len * self._win_norm)
                self.band_level_db[b] = 10.0 * np.log10(power + eps)
            else:
                self.band_level_db[b] = -120.0

        # Per-band gain computer + time-aware attack/release smoothing.
        a_atk = np.exp(-dt / (self.attack_ms / 1000.0))
        a_rel = np.exp(-dt / (self.release_ms / 1000.0))
        for b in range(self.n_bands):
            target = _compress_gr_db(self.band_level_db[b], self.threshold_db,
                                     self.ratio, self.knee_db)
            # Attack = reduction getting deeper (target below state) -> fast.
            coeff = a_atk if target < self.gr_state[b] else a_rel
            self.gr_state[b] = target + (self.gr_state[b] - target) * coeff

        # Assemble control points: anchors + per-band (static tone + reduction
        # + makeup). PCHIP through the band centres gives smooth overlap.
        points = [(FMIN, self.static_lo)]
        for b in range(self.n_bands):
            db = self.static_db[b] + self.gr_state[b] + self.makeup_db
            points.append((float(self.centers[b]), float(db)))
        points.append((FMAX, self.static_hi))

        b_taps = design_fir(points, self.engine.fs, self.N, self.M, self.beta)
        self.holder.update(b_taps)

        # Feed spectral hints to the limiter (shapes its adaptive recovery).
        lim = getattr(self.engine, "limiter", None)
        if lim is not None:
            mag = np.sqrt(mag2)
            total = np.sum(mag) + eps
            centroid = float(np.sum(self._freqs * mag) / total)
            low_bins = np.where(self._freqs < 200.0)[0]
            low_energy = float(np.sum(mag2[low_bins]) / (np.sum(mag2) + eps))
            rms = float(np.sqrt(np.mean(x ** 2)))
            lim.set_spectral(centroid=centroid, low_energy=low_energy, rms=rms)

        if self.on_curve is not None:
            self.on_curve(points, b_taps)
