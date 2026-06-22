"""Sample-accurate lookahead limiter — Python port of the AudioWorklet limiter
at https://github.com/Artur86k/limiter (``limiter-worklet.js``).

Faithful to the original DSP:
  - ring-buffer lookahead delay (the gain reacts *before* the peak reaches the
    output, because the envelope is read from the un-delayed signal),
  - instant-attack / adaptive-hold / exponential-decay peak envelope follower,
  - soft-knee gain reduction toward ``saturation_db``,
  - makeup ``output_gain_db`` (the original's "Output Gain"/normalize stage),
  - adaptive recovery + hold driven by optional spectral hints (centroid,
    low-band energy, rms) pushed from the analyser.

The envelope follower has a per-sample feedback dependency (instant attack,
data-dependent hold), so it cannot be vectorised in closed form — it is a tight
scalar loop. If Numba is present the loop is JIT-compiled (and warmed up at
construction so the first audio block never stalls); otherwise it falls back to
the same code running in plain CPython (slower, but identical output).

No GUI / no audio-device deps — unit-testable.
"""

from __future__ import annotations

import math

import numpy as np

# ---- Defaults (mirror the worklet constructor) ---------------------------
SATURATION_DB = -6.0     # limit ceiling before makeup
KNEE_DB = 0.0            # 0 = hard knee
OUTPUT_GAIN_DB = 6.0     # makeup after limiting (the "normalize" stage)
LOOKAHEAD_MS = 0.5
MIN_RECOVERY_MS = 20.0
MAX_LOOKAHEAD_MS = 0.020 * 1000  # 20 ms ring, as in the worklet


def _kernel(inL, inR, outL, outR, ringL, ringR,
            write_index, envelope, hold_timer, peak_rate,
            delay_samples, max_delay,
            sat, knee, out_gain_lin, decay,
            centroid, low_energy, lookahead_ms, fs):
    """Per-sample limiter loop. Mutates outL/outR/ringL/ringR in place and
    returns the updated scalar state ``(write_index, envelope, hold_timer,
    peak_rate)``. A straight transcription of the worklet's ``process()``."""
    n = inL.shape[0]
    for i in range(n):
        sL = inL[i]
        sR = inR[i]
        absL = sL if sL >= 0.0 else -sL
        absR = sR if sR >= 0.0 else -sR
        peak = absL if absL > absR else absR

        # --- Envelope follower ---
        peak_hit = 1.0 if peak >= envelope * 0.9 else 0.0
        peak_rate = peak_rate * 0.9995 + peak_hit * 0.0005

        if peak > envelope:
            # Instant attack.
            envelope = peak
            # Adaptive hold: denser / lower / darker content holds longer.
            density_factor = 1.0 + peak_rate * 4.0
            low_freq_factor = 1.0 + low_energy * 3.0
            hff = 1.0 - centroid / 8000.0
            high_freq_factor = hff if hff > 0.3 else 0.3
            hold_ms = 5.0 * density_factor * low_freq_factor * high_freq_factor
            min_hold_ms = hold_ms if hold_ms > lookahead_ms else lookahead_ms
            if min_hold_ms < 5.0:
                min_hold_ms = 5.0
            elif min_hold_ms > 200.0:
                min_hold_ms = 200.0
            hold_timer = min_hold_ms * 0.001 * fs
        elif hold_timer > 0.0:
            hold_timer -= 1.0
        else:
            envelope = envelope * decay + peak * (1.0 - decay)

        # --- Gain reduction (soft knee toward `sat`) ---
        env_db = 20.0 * math.log10(envelope if envelope > 1e-5 else 1e-5)
        gr_db = 0.0
        if knee <= 0.0 or env_db < sat - knee / 2.0:
            if env_db > sat:
                gr_db = sat - env_db
        elif env_db > sat + knee / 2.0:
            gr_db = sat - env_db
        else:
            x = env_db - (sat - knee / 2.0)
            gr_db = -(x * x) / (2.0 * knee)

        gain = (10.0 ** (gr_db / 20.0)) * out_gain_lin

        # --- Ring buffer: write current, read delayed (lookahead) ---
        ringL[write_index] = sL
        ringR[write_index] = sR
        read_index = write_index - delay_samples
        if read_index < 0:
            read_index += max_delay
        dL = ringL[read_index]
        dR = ringR[read_index]
        write_index += 1
        if write_index >= max_delay:
            write_index = 0

        outL[i] = dL * gain
        outR[i] = dR * gain

    return write_index, envelope, hold_timer, peak_rate


# JIT the kernel if Numba is available; otherwise use the pure-Python version.
try:                                    # pragma: no cover - env dependent
    from numba import njit
    _kernel_jit = njit(cache=True, fastmath=True)(_kernel)
    HAVE_NUMBA = True
except Exception:                       # pragma: no cover - env dependent
    _kernel_jit = _kernel
    HAVE_NUMBA = False


class Limiter:
    """Stereo lookahead limiter. Call :meth:`process` once per audio block.

    Adds ``lookahead_ms`` of latency (default 0.5 ms — negligible next to the
    FIR's ~21 ms group delay). State persists across blocks, so create one
    instance per stream and reuse it.
    """

    def __init__(self, fs, saturation_db=SATURATION_DB, knee_db=KNEE_DB,
                 output_gain_db=OUTPUT_GAIN_DB, lookahead_ms=LOOKAHEAD_MS,
                 min_recovery_ms=MIN_RECOVERY_MS, warmup=True):
        self.fs = float(fs)
        self.saturation_db = float(saturation_db)
        self.knee_db = float(knee_db)
        self.output_gain_db = float(output_gain_db)
        self.lookahead_ms = float(lookahead_ms)
        self.min_recovery_ms = float(min_recovery_ms)

        # Lookahead ring (max 20 ms, as in the worklet).
        self.max_delay = max(1, int(math.ceil(self.fs * 0.020)))
        self.ringL = np.zeros(self.max_delay, dtype=np.float64)
        self.ringR = np.zeros(self.max_delay, dtype=np.float64)
        self.write_index = 0

        # Envelope state.
        self.envelope = 0.0
        self.hold_timer = 0.0
        self.peak_rate = 0.0

        # Spectral hints (optional; from the analyser). Zero => fixed recovery.
        self.centroid = 0.0
        self.low_energy = 0.0
        self.rms = 0.0

        # Metering (read-only from the GUI side).
        self.in_peak = 0.0
        self.out_peak = 0.0

        # Preallocated scratch (no allocation in the hot path once sized).
        self._outL = np.zeros(0, dtype=np.float64)
        self._outR = np.zeros(0, dtype=np.float64)
        self._out = np.zeros((0, 2), dtype=np.float64)

        if warmup and HAVE_NUMBA:
            # Compile the kernel now (first call is slow) so the audio thread
            # never pays the JIT cost mid-stream.
            self.process(np.zeros((64, 2), dtype=np.float64))
            self.reset()

    # ---- parameter / hint setters ---------------------------------------
    def set_params(self, **kw):
        for k in ("saturation_db", "knee_db", "output_gain_db",
                  "lookahead_ms", "min_recovery_ms"):
            if k in kw and kw[k] is not None:
                setattr(self, k, float(kw[k]))

    def set_spectral(self, centroid=None, low_energy=None, rms=None):
        """Push spectral hints (from the analyser) that shape adaptive recovery
        and hold. All optional; left at 0 the limiter uses a fixed recovery."""
        if centroid is not None:
            self.centroid = float(centroid)
        if low_energy is not None:
            self.low_energy = float(low_energy)
        if rms is not None:
            self.rms = float(rms)

    def reset(self):
        self.ringL.fill(0.0)
        self.ringR.fill(0.0)
        self.write_index = 0
        self.envelope = 0.0
        self.hold_timer = 0.0
        self.peak_rate = 0.0

    # ---- the hot path ----------------------------------------------------
    def process(self, block):
        """Limit one stereo block.

        block : (frames, 2) array. Returns a (frames, 2) float64 array
        (reused scratch — copy it if you need to retain it past the next call).
        """
        x = np.asarray(block, dtype=np.float64)
        if x.ndim == 1:
            x = np.stack([x, x], axis=1)
        frames = x.shape[0]

        if self._outL.shape[0] != frames:
            self._outL = np.zeros(frames, dtype=np.float64)
            self._outR = np.zeros(frames, dtype=np.float64)
            self._out = np.zeros((frames, 2), dtype=np.float64)

        inL = np.ascontiguousarray(x[:, 0])
        inR = np.ascontiguousarray(x[:, 1])

        # Lookahead delay in samples (clamped to the ring).
        delay_samples = int(round(self.lookahead_ms * 0.001 * self.fs))
        delay_samples = min(max(delay_samples, 0), self.max_delay - 1)

        sat = self.saturation_db
        knee = self.knee_db
        out_gain_lin = 10.0 ** (self.output_gain_db / 20.0)

        # Adaptive recovery -> decay coefficient (per block, as in the worklet).
        freq_factor = max(0.2, 1.0 - self.centroid / 10000.0)
        energy_factor = 1.0 + self.low_energy * self.rms * 5.0
        recovery_ms = self.min_recovery_ms * energy_factor * freq_factor
        recovery_ms = min(max(recovery_ms, self.min_recovery_ms),
                          self.min_recovery_ms * 5.0)
        release_sec = recovery_ms / 1000.0
        decay = math.exp(-1.0 / (release_sec * self.fs))

        # Input metering.
        self.in_peak = float(np.max(np.abs(x))) if frames else 0.0

        (self.write_index, self.envelope, self.hold_timer,
         self.peak_rate) = _kernel_jit(
            inL, inR, self._outL, self._outR, self.ringL, self.ringR,
            self.write_index, self.envelope, self.hold_timer, self.peak_rate,
            delay_samples, self.max_delay,
            sat, knee, out_gain_lin, decay,
            self.centroid, self.low_energy, self.lookahead_ms, self.fs)

        self._out[:, 0] = self._outL
        self._out[:, 1] = self._outR
        self.out_peak = float(np.max(np.abs(self._out))) if frames else 0.0
        return self._out
