"""Real-time audio engine: looping playback + linear-phase FIR with a
seamless (click-free) coefficient hot-swap.

No GUI deps. The hot path (the sounddevice callback) does NO file I/O, NO
plotting, and NO allocation beyond a couple of small preallocated scratch
buffers.

Handoff is lock-free: the GUI thread sets ``holder.b_target`` (a whole new
array, by reference) and bumps ``holder.version``. The callback compares the
version and, on change, crossfades old->new over one block with a LINEAR
ramp -- equivalent to per-sample linear interpolation of the coefficients.
Both filters share group delay (N-1)/2, so y_old and y_new are in phase and
the linear blend moves the level monotonically with no +3 dB overshoot (an
equal-power cos/sin blend would bump, since it assumes uncorrelated signals).
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd
from scipy.signal import fftconvolve, resample_poly


class CoeffHolder:
    """Lock-free handoff of FIR taps from GUI thread to audio thread."""

    def __init__(self, b):
        self.b_target = b
        self.version = 0

    def update(self, b):
        # Assign whole array by reference, then bump version. Under CPython
        # both stores are atomic (GIL); the callback reads version last.
        self.b_target = b
        self.version += 1


class AudioEngine:
    def __init__(self, holder: CoeffHolder, blocksize=1024, fs_fallback=48000):
        self.holder = holder
        self.blocksize = blocksize
        self.fs_fallback = fs_fallback

        self.fs = fs_fallback
        self.audio = None          # (frames, 2) float32, resampled to device rate
        self.n_frames = 0
        self.read_pos = 0

        self.stream = None
        self.playing = False
        self._lock = threading.Lock()

        # Filter state -----------------------------------------------------
        b = holder.b_target
        self.b_current = b
        self.N = len(b)
        self.local_version = holder.version
        # Per-channel input history of N-1 samples (the coeff-independent
        # delay line). Valid for ANY coefficients -> swapping never corrupts.
        self.history = np.zeros((self.N - 1, 2), dtype=np.float64)

        # Status for the GUI (read-only from GUI side).
        self.last_peak = 0.0
        self.clipped = False
        self.input_gain = 1.0      # linear; set from the dB slider

        # Output recording (optional). The callback appends one block ref per
        # callback when armed — cheap enough; written to disk on stop.
        self.recording = False
        self._rec_blocks = []

    # ---- file loading ----------------------------------------------------
    def load_file(self, path):
        """Load + (re)sample a file to stereo at the current device rate."""
        import soundfile as sf  # local import keeps module import-clean

        data, file_fs = sf.read(path, dtype="float64", always_2d=True)
        if data.shape[1] == 1:                 # mono -> duplicate to stereo
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:                 # take first two channels
            data = data[:, :2]

        target_fs = self.fs
        if file_fs != target_fs:
            # resample_poly works column-wise; do it once at load time.
            from math import gcd
            g = gcd(int(file_fs), int(target_fs))
            up, down = int(target_fs // g), int(file_fs // g)
            data = resample_poly(data, up, down, axis=0)

        with self._lock:
            self.audio = np.ascontiguousarray(data, dtype=np.float32)
            self.n_frames = self.audio.shape[0]
            self.read_pos = 0

    # ---- transport -------------------------------------------------------
    def open_stream(self):
        if self.stream is not None:
            return
        self.fs = int(sd.query_devices(sd.default.device[1], "output")["default_samplerate"]) \
            if sd.default.device[1] is not None else self.fs_fallback
        self.stream = sd.OutputStream(
            samplerate=self.fs,
            blocksize=self.blocksize,
            channels=2,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def play(self):
        if self.stream is None:
            self.open_stream()
        self.playing = True

    def pause(self):
        self.playing = False

    def stop(self):
        self.playing = False
        with self._lock:
            self.read_pos = 0

    # ---- seek / position (for the progress slider) -----------------------
    def duration_s(self):
        with self._lock:
            n, fs = self.n_frames, self.fs
        return (n / fs) if (n and fs) else 0.0

    def position_s(self):
        with self._lock:
            n, fs, pos = self.n_frames, self.fs, self.read_pos
        return (pos / fs) if (n and fs) else 0.0

    def position_frac(self):
        with self._lock:
            n, pos = self.n_frames, self.read_pos
        return (pos / n) if n else 0.0

    def seek_frac(self, frac):
        """Jump to a fraction (0..1) of the loaded audio."""
        frac = float(min(max(frac, 0.0), 1.0))
        with self._lock:
            if self.n_frames:
                self.read_pos = int(frac * self.n_frames) % self.n_frames

    # ---- output recording ------------------------------------------------
    def start_record(self):
        self._rec_blocks = []
        self.recording = True

    def stop_record(self, path):
        """Stop recording and write captured output to `path` (32-bit float
        WAV). Returns the number of frames written (0 if nothing captured)."""
        self.recording = False
        blocks, self._rec_blocks = self._rec_blocks, []
        if not blocks or not path:
            return 0
        import soundfile as sf
        data = np.concatenate(blocks, axis=0)
        sf.write(path, data, int(self.fs), subtype="FLOAT")
        return data.shape[0]

    def close(self):
        self.playing = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    # ---- the hot path ----------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):
        # NEVER block here. No file IO, no plotting, minimal allocation.
        if (not self.playing) or self.audio is None or self.n_frames == 0:
            outdata.fill(0.0)
            return

        # Pull the next `frames` samples from the loop (handles wrap).
        in_block = self._read_loop(frames)            # (frames, 2) float64
        in_block *= self.input_gain

        N = self.N
        # x_ext = [history ; in_block]  -> length N-1+frames, per channel.
        x_ext = np.empty((N - 1 + frames, 2), dtype=np.float64)
        x_ext[: N - 1] = self.history
        x_ext[N - 1:] = in_block

        # Check for a redesigned filter (lock-free read).
        new_version = self.holder.version
        swap = new_version != self.local_version
        b_new = self.holder.b_target if swap else None

        y = self._convolve_valid(x_ext, self.b_current, frames)

        if swap and b_new is not None and len(b_new) == N:
            y_new = self._convolve_valid(x_ext, b_new, frames)
            # LINEAR crossfade old->new across the block. This is exactly a
            # per-sample linear interpolation of the coefficients
            # (b = (1-a)*b_old + a*b_new), so the output level moves
            # *monotonically* from the old level to the new one. Both filters
            # are linear-phase with the same group delay, so y_old and y_new
            # are in phase -> the linear blend preserves amplitude with no
            # bump. (An equal-power cos/sin blend would overshoot by up to
            # +3 dB here, because it assumes uncorrelated signals.)
            a = ((np.arange(frames) + 0.5) / frames)[:, None]   # 0 -> 1
            y = (1.0 - a) * y + a * y_new
            self.b_current = b_new
            self.local_version = new_version
        elif swap and b_new is not None:
            # Tap-count change: group delay differs -> don't blend (would
            # comb). Just adopt the new filter on the next block.
            self.b_current = b_new
            self.N = len(b_new)
            self.local_version = new_version
            self.history = np.zeros((self.N - 1, 2), dtype=np.float64)

        # Update the delay line with the most recent N-1 input samples.
        self.history = x_ext[-(N - 1):].copy()

        # Clip detect + write out.
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        self.last_peak = peak
        self.clipped = peak >= 1.0
        np.clip(y, -1.0, 1.0, out=y)
        outdata[:] = y.astype(np.float32)

        if self.recording:                 # capture exactly what's played
            self._rec_blocks.append(outdata.copy())

    def _read_loop(self, frames):
        """Return `frames` stereo samples from the looping source."""
        with self._lock:
            pos = self.read_pos
            n = self.n_frames
            audio = self.audio
        out = np.empty((frames, 2), dtype=np.float64)
        filled = 0
        while filled < frames:
            take = min(frames - filled, n - pos)
            out[filled: filled + take] = audio[pos: pos + take]
            filled += take
            pos += take
            if pos >= n:
                pos = 0
        with self._lock:
            self.read_pos = pos
        return out

    @staticmethod
    def _convolve_valid(x_ext, b, frames):
        """FFT convolution, 'valid' part -> exactly `frames` output samples."""
        out = np.empty((frames, 2), dtype=np.float64)
        for ch in range(2):
            out[:, ch] = fftconvolve(x_ext[:, ch], b, mode="valid")
        return out
