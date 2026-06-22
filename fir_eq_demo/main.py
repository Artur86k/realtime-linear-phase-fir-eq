"""Real-Time Linear-Phase FIR EQ — seamless coefficient-update demo.

Single-window matplotlib app:
  - Top axis: drag points to edit the target magnitude response.
  - Bottom axis: live FIR impulse response (stays symmetric -> linear phase).
  - Input-level slider (+/-20 dB) with a clip indicator.
  - Play / Pause / Stop / Load buttons.

Editing the curve redesigns the FIR (throttled) and hot-swaps it into the
audio thread with an equal-power crossfade -> no clicks.

Run:  python -m fir_eq_demo.main  [optional_audio_file]
  or: python main.py [optional_audio_file]
"""

from __future__ import annotations

import sys
import os

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button

# Support running as a module or as a plain script.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import filter_design as fd
    from audio_engine import AudioEngine, CoeffHolder
    from curve_editor import CurveEditor
    from limiter import Limiter
    from sound_improver import AnalysisEngine
else:
    from . import filter_design as fd
    from .audio_engine import AudioEngine, CoeffHolder
    from .curve_editor import CurveEditor
    from .limiter import Limiter
    from .sound_improver import AnalysisEngine

# ---- Constants -----------------------------------------------------------
FS_FALLBACK = 48000
N_TAPS = 2047
M_DESIGN = 8192
KAISER_BETA = 8.0
BLOCKSIZE = 1024
TAPS_VIEW = 400         # taps shown by default around the center
FMIN, FMAX = 20.0, 20000.0
GAIN_MIN_DB, GAIN_MAX_DB = -20.0, 20.0
REDESIGN_INTERVAL_MS = 16   # ~60 Hz throttle


class App:
    def __init__(self, audio_path=None):
        self.fs = FS_FALLBACK
        self.N = N_TAPS

        # Initial flat curve (0 dB).
        init_points = [(20.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)]
        b0 = fd.design_fir(init_points, self.fs, self.N, M_DESIGN, KAISER_BETA)

        self.holder = CoeffHolder(b0)
        self.engine = AudioEngine(self.holder, blocksize=BLOCKSIZE,
                                  fs_fallback=FS_FALLBACK)

        # Sound-improver (adaptive multiband) + limiter are built lazily on
        # first toggle, once open_stream() has finalised the device rate.
        self.improver = None
        self.limiter = None
        self._last_auto_version = self.holder.version

        # ---- Figure layout ----
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Linear-Phase FIR EQ Demo")
        gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.35,
                      left=0.10, right=0.96, top=0.93, bottom=0.30)
        self.ax_eq = self.fig.add_subplot(gs[0])
        self.ax_taps = self.fig.add_subplot(gs[1])

        self.editor = CurveEditor(self.ax_eq, init_points,
                                  on_change=self._mark_dirty,
                                  curve_fn=fd.interpolate_db)

        # Center-biased tap index: central tap = 0, neighbours ±1, ±2, ...
        # (group delay is (N-1)/2, so 0 here = the peak of the symmetric IR).
        self._tap_x = np.arange(self.N) - self.N // 2
        self.taps_line, = self.ax_taps.plot(self._tap_x, b0, color="#2ca02c", lw=0.8)
        self.ax_taps.set_title("FIR impulse response (taps) — symmetric => "
                               "linear phase")
        self.ax_taps.set_xlabel("Tap index (0 = center; zoom out for all "
                                f"{self.N} taps)")
        self.ax_taps.set_ylabel("Amplitude")
        # Default view: ~400 taps around the center; toolbar can zoom/pan out.
        self.ax_taps.set_xlim(-TAPS_VIEW // 2, TAPS_VIEW // 2)
        self.ax_taps.grid(True, alpha=0.3)

        self._build_widgets()

        self._dirty = False
        self._update_fir_overlay(b0)

        # ---- Blitting setup --------------------------------------------------
        # A full figure redraw is ~80 ms here; doing that 10x/s for the clock
        # alone pegged a core. Instead we cache the static background once and
        # only re-blit the handful of artists that actually change.
        self._bg = None
        self._animated = (self.editor.animated_artists + [self.taps_line]
                          + [self.prog_slider.poly, self.prog_slider._handle,
                             self.gain_slider.poly, self.gain_slider._handle,
                             self.clip_patch, self.time_text])
        for art in self._animated:
            art.set_animated(True)
        # Sliders must not trigger their own full redraw on set_val.
        self.prog_slider.drawon = False
        self.gain_slider.drawon = False
        # Recapture the background after every real (full) draw — e.g. resize,
        # or when we deliberately autoscale the taps axis.
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)

        # Throttle timer: coalesce drag events, redesign + blit at ~60 Hz.
        self.timer = self.fig.canvas.new_timer(interval=REDESIGN_INTERVAL_MS)
        self.timer.add_callback(self._tick)
        self.timer.start()

        if audio_path:
            self._do_load(audio_path)
        else:
            self._load_default_audio()

    # ---- widgets ---------------------------------------------------------
    def _build_widgets(self):
        # Progress / seek slider (0..1 fraction of the loaded audio).
        ax_prog = self.fig.add_axes([0.10, 0.19, 0.55, 0.03])
        self.prog_slider = Slider(ax_prog, "Position", 0.0, 1.0, valinit=0.0)
        self.prog_slider.valtext.set_visible(False)   # we show mm:ss instead
        self._syncing_prog = False                    # guard against feedback
        self.prog_slider.on_changed(self._on_seek)
        self.time_text = self.fig.text(0.67, 0.205, "0:00 / 0:00",
                                       fontsize=9, va="center")

        ax_gain = self.fig.add_axes([0.10, 0.12, 0.55, 0.03])
        self.gain_slider = Slider(ax_gain, "Input (dB)", GAIN_MIN_DB,
                                  GAIN_MAX_DB, valinit=0.0)
        self.gain_slider.on_changed(self._on_gain)

        # Clip indicator.
        self.ax_clip = self.fig.add_axes([0.70, 0.115, 0.06, 0.04])
        self.ax_clip.axis("off")
        self.clip_patch = self.ax_clip.text(0.5, 0.5, "OK", ha="center",
                                            va="center", fontsize=10,
                                            bbox=dict(boxstyle="round",
                                                      fc="#7CFC00", ec="k"))

        def mk_button(x, label, cb):
            a = self.fig.add_axes([x, 0.03, 0.10, 0.05])
            b = Button(a, label)
            b.on_clicked(cb)
            return b

        self.btn_play = mk_button(0.10, "Play", lambda e: self.engine.play())
        self.btn_pause = mk_button(0.22, "Pause", lambda e: self.engine.pause())
        self.btn_stop = mk_button(0.34, "Stop", lambda e: self.engine.stop())
        self.btn_load = mk_button(0.46, "Load", lambda e: self._on_load_click())
        self.btn_rec = mk_button(0.58, "Record", self._on_record)
        # Adaptive "sound improver" + final limiter toggles.
        self.btn_improve = mk_button(0.70, "Improve", self._on_improve)
        self.btn_limit = mk_button(0.82, "Limiter", self._on_limiter)

        # Status timer: clip indicator + progress/time readout.
        self._last_clip = False
        self._last_time = ""
        self.status_timer = self.fig.canvas.new_timer(interval=150)
        self.status_timer.add_callback(self._update_status)
        self.status_timer.start()

    # ---- blitting --------------------------------------------------------
    def _on_draw(self, event):
        # Called after every full draw: cache the static background, then
        # paint the animated artists on top so they become visible again
        # (animated artists are skipped by the normal draw).
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_animated()
        self.fig.canvas.blit(self.fig.bbox)

    def _draw_animated(self):
        for art in self._animated:
            ax = art.axes
            if ax is not None:
                ax.draw_artist(art)
            else:                       # figure-level text
                self.fig.draw_artist(art)

    def _blit(self):
        """Full-figure blit. Cheap-but-not-free (~6 ms pixel push); used for
        the infrequent widget/clock updates."""
        if self._bg is None:            # not drawn yet; force a full draw
            self.fig.canvas.draw_idle()
            return
        self.fig.canvas.restore_region(self._bg)
        self._draw_animated()
        self.fig.canvas.blit(self.fig.bbox)

    def _blit_region(self, artists, ax):
        """Repaint, pushing ONLY one axes rectangle to screen. The full
        background restore is cheap (~0.1 ms) and reliably clears the previous
        frame (a sub-bbox restore leaves a trail); the costly part — the pixel
        blit (~6 ms full vs ~2 ms region) — is what we limit to `ax`."""
        if self._bg is None:
            self.fig.canvas.draw_idle()
            return
        self.fig.canvas.restore_region(self._bg)   # full restore: clears old curve
        for a in artists:
            a.axes.draw_artist(a)
        self.fig.canvas.blit(ax.bbox)              # push only this region

    # ---- redesign throttle ----------------------------------------------
    def _mark_dirty(self):
        self._dirty = True

    def _tick(self):
        # When the sound-improver drives the curve, just mirror its output
        # (taps + measured overlay) into the plots; ignore manual edits.
        if self.improver is not None and self.improver.enabled:
            self._auto_refresh()
            return
        if not self._dirty:
            return
        self._dirty = False
        b = fd.design_fir(self.editor.points, self.fs, self.N,
                          M_DESIGN, KAISER_BETA)
        self.holder.update(b)                # atomic handoff to audio thread
        # Update taps plot + measured-FIR overlay (GUI thread only).
        self.taps_line.set_ydata(b)
        self._update_fir_overlay(b)

        # Taps autoscale: only do a (slow) full redraw when the impulse grows
        # past the current y-limit; otherwise region-blit. Keeps drag at 60fps.
        peak = float(np.max(np.abs(b))) or 1e-6
        lo, hi = self.ax_taps.get_ylim()
        if peak > hi or peak < hi * 0.4:
            self.ax_taps.set_ylim(-peak * 1.15, peak * 1.15)
            self.fig.canvas.draw_idle()      # recaptures bg via _on_draw
        else:
            # Region-limited blits: only the two plot rectangles change while
            # dragging, so we never pay for the full-window pixel push.
            self._blit_region(self.editor.animated_artists, self.ax_eq)
            self._blit_region([self.taps_line], self.ax_taps)

    def _auto_refresh(self):
        """GUI-thread mirror of the improver's latest curve. The improver
        thread only writes the holder; all matplotlib calls stay here."""
        v = self.holder.version
        if v == self._last_auto_version:
            return
        self._last_auto_version = v
        b = self.holder.b_target
        self.taps_line.set_ydata(b)
        self._update_fir_overlay(b)
        peak = float(np.max(np.abs(b))) or 1e-6
        lo, hi = self.ax_taps.get_ylim()
        if peak > hi or peak < hi * 0.4:
            self.ax_taps.set_ylim(-peak * 1.15, peak * 1.15)
            self.fig.canvas.draw_idle()
        else:
            self._blit_region(self.editor.animated_artists, self.ax_eq)
            self._blit_region([self.taps_line], self.ax_taps)

    def _update_fir_overlay(self, b):
        freqs = self.editor.freq_grid
        mag_db = fd.fir_magnitude_db(b, self.fs, freqs)
        self.editor.set_fir_overlay(freqs, mag_db)

    # ---- sound-improver / limiter toggles --------------------------------
    def _on_improve(self, event):
        # Build lazily once open_stream() has finalised the device rate.
        self.engine.open_stream()
        if self.improver is None:
            self.improver = AnalysisEngine(self.engine, self.holder, self.N,
                                           M_DESIGN, KAISER_BETA)
            self.improver.start()
        on = not self.improver.enabled
        self.improver.set_enabled(on)
        self.editor.enabled = not on      # suppress manual edits while auto
        self.btn_improve.label.set_text("Improve*" if on else "Improve")
        self.btn_improve.ax.set_facecolor("#90EE90" if on else "0.85")
        self.fig.canvas.draw_idle()

    def _on_limiter(self, event):
        self.engine.open_stream()
        if self.limiter is None:
            # First build compiles the JIT kernel (warmup) — done here on the
            # GUI thread so the audio callback never pays that cost.
            self.limiter = Limiter(self.engine.fs)
            self.engine.limiter = self.limiter
        on = not self.engine.limiter_enabled
        self.engine.limiter_enabled = on
        self.btn_limit.label.set_text("Limiter*" if on else "Limiter")
        self.btn_limit.ax.set_facecolor("#90EE90" if on else "0.85")
        self.fig.canvas.draw_idle()

    # ---- widget callbacks ------------------------------------------------
    def _on_gain(self, val):
        self.engine.input_gain = 10.0 ** (val / 20.0)
        self._blit()                    # slider drawon=False -> blit ourselves

    def _on_seek(self, val):
        # Ignore programmatic updates from the status timer (no feedback loop).
        if self._syncing_prog:
            return
        self.engine.seek_frac(val)
        self._blit()

    @staticmethod
    def _fmt_time(s):
        s = int(s)
        return f"{s // 60}:{s % 60:02d}"

    def _update_status(self):
        clipped = self.engine.clipped
        time_str = (f"{self._fmt_time(self.engine.position_s())} / "
                    f"{self._fmt_time(self.engine.duration_s())}")

        # Nothing visibly changed (e.g. paused, no clip) -> don't burn a blit.
        if clipped == self._last_clip and time_str == self._last_time:
            return
        self._last_clip, self._last_time = clipped, time_str

        if clipped:
            self.clip_patch.set_text("CLIP")
            self.clip_patch.get_bbox_patch().set_facecolor("#FF3030")
        else:
            self.clip_patch.set_text("OK")
            self.clip_patch.get_bbox_patch().set_facecolor("#7CFC00")

        # Progress slider + time readout. Skip while the user is dragging it.
        if not getattr(self.prog_slider, "drag_active", False):
            frac = self.engine.position_frac()
            self._syncing_prog = True
            self.prog_slider.set_val(frac)          # programmatic, no seek
            self._syncing_prog = False
        self.time_text.set_text(time_str)
        self._blit()

    def _on_load_click(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Open audio file",
                filetypes=[("Audio", "*.wav *.flac *.ogg *.aiff *.mp3"),
                           ("All", "*.*")])
            root.destroy()
            if path:
                self._do_load(path)
        except Exception as e:
            print(f"Load dialog failed: {e}")

    def _on_record(self, event):
        # Toggle: arm recording, then on the second press write what was
        # captured (exactly the samples sent to the device) to a WAV.
        if not self.engine.recording:
            self.engine.start_record()
            self.btn_rec.label.set_text("Stop Rec")
            self.btn_rec.ax.set_facecolor("#FF6060")
            print("Recording output... press 'Stop Rec' to save.")
        else:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                path = filedialog.asksaveasfilename(
                    title="Save recorded output",
                    defaultextension=".wav",
                    initialfile="eq_output.wav",
                    filetypes=[("WAV", "*.wav")])
                root.destroy()
            except Exception as e:
                path = None
                print(f"Save dialog failed: {e}")
            n = self.engine.stop_record(path)
            self.btn_rec.label.set_text("Record")
            self.btn_rec.ax.set_facecolor("0.85")
            if path and n:
                print(f"Saved {n} frames ({n / self.engine.fs:.2f}s @ "
                      f"{self.engine.fs} Hz, 32-bit float) -> {path}")
            elif not path:
                print("Recording discarded (no file chosen).")
            else:
                print("Nothing was recorded (was it playing?).")
        self.fig.canvas.draw_idle()        # repaint button label/colour

    def _do_load(self, path):
        try:
            self.engine.load_file(path)
            print(f"Loaded: {path}  ({self.engine.n_frames} frames @ {self.engine.fs} Hz)")
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    def _load_default_audio(self):
        """Generate a stereo test tone so the demo runs without a file."""
        fs = self.fs
        dur = 4.0
        t = np.arange(int(fs * dur)) / fs
        # Pink-ish noise + a couple of tones so EQ changes are obvious.
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(len(t))
        # 1st-order pinkening.
        noise = np.cumsum(noise)
        noise = noise / np.max(np.abs(noise))
        tones = 0.2 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 3000 * t))
        left = 0.4 * noise + tones
        right = 0.4 * np.roll(noise, 5) + tones
        sig = np.stack([left, right], axis=1)
        sig = sig / np.max(np.abs(sig)) * 0.5
        self.engine.audio = np.ascontiguousarray(sig, dtype=np.float32)
        self.engine.n_frames = sig.shape[0]
        self.engine.read_pos = 0
        print("No file given — generated a 4 s stereo test signal "
              "(pink noise + 220/3000 Hz tones).")

    def run(self):
        try:
            plt.show()
        finally:
            if self.improver is not None:
                self.improver.stop()
            self.engine.close()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    App(path).run()


if __name__ == "__main__":
    main()
