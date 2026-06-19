"""Matplotlib point-dragging logic for the frequency-response editor axis.

Handles: add point (left-click empty), move point (left-drag marker),
delete point (right-click marker, keep >= 2). Points are constrained to
20 Hz..20 kHz and +/-20 dB.

The editor owns the point set and fires an ``on_change`` callback (the GUI
wires this to a throttled FIR redesign).
"""

from __future__ import annotations

import numpy as np

FMIN, FMAX = 20.0, 20000.0
GAIN_MIN_DB, GAIN_MAX_DB = -20.0, 20.0
PICK_RADIUS_PX = 12.0


class CurveEditor:
    def __init__(self, ax, points, on_change, curve_fn):
        """
        ax        : the matplotlib Axes (semilogx) to draw on.
        points    : initial list of (freq_hz, gain_db).
        on_change : callback() invoked whenever the point set changes.
        curve_fn  : function(points, freqs_hz) -> gain_db array, for the curve.
        """
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.points = [list(p) for p in points]
        self.on_change = on_change
        self.curve_fn = curve_fn

        self._drag_idx = None

        self.curve_line, = ax.semilogx([], [], "-", color="#1f77b4", lw=2,
                                       label="target")
        self.fir_line, = ax.semilogx([], [], "--", color="#ff7f0e", lw=1,
                                     alpha=0.7, label="measured FIR")
        self.markers, = ax.semilogx([], [], "o", color="#d62728", ms=8,
                                    picker=False, zorder=5)

        ax.set_xlim(FMIN, FMAX)
        ax.set_ylim(GAIN_MIN_DB, GAIN_MAX_DB)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_title("Drag points to edit EQ  |  L-click=add  "
                     "R-click / double-click=delete")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        self._freq_grid = np.logspace(np.log10(FMIN), np.log10(FMAX), 512)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self.redraw_curve()

    # ---- geometry helpers ------------------------------------------------
    def _sorted_points(self):
        return sorted(self.points, key=lambda p: p[0])

    def _nearest_point(self, event):
        """Index of marker within PICK_RADIUS_PX of the cursor, else None."""
        if event.xdata is None:
            return None
        best, best_d = None, PICK_RADIUS_PX
        for i, (f, g) in enumerate(self.points):
            px, py = self.ax.transData.transform((f, g))
            d = np.hypot(px - event.x, py - event.y)
            if d < best_d:
                best, best_d = i, d
        return best

    @staticmethod
    def _clamp(f, g):
        f = float(np.clip(f, FMIN, FMAX))
        g = float(np.clip(g, GAIN_MIN_DB, GAIN_MAX_DB))
        return f, g

    # ---- event handlers --------------------------------------------------
    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        idx = self._nearest_point(event)

        # Right-click, or double-click on a marker -> delete (keep >= 2).
        # When only 2 points remain, deleting is not allowed; instead reset
        # both to the flat default endpoints (20 Hz & 20 kHz @ 0 dB).
        if (event.button == 3) or (event.dblclick and idx is not None):
            if idx is not None:
                if len(self.points) > 2:
                    del self.points[idx]
                    self._changed()
                else:
                    self.points = [[FMIN, 0.0], [FMAX, 0.0]]
                    self._changed()
            return

        if event.button == 1:                  # left-click
            if idx is not None:
                self._drag_idx = idx           # start dragging existing
            else:
                f, g = self._clamp(event.xdata, event.ydata)
                self.points.append([f, g])
                self._drag_idx = len(self.points) - 1
                self._changed()

    def _on_motion(self, event):
        if self._drag_idx is None or event.inaxes != self.ax or event.xdata is None:
            return
        f, g = self._clamp(event.xdata, event.ydata)
        self.points[self._drag_idx] = [f, g]
        self._changed()

    def _on_release(self, event):
        self._drag_idx = None

    def _changed(self):
        self.redraw_curve()
        if self.on_change:
            self.on_change()

    # ---- drawing ---------------------------------------------------------
    # NOTE: these only set artist data. The owning app blits the figure on a
    # throttled timer, so we never trigger a full (slow) canvas redraw here.
    def redraw_curve(self):
        pts = self._sorted_points()
        fx = [p[0] for p in pts]
        gy = [p[1] for p in pts]
        self.markers.set_data(fx, gy)
        db = self.curve_fn(self.points, self._freq_grid)
        self.curve_line.set_data(self._freq_grid, db)

    def set_fir_overlay(self, freqs, mag_db):
        self.fir_line.set_data(freqs, mag_db)

    @property
    def animated_artists(self):
        return [self.curve_line, self.fir_line, self.markers]

    @property
    def freq_grid(self):
        return self._freq_grid
