# Real-Time Linear-Phase FIR EQ — Seamless Coefficient Update Demo

## Goal

Build a single-window Python application that demonstrates **seamless, click-free swapping of FIR coefficients on a live audio stream** while the user edits the target magnitude response by dragging points on a curve. The whole point of the demo is to *prove audibly and visually* that the filter can be redesigned continuously without artifacts.

The app plays a stereo audio file in a loop and applies a **linear-phase FIR** whose magnitude response is defined by a smooth curve through user-placed points. As the user drags points, the FIR is redesigned and hot-swapped into the audio thread with a crossfade so there are no clicks.

---

## Functional requirements

1. **Stereo file playback.** Load a stereo audio file (WAV / FLAC / OGG via `soundfile`). Loop it. Provide Play / Pause / Stop. Resample to the output device rate if needed.
2. **Plot 1 — frequency-response editor.**
   - Log frequency x-axis, **20 Hz → 20 kHz**.
   - Linear gain y-axis, **−20 dB → +20 dB**.
   - User can **add a point** (e.g. left-click on empty area), **move a point** (drag, constrained to ±20 dB and 20 Hz–20 kHz), and **delete a point** (e.g. right-click).
   - A smooth curve passes **through** all points (see "Curve" below) and *is* the target magnitude response.
3. **Plot 2 — FIR coefficient viewer.**
   - Shows the current FIR impulse response (the `N` tap values) as a line or stem plot, x = tap index, y = amplitude.
   - Updates live whenever the filter is redesigned, so the user sees the impulse response morph (and stays symmetric → visibly linear-phase).
4. **Input-level slider.** Range **−20 dB … +20 dB**, applied as a gain to the signal to avoid clipping/overload. Show a simple clip indicator (turns red if any output sample |x| ≥ 1.0 in the last block).
5. **Live, seamless update.** Editing the curve must change what you hear within ~one audio block, with **no clicks/pops**. This is the headline feature.

---

## Architecture (threading + data flow)

Three concerns, kept separate:

```
GUI thread (matplotlib)                     Audio thread (sounddevice callback)
─────────────────────                       ──────────────────────────────────
point added/moved  ──► redesign FIR ──►  shared handoff  ──►  pick up new taps,
(throttled 30–60Hz)    (oversampled            (atomic           crossfade old→new
                        IFFT + window)          ref swap)         over one block
```

- **GUI / main thread:** matplotlib figure with the two axes and the slider. Mouse event handlers edit the point set. On any change, redesign is **throttled** (only run on actual change, coalesce rapid drags — a `~16 ms` timer or "dirty" flag is enough; do **not** redesign per mouse-move pixel).
- **Filter design:** pure function `points -> b` (numpy array of `N` taps). Cheap enough (<1 ms for `N≈2048`, `M=8192`) to run inline on the GUI thread.
- **Handoff:** lock-free. Keep a holder object with attributes `b_target` and a monotonically increasing `version` int. GUI assigns `holder.b_target = new_b; holder.version += 1`. Under CPython the attribute assignment is atomic (GIL), so no lock is needed; the audio callback just reads and compares `version`. (A `queue.Queue(maxsize=1)` is an acceptable alternative.)
- **Audio callback:** detects a new version, captures `b_new`, and crossfades from `b_current` to `b_new` over the current block, then sets `b_current = b_new`.

**Never block the audio callback** — no file I/O, no plotting, no allocation in the hot path (preallocate buffers). Redesign and plotting happen only on the GUI thread.

---

## DSP: linear-phase FIR design from the curve

Type I FIR (odd `N`, symmetric taps → exact linear phase, group delay `(N−1)/2`).

**Default `N = 2047`** taps (make it a constant; expose a few presets e.g. 1023 / 2047 / 4095). Note the tradeoff in the UI/README: more taps → finer low-frequency resolution but more latency and more pre-ring.

Design steps (`design_fir(points, fs, N, M)`):

1. **Build the dB curve.** Sort points by frequency. Clamp/extend so the curve is defined across the whole band: hold the first point's gain below its frequency and the last point's gain above its frequency (flat extension to DC and Nyquist).
2. **Interpolate** the gain in **log-frequency / linear-dB** space (this is the perceptually natural EQ space). Use the **Curve** method below.
3. **Sample onto the FFT grid.** `M` is the oversampled design length (default `M = 8192`, must be ≥ ~4·`N`). Evaluate the interpolated curve at the real-FFT bin frequencies `f_k = k·fs/M`, `k = 0 … M/2`. Convert dB → linear magnitude. This gives a real, **zero-phase** spectrum `H` of length `M/2+1`.
4. **IFFT.** `h_full = np.fft.irfft(H, n=M)` → real impulse response, symmetric about index 0 (wrapped).
5. **Center + truncate + window.** `np.fft.fftshift(h_full)`, take the center `N` samples, multiply by a **Kaiser window (β ≈ 8.0)** (or Blackman–Harris). This is the oversampled-then-windowed frequency-sampling method — it decouples curve resolution (`M`) from tap count (`N`) and suppresses Gibbs ripple between control points.
6. Return `b` (float64, length `N`). It is symmetric ⇒ linear phase.

> Rationale already validated: the impulse response is coefficient-only; the audio thread's delay line holds **past input samples**, which are valid for any coefficients, so swapping never corrupts state. The only artifact is the output step between two weighted sums — handled by the crossfade.

### Curve method (passes *through* the points)

The user wants the curve to "stick to" the points. A single Bézier only passes through its two endpoints, so use a **Catmull–Rom spline**, which interpolates **every** control point and is expressible exactly as a chain of cubic Bézier segments (so it's "Bézier" in the piecewise sense). Implement Catmull–Rom in (log10 f, dB) space with endpoint tangents from one-sided differences. `scipy.interpolate.CubicSpline` with `bc_type='natural'`, or `PchipInterpolator` (monotone, no overshoot) are acceptable substitutes — PCHIP is the safest for avoiding overshoot between widely spaced points. Pick **PCHIP** as the default and leave a comment that Catmull–Rom is the "true Bézier-chain" alternative.

---

## Real-time filtering + seamless swap

Use the **coefficient-independent delay-line** convolution so that swapping taps is provably safe:

Per channel, maintain an input history of `N−1` samples. For each callback block of `L` frames:
1. `x_ext = concat(history, in_block)` (length `N−1+L`).
2. `y = fftconvolve(x_ext, b_current, mode='valid')` → length `L`. Use `scipy.signal.oaconvolve` / `fftconvolve` (FFT-based) so `N≈2048` stereo at 48 kHz runs comfortably; direct `np.convolve` is fine only for small `N`.
3. **If a new version arrived:** also compute `y_new` with `b_new`, then blend with an **equal-power crossfade** over the block: `y = w_old·y_old + w_new·y_new`, `w_old = cos(θ)`, `w_new = sin(θ)`, `θ` sweeping `0→π/2` across the `L` samples. After the block, `b_current = b_new`. (Both filters share the same group delay `(N−1)/2`, so the crossfade is phase-coherent — no comb filtering during the transition.)
4. `history = x_ext[−(N−1):]`.

Apply the **input-gain slider** value (linear, from dB) to `in_block` *before* filtering. Compute peak of the output block for the clip indicator.

Latency note for the README: group delay = `(N−1)/2` samples (e.g. `N=2047` @ 48 kHz ≈ 21 ms) plus block I/O latency. This is expected for linear phase.

---

## GUI layout

- One matplotlib figure, `GridSpec` with two stacked axes:
  - **Top:** frequency-response editor (log-x, semilogx). Draw the points as draggable markers and the interpolated curve as a line. Overlay (optional, light) the **actual measured FIR magnitude** (`np.abs(rfft(b))` mapped to the same axes) so the user can see how well the FIR tracks the target — great for showing the windowing tradeoff.
  - **Bottom:** FIR taps line/stem plot.
- A `matplotlib.widgets.Slider` for input level (−20…+20 dB) and `Button`s for Play / Pause / Stop / Load file.
- Use **blitting** or throttled `draw_idle()` for smooth redraws; redraw the taps plot only on redesign, not every frame.

Mouse interaction (top axis):
- Left-click empty area → add point at cursor (snap to band limits).
- Left-drag on a marker → move (clamp to ±20 dB, 20 Hz–20 kHz).
- Right-click on a marker → delete (keep a minimum of 2 points).

---

## Dependencies

```
numpy
scipy
sounddevice      # PortAudio callback streaming
soundfile        # file loading (libsndfile)
matplotlib       # GUI + plots + widgets
```

`pip install numpy scipy sounddevice soundfile matplotlib`
(`sounddevice`/`soundfile` need PortAudio/libsndfile present on the system.)

---

## Suggested file structure

```
fir_eq_demo/
├── main.py            # app entry: builds GUI, wires callbacks, starts stream
├── filter_design.py   # design_fir(points, fs, N, M) -> b ; curve interpolation
├── audio_engine.py    # sounddevice stream, callback, delay-line convolution, crossfade, gain
├── curve_editor.py    # matplotlib point-dragging logic for the top axis
└── README.md          # latency / pre-ring / tap-count tradeoffs explained
```

Keep `filter_design.py` and `audio_engine.py` import-clean and unit-testable (no GUI deps).

---

## Constants (defaults)

```python
FS_FALLBACK = 48000     # if file rate unusable
N_TAPS      = 2047      # odd → Type I linear phase
M_DESIGN    = 8192      # oversampled IFFT length (>= ~4*N)
KAISER_BETA = 8.0
BLOCKSIZE   = 1024      # sounddevice block; crossfade spans one block
FMIN, FMAX  = 20.0, 20000.0
GAIN_MIN_DB, GAIN_MAX_DB = -20.0, 20.0
```

---

## Acceptance criteria

1. Loads and loops a stereo file; Play/Pause/Stop work.
2. Dragging a point changes the audible EQ within ~one block, **with no clicks** (verify with a steep cut/boost — the crossfade must hide the transition).
3. Top plot: smooth curve passes exactly through every point; points constrained to the band and ±20 dB.
4. Bottom plot: tap values update live and remain **symmetric** (linear phase) for every curve.
5. Input slider scales level over ±20 dB; clip indicator fires on overload.
6. Audio callback never blocks (no glitches/underruns during heavy dragging); redesign + plotting are off the audio thread.
7. (Nice-to-have) Overlaid measured FIR magnitude tracks the target curve; mismatch visibly grows for steep segments — demonstrates the resolution/tap-count tradeoff.

---

## Implementation notes / gotchas

- **Throttle redesign.** Coalesce drag events; redesign at most ~60 Hz. Per-pixel redesign will choke the GUI thread, not the audio.
- **Preallocate** all audio-thread buffers; do no allocation/IO inside the callback.
- **Atomic handoff**: assign whole numpy arrays by reference + bump an int `version`; don't mutate a shared array in place from the GUI thread.
- **Group-delay match**: only crossfade between filters of the **same `N` and symmetry** so group delay is identical — otherwise the blend combs. If you add a tap-count selector, force a full crossfade (or brief silence) on `N` change rather than blending mismatched delays.
- **Pre-ring caveat (document, don't fix):** linear-phase symmetric taps pre-ring on steep edits (audible pre-echo on transients). It's expected here. Optionally add a toggle comment for a future **minimum-phase** variant (cepstral conversion of the same magnitude) that trades the linear-phase guarantee for no pre-ring.
- **Mono files**: duplicate to stereo. **Sample-rate**: resample file to device rate once at load (`scipy.signal.resample_poly`), don't resample per block.
