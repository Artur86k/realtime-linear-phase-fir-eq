# Real-Time Linear-Phase FIR EQ — Seamless Coefficient-Update Demo

Proves, audibly and visually, that a linear-phase FIR can be **redesigned
continuously while audio is playing** — no clicks, no pops — by hot-swapping
the tap coefficients into the audio thread with a linear crossfade.

## Run

```bash
pip install numpy scipy sounddevice soundfile matplotlib
python main.py                 # uses a generated stereo test signal
python main.py path/to/file.wav   # loop your own stereo file
```

(`sounddevice`/`soundfile` need PortAudio/libsndfile present on the system.)

A single window opens:

- **Top plot** — the EQ editor (log frequency 20 Hz–20 kHz, ±20 dB).
  - Left-click empty area → add a point.
  - Left-drag a marker → move it (clamped to band & ±20 dB).
  - Right-click **or** double-click a marker → delete. When only 2 points
    remain, deleting instead **resets** both to the flat default (20 Hz &
    20 kHz @ 0 dB).
  - Blue line = the target curve (passes *through* every point, PCHIP in
    log-f/dB space). Orange dashed = the **measured** FIR magnitude, so you
    can watch how well the FIR tracks the target.
- **Bottom plot** — the live FIR impulse response (the taps), indexed relative
  to the **center tap (0)** with neighbours ±1, ±2, … It stays **symmetric**
  about 0 for every curve → that symmetry *is* the linear-phase guarantee. The
  default view shows ~400 taps around the center (where the action is); use the
  toolbar to zoom/pan out to all 2047.
- **Position slider** — shows playback progress (`m:ss / m:ss`) and seeks: drag
  it to jump to any time in the loop.
- **Input (dB) slider** — ±20 dB pre-filter gain. The **clip indicator** turns
  red if any output sample reaches |x| ≥ 1.0.
- **Play / Pause / Stop / Load** buttons.
- **Improve** button — toggles the **adaptive multiband sound-improver**: it
  analyses the playing audio and continuously rewrites the EQ curve toward a
  broadcast "radio voice" target (see below). While active, manual editing is
  suspended and you watch the measured (orange) curve and the taps morph live.
- **Limiter** button — toggles the final-stage **lookahead limiter** (a Python
  port of the JS worklet limiter) for peak control + makeup.
- **Record** button — captures exactly what's sent to the output (post-filter,
  post-gain) while you drag the curve, and writes it to a 32-bit float WAV when
  you press *Stop Rec*. Great for capturing a live edit session to analyze in
  Sound Forge. (For a deterministic, scripted render with swaps at known sample
  positions, use `render_test.py` instead — see below.)

Drag a steep cut or boost while playing: the sound changes within ~one audio
block (~21 ms) with no click.

## How the seamless swap works

The audio callback keeps a **coefficient-independent delay line**: a history
of the last `N−1` *input* samples per channel. Those past inputs are valid for
*any* set of coefficients, so swapping taps never corrupts filter state. Each
block:

1. `x_ext = [history ; in_block]`
2. `y_old = fftconvolve(x_ext, b_current, 'valid')`
3. If a new design arrived (lock-free version bump), also compute `y_new` with
   `b_new`, then blend across the block with a **linear crossfade**
   (`w_old = 1−a`, `w_new = a`, `a`: 0→1). This is exactly a per-sample linear
   interpolation of the coefficients, so the level moves *monotonically* from
   old to new. Both filters share group delay `(N−1)/2`, so `y_old`/`y_new` are
   in phase and the linear blend preserves amplitude with **no +3 dB bump** (an
   equal-power `cos/sin` blend overshoots here, since it assumes *uncorrelated*
   signals — verified on a 3 kHz sine). No comb filtering either, thanks to the
   matched group delay.
4. `history = last N−1 samples of x_ext`.

Handoff is lock-free: the GUI assigns a whole new array to `holder.b_target`
and bumps `holder.version`; the callback reads `version` and picks it up. No
locks, no allocation, no I/O in the hot path.

## Performance

The audio thread is cheap (~1.7% of the block budget; ~3.4% during a swap) —
it was never the bottleneck. The CPU cost was the GUI: a full matplotlib
figure redraw is ~80 ms here, and redrawing it 10×/s just to move the clock
pegged a core.

The fix is **blitting**: the static background (axes, grid, labels, the EQ
curve when idle) is cached once, and each frame only re-paints the few artists
that actually change (curve, markers, taps, sliders, clock). On top of that:

- Sliders run with `drawon=False` so `set_val` never forces a full redraw.
- The taps axis only does a real (slow) rescale when the impulse grows past the
  current y-limit; otherwise it blits.
- The status timer skips the blit entirely when nothing visibly changed
  (paused, no clip), so an idle window costs ~0%.
- **Region-limited blits while dragging.** Profiling showed the artist drawing
  is trivial (~0.5 ms) and restoring the cached background is cheap (~0.1 ms);
  the real cost was the full-window pixel *push* (~6 ms vs ~2 ms for one axes).
  So the drag path restores the full background (cheap, and it reliably erases
  the previous frame — a sub-bbox restore leaves a curve *trail*) but pushes
  only the two plot rectangles (EQ + taps) to screen. A full drag frame —
  move point → redesign FIR → repaint both plots — is **~6 ms (≈168 fps)**,
  comfortably inside the 16.6 ms / 60 fps budget. The redesign timer runs at
  16 ms, so dragging renders at a steady 60 fps. The infrequent widget/clock
  updates keep the simpler full blit.

Result: idle/playback GUI load dropped ~**80% → <10%** of a core (≈0% paused),
and dragging went from ~12 fps (full redraws) to a steady **60 fps** with
headroom to spare.

## Adaptive sound improver (multiband) + limiter

The **Improve** button turns the manual EQ into an *adaptive multiband
leveler/compressor* that pushes a voice toward a broadcast ("radio voice")
sound — automatically, while it plays.

**Key idea — the curve *is* the per-band gain.** In a linear-phase FIR the
magnitude response is literally the gain at each frequency, so a "band" is just
a region of that curve and a multiband compressor is nothing more than per-band
gain that reacts to per-band level over time. No separate crossover filterbank
is needed: the existing `design_fir` + lock-free `CoeffHolder` + crossfade *are*
the de-zippering for a continuously moving compressor. Because every redesigned
curve shares the same `N` and linear phase (identical group delay), consecutive
swaps stay phase-coherent — no combing during the motion.

`sound_improver.py` runs an **analysis thread** (off the audio thread) that, at
~80 Hz:

1. pulls the most recent ~4096 input frames from a lock-free ring the audio
   callback fills (feed-forward detector);
2. computes a windowed power spectrum and per-band level (6 voice bands:
   ~20–120, 120–350, 350–900, 0.9–2.5 k, 2.5–6 k, 6–20 k Hz);
3. per band, a soft-knee **gain computer** + **attack/release** smoothing
   (time-aware one-pole) produces a gain-reduction value;
4. adds the fixed **"radio" tone target** (low-rumble cut, low-mid warmth,
   2–5 kHz presence lift, gentle de-ess) and emits control points →
   `design_fir` → `holder.update`.

The min effective attack is one audio block (~21 ms at 1024/48 k) — ideal for
**density / leveling**, not fast peak control. The improver deliberately leaves
headroom (it shapes *balance*, not loudness); fast peak control + makeup is the
limiter's job.

### Limiter (`limiter.py`)

A faithful Python port of the sample-accurate lookahead limiter at
<https://github.com/Artur86k/limiter> (`limiter-worklet.js`):
ring-buffer lookahead delay, instant-attack / adaptive-hold / exponential-decay
peak envelope follower, soft-knee reduction toward `saturation_db`, and makeup
`output_gain_db`. The analyser also feeds it spectral hints (centroid,
low-band energy, rms) that shape its adaptive recovery and hold — exactly as the
original took them from its main-thread analyser.

The envelope has a per-sample feedback dependency (instant attack, data-driven
hold), so it's a tight scalar loop. With **Numba** present it's JIT-compiled
(and warmed up at construction so the first audio block never stalls);
otherwise the identical code runs in CPython — measured at **~1.0 ms/block**
(4.7 % of the 21.3 ms deadline at 1024/48 k), so Numba is optional. It adds its
own lookahead latency (default 0.5 ms, negligible next to the FIR's ~21 ms).

> **Note on the original limiter's scope.** That repo is a *Chrome extension*
> (Web Audio AudioWorklet) and can only limit audio in a browser tab — it can't
> touch this app's live `sounddevice` output. This port brings the same
> algorithm into the realtime Python path so the live demo is peak-safe too.

## DSP: linear-phase FIR from the curve

Type I FIR (odd `N`, symmetric taps). `design_fir`:

1. Interpolate the points in (log f, dB) with **PCHIP** (monotone, no
   overshoot; Catmull-Rom is the "true Bézier-chain" alternative).
2. Sample onto the oversampled rfft grid (`M=8192`), dB→linear → zero-phase
   spectrum `H`.
3. `irfft` → symmetric impulse response, `fftshift`, take the center `N`,
   multiply by a **Kaiser window (β=8)**. This *oversampled
   frequency-sampling + window* method decouples curve resolution (`M`) from
   tap count (`N`) and suppresses Gibbs ripple between points.

## Tradeoffs (the point of the demo)

- **Tap count `N`** (presets 1023 / 2047 / 4095): more taps → finer
  low-frequency resolution and tighter tracking of steep edits, but **more
  latency** and **more pre-ring**.
- **Latency** = group delay `(N−1)/2` samples + block I/O. `N=2047 @ 48 kHz ≈
  21 ms`. This is inherent to linear phase.
- **Pre-ring (documented, not fixed):** symmetric taps pre-ring on steep
  edits → audible pre-echo on transients. Expected. A future minimum-phase
  variant (cepstral conversion of the same magnitude) would trade the
  linear-phase guarantee for no pre-ring.
- **Resolution vs taps:** for steep segments the measured (orange) curve
  visibly departs from the target (blue) — the windowing/resolution tradeoff
  in action.

## Files

| File               | Responsibility                                            |
|--------------------|-----------------------------------------------------------|
| `main.py`            | GUI, wiring, throttled redesign, transport, slider, taps |
| `filter_design.py`   | `design_fir`, curve interpolation, measured magnitude    |
| `audio_engine.py`    | sounddevice stream, callback, delay-line conv, crossfade, analysis ring, limiter hook |
| `curve_editor.py`    | matplotlib point add/move/delete on the top axis         |
| `sound_improver.py`  | adaptive multiband leveler/compressor (analysis thread)  |
| `limiter.py`         | lookahead limiter (Python port of the JS worklet)        |
| `test_dsp.py`        | headless tests: symmetry, tracking, **no-click** swap    |
| `test_sound_improver.py` | headless tests: limiter ceiling, gain computer, adaptive step |

## Tests

```bash
python test_dsp.py
python test_sound_improver.py
```

`test_dsp.py` verifies (no GUI / no audio device needed): taps are odd &
symmetric, the FIR tracks the target at control points, the steady-state engine
output matches a clean reference convolution, and the coefficient swap produces
**no click** at the block boundary.

`test_sound_improver.py` verifies: the limiter holds loud input below its
ceiling and passes quiet input unchanged, the compressor gain computer is
monotone, and one adaptive step yields symmetric (linear-phase) taps while
pulling down a band that is over threshold.

## Offline smoothness render (`render_test.py`)

Renders a file through the **exact same** `AudioEngine._callback` (delay-line
conv + linear crossfade), forcing coefficient swaps at known sample
positions, and writes a 32-bit float WAV you can open in Sound Forge.

```bash
# make a probe (or use your own Sound Forge file)
python render_test.py gen --out sine.wav  --signal sine --freq 1000 --dur 8 --stereo
python render_test.py gen --out noise.wav --signal noise --dur 8 --stereo

# render: toggle two EQs every 0.5 s; also write a hard-switch baseline
python render_test.py render sine.wav --mode toggle --interval 0.5 \
       --out out.wav --also-hard

# drag simulation: a fresh design every block (notch glides 300 Hz -> 6 kHz)
python render_test.py render sine.wav --mode sweep --out sweep.wav
```

It prints the swap instants (where to zoom in Sound Forge) and an objective
**click metric**: the broadband energy *added* at each swap, relative to the
signal, in dB.

**Important — pick the right probe.** White noise is great for *seeing the
spectral transition* (spectrogram / RMS envelope shows the ~21 ms crossfade
ramp), but it is a **poor click detector**: its own broadband content masks any
switching transient (the tool detects this and says the metric is N/A). Use a
**sustained sine** to detect clicks. Measured here (1 kHz sine, flat↔cut toggle):

| transition             | added splatter | verdict          |
|------------------------|----------------|------------------|
| linear crossfade       | **−112.3 dB**  | clean / smooth   |
| hard switch (no fade)  | −41.5 dB       | faint transient  |
| drag-sim sweep (xfade) | −200 dB        | clean / smooth   |

i.e. the crossfade adds ~**70 dB less** transient energy than an instantaneous
coefficient switch — the swap is effectively inaudible.

**Why linear, not equal-power?** An equal-power (`cos/sin`) crossfade is the
right choice for *uncorrelated* signals, but the two filter outputs here are
**in phase** (same linear-phase group delay). For in-phase signals
`cos θ + sin θ` reaches √2 at the midpoint → a **+3 dB level bump** in the
transition (its envelope looks like a sine — visible on a 3 kHz tone). The
**linear** blend (`1−a`, `a`) is a true coefficient interpolation: the level
moves monotonically old→new with no bump, and it measures even cleaner.
