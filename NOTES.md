# Project Notes — Real-Time Linear-Phase FIR EQ + Adaptive Sound Improver

Working notes that travel with the repo (so they're available on any machine).
For the *how it works* detail see `fir_eq_demo/README.md`; this file is the
project status + decisions + what's next.

Last updated: 2026-06-22.

## Where things stand

The base app is a single-window linear-phase FIR EQ that plays a looping stereo
file and hot-swaps redesigned coefficients into the audio thread click-free
(crossfade). On top of that, an **adaptive multiband "sound improver"** and a
**ported lookahead limiter** were added.

Status: classic-DSP sound-improver is **done and tested**. The CNN layer is
**not started** (agreed next step).

## Key design decision (the load-bearing idea)

In a linear-phase FIR the magnitude curve **is** the per-frequency gain, so a
"band" is just a region of that curve and a multiband compressor is nothing
more than *per-band gain that reacts to per-band level over time*. Therefore:

- **No separate crossover filterbank is needed.** The existing
  `design_fir` + lock-free `CoeffHolder` + per-block crossfade already
  de-zipper a continuously moving curve.
- Every redesigned curve shares the same `N` and linear phase (identical group
  delay) → consecutive swaps are phase-coherent, no combing.
- **Minimum effective attack = one audio block (~21 ms @ 1024/48k).** Great for
  density/leveling; *not* for fast peak control. Fast peak control + makeup is
  the limiter's job. The improver deliberately leaves headroom (shapes balance,
  not loudness).

## What each new piece does

- `fir_eq_demo/sound_improver.py` — `AnalysisEngine`: a worker thread (~80 Hz,
  off the audio thread) that reads the most recent ~4096 input frames from a
  lock-free ring the audio callback fills (feed-forward detector), measures 6
  voice bands, runs a soft-knee gain computer + time-aware attack/release per
  band, adds a fixed `RADIO_TARGET` tone curve, and emits new taps via
  `design_fir` → `holder.update`. Also feeds spectral hints to the limiter.
- `fir_eq_demo/limiter.py` — `Limiter`: faithful Python port of the JS
  AudioWorklet limiter at <https://github.com/Artur86k/limiter>. Ring-buffer
  lookahead delay, instant-attack/adaptive-hold/exp-decay envelope, soft-knee
  reduction toward `saturation_db`, makeup `output_gain_db`. Numba-JIT if
  available (warmed up at construction so the audio thread never stalls);
  pure-Python fallback ≈1.0 ms/block (4.7% of the deadline) → Numba optional.
  NOTE: the original repo is a *Chrome extension* and can only limit a browser
  tab — it can't touch this app's live `sounddevice` output, which is why it
  was ported.
- `fir_eq_demo/audio_engine.py` — added the lock-free analysis ring (single
  producer = audio callback, single consumer = improver) and the limiter hook.
  The old hard `np.clip` is now a last-resort safety clamp behind the limiter;
  the clip indicator still works.
- `fir_eq_demo/main.py` — **Improve** and **Limiter** toggle buttons, built
  lazily after `open_stream()` finalises the device rate. While Improve is on,
  manual editing is suspended and the GUI mirrors the improver's curve (taps +
  measured overlay) on the GUI thread (the improver thread only writes the
  holder; no matplotlib calls off-thread).
- `fir_eq_demo/curve_editor.py` — `enabled` flag to suspend manual edits while
  Auto drives the curve.

## How to run / resume

```bash
git pull
pip install -r fir_eq_demo/requirements.txt   # numba optional, speeds limiter
cd fir_eq_demo
python main.py                 # Play -> Improve -> Limiter
python test_dsp.py
python test_sound_improver.py
```

## Tuning knobs

- Compression character: `threshold_db`, `ratio`, `knee_db`, `attack_ms`,
  `release_ms` on `AnalysisEngine` (in `sound_improver.py`).
- Tone target: `RADIO_TARGET` and `BAND_EDGES` constants in `sound_improver.py`.
- Limiter: defaults are faithful to the worklet (`saturation_db=-6`,
  `output_gain_db=+6` → ceiling at 0 dBFS, "normalize"-style). Lower
  `output_gain_db` for purely protective (non-loudness-maximizing) behavior.
- Thresholds are on a dBFS-ish scale and **need tuning by ear** against real
  voice material. `AnalysisEngine.band_level_db` is exposed for live printing.

## Next steps (planned, not started)

1. Tune band thresholds against a real voice file (print `band_level_db` live).
2. **CNN layer above the DSP** (the original goal):
   - First: a mel-spectrogram **VAD / content classifier** (speech / music /
     silence) that *gates* adaptation so it doesn't pump on silence or wreck
     music. Small net, trainable without a big corpus. Adds PyTorch/ONNX.
   - Then: a **target-curve predictor** so `RADIO_TARGET` adapts per voice
     (deep male vs thin female land on "radio", not the same literal spectrum).
   - Later (big, needs a paired dataset): end-to-end learned band gains.
3. Optional: a true fast multiband compressor (crossover filterbank in the
   audio thread) only if block-rate density isn't punchy enough.

## Gotchas to remember

- Don't call matplotlib from the improver thread — it only writes the holder;
  the GUI mirrors on its own timer.
- Keep crossfades between filters of the **same N** (matched group delay).
- The improver measures the **input** (feed-forward); the FIR's ~21 ms group
  delay means the gain effectively has built-in lookahead — that's fine.
