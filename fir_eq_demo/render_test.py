"""Offline render harness to test crossfade smoothness on a real file.

It drives the *exact same* `AudioEngine._callback` used by the live app
(delay-line FFT convolution + equal-power crossfade), forces FIR coefficient
swaps at known sample positions, and writes the filtered result to a WAV you
can analyze in Sound Forge (waveform zoom at the swap marks, spectrogram, etc).

Two render modes:
  * toggle : alternate between two EQ curves every --interval seconds. Swap
             instants are at known sample positions -> zoom there to check for
             clicks. This is the click stress test.
  * sweep  : move a deep notch's center frequency a little every block, so a
             fresh design is handed off ~every block -> simulates dragging the
             curve with "several coefficient sets". The spectrogram should show
             a smoothly gliding notch with no vertical click streaks.

Use --also-hard to additionally render a HARD-SWITCH baseline (instantaneous
coefficient change, no crossfade) so you can see/hear the click the crossfade
removes.

Examples
--------
  # make a test file if you don't have one yet (or use your Sound Forge file)
  python render_test.py gen --out noise.wav --dur 12 --stereo

  # render the crossfade output + a hard-switch baseline for comparison
  python render_test.py render noise.wav --mode toggle --interval 0.5 \
         --out out_xfade.wav --also-hard

  python render_test.py render noise.wav --mode sweep --out out_sweep.wav

Filters are CUT-only + input headroom so the output never clips (clipping
would add its own artifacts and confuse the smoothness analysis).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filter_design as fd
from audio_engine import AudioEngine, CoeffHolder

N_TAPS = fd.N_TAPS
M_DESIGN = fd.M_DESIGN
BLOCKSIZE = 1024


# ---- test signal generator ----------------------------------------------
def cmd_gen(args):
    n = int(args.dur * args.fs)
    t = np.arange(n) / args.fs
    if args.signal == "noise":
        rng = np.random.default_rng(args.seed)
        x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
        desc = "white noise"
    elif args.signal == "sine":
        x = np.sin(2 * np.pi * args.freq * t)
        desc = f"{args.freq:.0f} Hz sine"
    else:
        raise ValueError(args.signal)
    x = (x * args.amp).astype(np.float32)
    x = x[:, None] * np.ones((1, 2 if args.stereo else 1), dtype=np.float32)
    sf.write(args.out, x, args.fs, subtype="FLOAT")
    print(f"Wrote {args.out}: {args.dur}s {'stereo' if args.stereo else 'mono'} "
          f"{desc} @ {args.fs} Hz, peak {args.amp}")


# ---- filter banks --------------------------------------------------------
def filter_flat(fs):
    return fd.design_fir([(20, 0), (1000, 0), (20000, 0)], fs, N_TAPS, M_DESIGN)


def filter_cut(fs):
    # Strong, broadband CUT (never boosts -> no clipping). Big magnitude change
    # from flat makes the crossfade work hard.
    pts = [(20, 0), (300, -18), (1500, -3), (6000, -20), (20000, -10)]
    return fd.design_fir(pts, fs, N_TAPS, M_DESIGN)


def filter_notch(fs, fc):
    # Deep narrow-ish notch at fc, flat elsewhere (cut-only).
    pts = [(20, 0), (fc / 2, 0), (fc, -22), (fc * 2, 0), (20000, 0)]
    return fd.design_fir(pts, fs, N_TAPS, M_DESIGN)


# ---- core render loop ----------------------------------------------------
def render(audio, fs, schedule, gain_db, hard):
    """Run `audio` (n,2) through the engine, applying coefficient sets per the
    schedule {block_index: b}. Returns (out (M,2), swap_sample_indices)."""
    b0 = filter_flat(fs)
    holder = CoeffHolder(b0)
    eng = AudioEngine(holder, blocksize=BLOCKSIZE, fs_fallback=fs)
    eng.fs = fs
    eng.audio = np.ascontiguousarray(audio, dtype=np.float32)
    eng.n_frames = audio.shape[0]
    eng.read_pos = 0
    eng.playing = True
    eng.input_gain = 10.0 ** (gain_db / 20.0)

    L = BLOCKSIZE
    nblocks = eng.n_frames // L
    out = np.zeros((nblocks * L, 2), dtype=np.float32)
    buf = np.zeros((L, 2), dtype=np.float32)
    swaps = []

    for blk in range(nblocks):
        if blk in schedule:
            b = schedule[blk]
            swaps.append(blk * L)
            if hard:
                # Instantaneous coefficient change: sync version so the
                # callback does NOT crossfade -> a step at the block edge.
                eng.b_current = b
                eng.local_version = holder.version
            else:
                holder.update(b)         # -> callback crossfades over block
        eng._callback(buf, L, None, None)
        out[blk * L:(blk + 1) * L] = buf

    return out, swaps, eng


def build_schedule(mode, fs, nblocks, interval_s):
    L = BLOCKSIZE
    sched = {}
    if mode == "toggle":
        step = max(1, int(round(interval_s * fs / L)))
        flat = filter_flat(fs)
        cut = filter_cut(fs)
        for i, blk in enumerate(range(step, nblocks, step)):
            sched[blk] = cut if i % 2 == 0 else flat
    elif mode == "sweep":
        # New notch design every block, center gliding 300 Hz -> 6 kHz.
        fcs = np.logspace(np.log10(300), np.log10(6000), nblocks)
        for blk in range(1, nblocks):
            sched[blk] = filter_notch(fs, float(fcs[blk]))
    else:
        raise ValueError(mode)
    return sched


# ---- smoothness report ---------------------------------------------------
def _band_powers(seg, fs):
    """Windowed energy above and at/below fs/4 (out-of-band, in-band)."""
    w = np.hanning(len(seg))
    S = np.abs(np.fft.rfft(seg * w)) ** 2
    f = np.fft.rfftfreq(len(seg), 1.0 / fs)
    return float(S[f > fs / 4].sum()), float(S[f <= fs / 4].sum())


def report(out, swaps, fs):
    mono = out.mean(axis=1)
    L = BLOCKSIZE

    # Raw sample-to-sample jump (only meaningful for tonal/quiet signals —
    # white noise masks it because the signal itself jumps every sample).
    d = np.abs(np.diff(out, axis=0)).max(axis=1)
    typ = np.percentile(d, 99.9)

    # Click audibility: the broadband energy ADDED at a swap (over the
    # signal's own steady-state out-of-band level), normalized by the in-band
    # signal level. This is signal-independent and matches perception:
    #   * pure tone, clean crossfade -> added ~0  -> very negative dB -> clean
    #   * pure tone, hard switch     -> step splatter -> high dB -> click
    #   * white noise                -> swap looks like steady -> added ~0 ->
    #     "clean", i.e. any click is genuinely buried/inaudible in the noise
    #     (which is exactly why a tone is the better probe).
    half = L
    rng = np.random.default_rng(0)
    swap_set = {int(s) for s in swaps}
    ref_starts = [int(x) for x in rng.integers(half, len(mono) - 2 * half, 300)
                  if not any(abs(int(x) + half - s) < 2 * half for s in swap_set)]
    oob_steady = np.median([_band_powers(mono[s:s + 2 * half], fs)[0]
                            for s in ref_starts])
    inband_sig = np.median([_band_powers(mono[s:s + 2 * half], fs)[1]
                            for s in ref_starts]) + 1e-20

    worst_db = -200.0
    for s in swaps:
        a, b = max(0, s - half), min(len(mono), s + half)
        if b - a >= half:
            oob_swap = _band_powers(mono[a:b], fs)[0]
            added = max(oob_swap - oob_steady, 0.0)
            worst_db = max(worst_db, 10.0 * np.log10(added / inband_sig + 1e-20))

    # Is the probe tonal enough for the splatter metric to mean anything?
    signal_oob_db = 10.0 * np.log10(oob_steady / inband_sig + 1e-20)
    reliable = signal_oob_db < -20.0

    print(f"\nSmoothness report ({len(swaps)} swaps):")
    print(f"  raw |diff| 99.9th pct = {typ:.5f}  max = {d.max():.5f}  "
          f"(masked by broadband signals)")
    if reliable:
        print(f"  worst added broadband splatter at a swap = {worst_db:.1f} dB "
              f"rel. signal  ({_verdict(worst_db)})")
    else:
        print(f"  signal is broadband (out-of-band floor {signal_oob_db:.0f} dB) "
              f"-> splatter metric N/A; re-run with `gen --signal sine`")
    print(f"  output peak = {np.max(np.abs(out)):.3f} "
          f"({'CLIPPED' if np.max(np.abs(out)) >= 0.999 else 'no clip'})")


def _verdict(db):
    # Added broadband (out-of-band) energy at the transition, rel. to signal.
    if db > -30:
        return "AUDIBLE CLICK"
    if db > -50:
        return "faint transient"
    return "clean / smooth (no added splatter)"


def cmd_render(args):
    audio, file_fs = sf.read(args.input, dtype="float64", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    fs = int(file_fs)
    nblocks = audio.shape[0] // BLOCKSIZE
    if nblocks < 2:
        sys.exit("Input too short.")

    sched = build_schedule(args.mode, fs, nblocks, args.interval)
    swap_times = sorted(blk * BLOCKSIZE / fs for blk in sched)

    out, swaps, eng = render(audio, fs, sched, args.gain_db, hard=False)
    sf.write(args.out, out, fs, subtype="FLOAT")
    print(f"Wrote {args.out}  ({len(out)/fs:.2f}s @ {fs} Hz, 32-bit float)")
    if args.mode == "toggle":
        shown = ", ".join(f"{t:.3f}s" for t in swap_times[:12])
        print(f"Swap instants (zoom here in Sound Forge): {shown}"
              f"{' ...' if len(swap_times) > 12 else ''}")
    else:
        print(f"{len(sched)} per-block designs (notch glide 300 Hz -> 6 kHz).")
    report(out, swaps, fs)

    if args.also_hard:
        outh, swapsh, _ = render(audio, fs, sched, args.gain_db, hard=True)
        hard_path = os.path.splitext(args.out)[0] + "_hardswitch.wav"
        sf.write(hard_path, outh, fs, subtype="FLOAT")
        print(f"\nWrote {hard_path}  (instantaneous switch, NO crossfade — "
              f"baseline for comparison)")
        report(outh, swapsh, fs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen", help="generate a test WAV (noise or sine)")
    g.add_argument("--out", required=True)
    g.add_argument("--signal", choices=["noise", "sine"], default="noise",
                   help="noise = good for seeing the spectral transition; "
                        "sine = sensitive click detector")
    g.add_argument("--freq", type=float, default=1000.0, help="sine frequency")
    g.add_argument("--dur", type=float, default=12.0)
    g.add_argument("--fs", type=int, default=48000)
    g.add_argument("--amp", type=float, default=0.9)
    g.add_argument("--stereo", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_gen)

    r = sub.add_parser("render", help="render filtered output with swaps")
    r.add_argument("input")
    r.add_argument("--out", required=True)
    r.add_argument("--mode", choices=["toggle", "sweep"], default="toggle")
    r.add_argument("--interval", type=float, default=0.5,
                   help="toggle period in seconds (toggle mode)")
    r.add_argument("--gain-db", type=float, default=-6.0,
                   help="input headroom gain (dB) to avoid clipping")
    r.add_argument("--also-hard", action="store_true",
                   help="also render a hard-switch (no crossfade) baseline")
    r.set_defaults(func=cmd_render)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
