#!/usr/bin/env python3
"""Track Analyzer — release readiness for darksynth/industrial."""

import argparse, glob, json, os, sys
import numpy as np
import librosa
import pyloudnorm as pyln
from scipy import signal

DEFAULT_EXTENSIONS = {".wav", ".aiff", ".aif", ".flac"}

BANDS = [
    ("sub",       20,    60),
    ("bass",      60,   250),
    ("low-mid",  250,  1000),
    ("high-mid", 1000,  6000),
    ("presence", 6000, 12000),
    ("air",     12000, 20000),
]
# Expected delta vs bass band (min_dB, max_dB) for darksynth/industrial
GENRE_PROFILE = {
    "sub":      (-3,   5),
    "bass":     ( 0,   0),
    "low-mid":  (-16,  -7),
    "high-mid": (-26, -14),
    "presence": (-36, -22),
    "air":      (-52, -30),
}
WEIGHTS = {"technical": .25, "frequency": .20, "stereo": .20,
           "dynamics": .15, "artifacts": .10, "genre": .10}


# ── Audio ─────────────────────────────────────────────────────────────────────

def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.ndim == 2 and y.shape[0] > 2:
        y = y[:2]
    return y.astype(np.float64), int(sr)


# ── Measurements ──────────────────────────────────────────────────────────────

def measure_loudness(y, sr):
    data = y.T
    meter = pyln.Meter(sr)
    try:
        integrated = float(meter.integrated_loudness(data))
    except Exception:
        integrated = float("nan")
    try:
        lra = float(meter.loudness_range(data))
    except Exception:
        lra = float("nan")
    win, hop = int(3 * sr), int(sr)
    st_vals = []
    if data.shape[0] >= win:
        for i in range(0, data.shape[0] - win, hop):
            try:
                v = float(meter.integrated_loudness(data[i:i+win]))
                if np.isfinite(v):
                    st_vals.append(v)
            except Exception:
                pass
    st_max = max(st_vals) if st_vals else integrated
    tp = 0.0
    for ch in range(y.shape[0]):
        over = signal.resample_poly(y[ch], 4, 1)
        tp = max(tp, float(np.max(np.abs(over))))
    tp_dbtp = 20.0 * np.log10(tp) if tp > 0 else -120.0
    clips = int(np.sum(np.any(np.abs(y) >= 0.9999, axis=0)))
    dc = float(np.max(np.abs([np.mean(y[ch]) for ch in range(y.shape[0])])))
    return {"integrated": integrated, "short_term_max": float(st_max),
            "true_peak": float(tp_dbtp), "lra": lra, "clips": clips, "dc": dc}


def measure_bands(y, sr):
    y_mono = np.mean(y, axis=0)
    n_fft = 4096
    S = np.abs(librosa.stft(y_mono, n_fft=n_fft, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = S ** 2
    out = {}
    for name, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            out[name] = None
            continue
        rms = float(np.sqrt(np.mean(power[mask, :])))
        out[name] = float(20 * np.log10(rms)) if rms > 0 else None
    return out


def measure_stereo(y, sr):
    L, R = y[0], y[1]
    mid, side = (L + R) * .5, (L - R) * .5
    rms_mid = float(np.sqrt(np.mean(mid ** 2)))
    rms_side = float(np.sqrt(np.mean(side ** 2)))
    width = rms_side / rms_mid if rms_mid > 0 else 0.0
    stereo_rms = float(np.sqrt(np.mean((L ** 2 + R ** 2) * .5)))
    mono_db = 20 * np.log10(rms_mid / stereo_rms) if stereo_rms > 0 and rms_mid > 0 else 0.0
    nyq = sr / 2.0
    band_corr = {}
    for name, lo, hi in BANDS:
        lo_n, hi_n = lo / nyq, min(hi / nyq, .999)
        if lo_n <= 0 or lo_n >= hi_n:
            band_corr[name] = 1.0
            continue
        try:
            b, a = signal.butter(2, [lo_n, hi_n], btype="band")
            Lf, Rf = signal.filtfilt(b, a, L), signal.filtfilt(b, a, R)
            c = float(np.corrcoef(Lf, Rf)[0, 1])
            band_corr[name] = c if np.isfinite(c) else 1.0
        except Exception:
            band_corr[name] = 1.0
    return {"width": float(width), "mono_db": float(mono_db), "band_corr": band_corr}


def measure_phase(y, sr):
    L, R = y[0], y[1]
    frame = int(.1 * sr)
    corrs = [float(np.corrcoef(L[i:i+frame], R[i:i+frame])[0, 1])
             for i in range(0, len(L) - frame, frame)
             if np.std(L[i:i+frame]) > 1e-9 and np.std(R[i:i+frame]) > 1e-9]
    if not corrs:
        return {"sustained_neg": False, "min": 1.0, "avg": 1.0}
    arr = np.array(corrs)
    count, sustained = 0, False
    for neg in arr < 0:
        count = (count + 1) if neg else 0
        if count >= 5:
            sustained = True
            break
    return {"sustained_neg": bool(sustained), "min": float(np.min(arr)),
            "avg": float(np.mean(arr))}


def measure_dynamics(y, sr, loud):
    y_mono = np.mean(y, axis=0)
    rms = float(np.sqrt(np.mean(y_mono ** 2)))
    peak = float(np.max(np.abs(y_mono)))
    crest = 20 * np.log10(peak / rms) if rms > 0 else 0.0
    psr = loud["true_peak"] - loud["short_term_max"]
    return {"crest": float(crest), "psr": float(psr)}


def measure_artifacts(y, sr):
    y_mono = np.mean(y, axis=0)
    n = len(y_mono)

    # Clicks: diff > 1.0 (half-scale jump per sample — above 8kHz at 0dBFS; musical content won't reach this)
    diff = np.abs(np.diff(y_mono))
    ck_peaks, _ = signal.find_peaks(diff, height=1.0, distance=int(.05 * sr))
    clicks = len(ck_peaks)

    # Noise floor: only meaningful when the track has genuinely quiet sections (>30dB below peak).
    # For continuously loud/compressed music the 5th-percentile RMS is quiet musical content, not noise.
    frame = max(int(.01 * sr), 1)
    rms_db = [20 * np.log10(max(float(np.sqrt(np.mean(y_mono[i:i+frame] ** 2))), 1e-10))
              for i in range(0, n - frame, frame)]
    peak_rms = float(np.max(rms_db)) if rms_db else 0.0
    quiet = [v for v in rms_db if v < peak_rms - 30]
    noise_floor = float(np.median(quiet)) if len(quiet) >= len(rms_db) * 0.01 else None

    # Zero-crossing spikes: abrupt ZCR changes indicating edit glitches
    zcr = np.array([float(np.sum(np.diff(np.signbit(y_mono[i:i+frame]))) / frame)
                    for i in range(0, n - frame, frame)])
    zcr_spikes = 0
    if len(zcr) > 2:
        zd = np.abs(np.diff(zcr))
        thresh = np.percentile(zd, 99) * 5
        if thresh > 0:
            zp, _ = signal.find_peaks(zd, height=thresh, distance=20)
            zcr_spikes = len(zp)

    # Internal silence gaps > 200ms in middle 80% of track
    mid_s, mid_e = n // 10, n - n // 10
    in_gap, gaps, run = False, 0, 0
    min_run = int(.2 * sr)
    for i in range(mid_s, mid_e - frame, frame):
        silent = float(np.sqrt(np.mean(y_mono[i:i+frame] ** 2))) < 0.001
        if silent:
            run += frame
            if run >= min_run and not in_gap:
                gaps += 1
                in_gap = True
        else:
            run, in_gap = 0, False

    return {"clicks": clicks, "noise_floor": noise_floor,
            "zcr_spikes": zcr_spikes, "silence_gaps": gaps}


# ── Scoring (each returns (0-100, [(priority, actionable_msg)]) ───────────────

def score_technical(loud):
    s, issues = 100, []
    if loud["clips"] > 0:
        s -= 40
        issues.append((0, f"{loud['clips']} clipped sample(s) — reduce pre-limiter gain"))
    if loud["true_peak"] > -1.0:
        s -= 25
        issues.append((1, f"True peak {loud['true_peak']:.1f} dBTP — hard limit to -1.0 dBTP before export"))
    elif loud["true_peak"] > -3.0:
        s -= 8
        issues.append((6, f"True peak {loud['true_peak']:.1f} dBTP — tighten limiter ceiling to -1.5"))
    lufs = loud["integrated"]
    if np.isfinite(lufs):
        if lufs > -9:
            s -= 20
            issues.append((2, f"Integrated {lufs:.1f} LUFS too hot — lower limiter threshold"))
        elif lufs < -23:
            s -= 10
            issues.append((7, f"Integrated {lufs:.1f} LUFS too quiet — check gain staging"))
        elif lufs > -12 or lufs < -20:
            s -= 5
    else:
        s -= 15
    if loud["dc"] > 0.01:
        s -= 20
        issues.append((3, f"DC offset {loud['dc']:.4f} — apply DC filter before export"))
    elif loud["dc"] > 0.001:
        s -= 5
    return max(0, min(100, s)), issues


def score_frequency(bands, ref_bands=None):
    if ref_bands is None:
        s, issues = 80, []
        sub, bass = bands.get("sub"), bands.get("bass")
        if sub is not None and bass is not None:
            if sub < bass - 12:
                s -= 12
                issues.append((9, f"Sub very weak vs bass ({sub-bass:+.0f}dB) — check low-end content"))
        hm = bands.get("high-mid")
        if hm is not None and bass is not None and hm > bass - 8:
            s -= 10
            issues.append((8, f"High-mid {hm-bass:+.0f}dB vs bass — may sound harsh/thin"))
        return max(0, min(100, s)), issues

    s, issues, deltas = 100, [], {}
    for name, *_ in BANDS:
        t, r = bands.get(name), ref_bands.get(name)
        if t is None or r is None:
            continue
        d = t - r
        deltas[name] = d
        abd = abs(d)
        if abd > 9:
            s -= 18
        elif abd > 6:
            s -= 10
        elif abd > 3:
            s -= 4

    hz_hint = {"sub": "40Hz shelf", "bass": "100Hz", "low-mid": "350Hz",
               "high-mid": "2kHz", "presence": "8kHz", "air": "14kHz shelf"}
    for name, d in sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:2]:
        if abs(d) > 3:
            action = "cut" if d > 0 else "boost"
            issues.append((4, f"{name} {d:+.1f}dB vs reference — {action} {hz_hint.get(name, name)}"))
    return max(0, min(100, s)), issues


def score_stereo(stereo, phase):
    s, issues = 100, []
    if phase["sustained_neg"]:
        s -= 35
        worst = min(stereo["band_corr"], key=stereo["band_corr"].get)
        issues.append((1, f"Phase correlation negative (min {phase['min']:.2f}) "
                          f"— check stereo widener, worst in {worst}"))
    sub_c = stereo["band_corr"].get("sub", 1.0)
    if sub_c < 0.7:
        s -= 25
        issues.append((2, f"Sub L/R correlation {sub_c:.2f} — collapse sub below 80Hz to mono"))
    elif sub_c < 0.85:
        s -= 12
        issues.append((6, f"Sub L/R correlation {sub_c:.2f} — consider mono filter at 80Hz"))
    if stereo["mono_db"] < -3.0:
        s -= 20
        issues.append((3, f"Mono compat {stereo['mono_db']:.1f}dB loss — anti-phase content present"))
    elif stereo["mono_db"] < -1.0:
        s -= 8
        issues.append((7, f"Mono compat {stereo['mono_db']:.1f}dB loss — verify mono playback"))
    if stereo["width"] > 1.1:
        s -= 10
        issues.append((8, f"Stereo width {stereo['width']:.2f} M/S ratio — may fold oddly on some systems"))
    return max(0, min(100, s)), issues


def score_dynamics(dyn, loud):
    s, issues = 100, []
    lra = loud["lra"]
    if np.isfinite(lra):
        if lra < 2:
            s -= 25
            issues.append((3, f"LRA {lra:.1f} LU — severely over-compressed, ease limiter by 4+ dB"))
        elif lra < 4:
            s -= 12
            issues.append((5, f"LRA {lra:.1f} LU — over-compressed, ease limiter threshold"))
        elif lra > 14:
            s -= 8
            issues.append((7, f"LRA {lra:.1f} LU — very dynamic, may need limiting for streaming"))
    if dyn["crest"] < 6:
        s -= 20
        issues.append((4, f"Crest factor {dyn['crest']:.1f}dB — heavily limited, check limiter settings"))
    elif dyn["crest"] < 9:
        s -= 8
    if dyn["psr"] < 4:
        s -= 15
        issues.append((6, f"PSR {dyn['psr']:.1f}dB — over-limited, increase peak-to-loudness margin"))
    return max(0, min(100, s)), issues


def score_artifacts(arts):
    s, issues = 100, []
    if arts["clicks"] > 0:
        s -= min(40, arts["clicks"] * 15)
        issues.append((2, f"{arts['clicks']} click(s) detected — check edit points and clip limiting"))
    nf = arts["noise_floor"]
    if nf is not None:
        if nf > -40:
            s -= 20
            issues.append((4, f"Noise floor {nf:.0f}dB — check source recordings for hum/hiss"))
        elif nf > -50:
            s -= 8
            issues.append((8, f"Noise floor {nf:.0f}dB — mild background noise"))
    if arts["silence_gaps"] > 0:
        s -= min(20, arts["silence_gaps"] * 10)
        issues.append((5, f"{arts['silence_gaps']} internal silence gap(s) — dead spots in arrangement"))
    if arts["zcr_spikes"] > 5:
        s -= 10
        issues.append((7, f"{arts['zcr_spikes']} zero-crossing anomalies — possible edit glitches"))
    return max(0, min(100, s)), issues


def score_genre(bands):
    bass_val = bands.get("bass")
    if bass_val is None:
        return 70, []
    s, issues, deviations = 100, [], []
    for name, (lo, hi) in GENRE_PROFILE.items():
        if name == "bass":
            continue
        val = bands.get(name)
        if val is None:
            continue
        actual_delta = val - bass_val
        center = (lo + hi) / 2
        tolerance = (hi - lo) / 2 + 3
        overshoot = max(0.0, abs(actual_delta - center) - tolerance)
        if overshoot > 6:
            s -= 12
            deviations.append((overshoot, name, actual_delta, center))
        elif overshoot > 0:
            s -= 5
    if deviations:
        _, worst_name, actual_delta, target = max(deviations, key=lambda x: x[0])
        diff = actual_delta - target
        tag = "too prominent" if diff > 0 else "too weak"
        issues.append((9, f"{worst_name} {tag} for darksynth profile "
                          f"({actual_delta:+.0f}dB vs bass, target {target:+.0f}dB)"))
    return max(0, min(100, s)), issues


# ── Aggregation ───────────────────────────────────────────────────────────────

def weighted_overall(scores):
    return sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)


def top_blockers(all_issues, n=3):
    seen, out = set(), []
    for _, msg in sorted(all_issues):
        if msg not in seen:
            out.append(msg)
            seen.add(msg)
        if len(out) == n:
            break
    return out


def fix_effort(scores, ov):
    if min(scores.values()) < 50 or ov < 60:
        return "High"
    if min(scores.values()) >= 70 and ov >= 80:
        return "Low"
    return "Medium"


def verdict(ov):
    return "Ready for release" if ov > 80 else ("Needs work" if ov >= 60 else "Significant issues")


def status_of(s):
    return "PASS" if s >= 80 else ("WARNING" if s >= 60 else "FAIL")


# ── Output ────────────────────────────────────────────────────────────────────

CATS = [
    ("Technical safety",  "technical"),
    ("Frequency balance", "frequency"),
    ("Stereo / phase",    "stereo"),
    ("Dynamics",          "dynamics"),
    ("Artifacts",         "artifacts"),
    ("Genre fit",         "genre"),
]


_SHORT = {"sub": "sub", "bass": "bass", "low-mid": "lo-mid",
          "high-mid": "hi-mid", "presence": "pres", "air": "air"}
_BNAMES = [b[0] for b in BANDS]


def print_measured_data(raw, ref_bands=None):
    loud   = raw["loudness"]
    bands  = raw["bands"]
    stereo = raw["stereo"]
    phase  = raw["phase"]
    dyn    = raw["dynamics"]
    arts   = raw["artifacts"]

    def band_row(fn):
        return "   ".join(f"{_SHORT[n]} {fn(n)}" for n in _BNAMES)

    lufs = f"{loud['integrated']:.1f}" if np.isfinite(loud["integrated"]) else "n/a"
    lra  = f"{loud['lra']:.1f}"        if np.isfinite(loud["lra"])        else "n/a"

    print("MEASURED DATA")
    print(f"  Loudness    {lufs} LUFS   {loud['true_peak']:.2f} dBTP   "
          f"{lra} LU LRA   {loud['short_term_max']:.1f} LUFS ST max")
    def _energy(n):
        v = bands.get(n)
        return f"{v:.1f}" if v is not None else "n/a"
    def _corr(n):
        return f"{stereo['band_corr'].get(n, 1.0):.2f}"
    print(f"  Freq (dB)   {band_row(_energy)}")
    if ref_bands:
        def _delta(n):
            t, r = bands.get(n), ref_bands.get(n)
            return f"{t-r:+.1f}" if t is not None and r is not None else "n/a"
        print(f"  vs ref      {band_row(_delta)}")
    else:
        print( "  vs ref      no reference")
    print(f"  Stereo      width {stereo['width']:.2f}   mono {stereo['mono_db']:+.1f} dB")
    print(f"  L/R corr    {band_row(_corr)}")
    print(f"  Dynamics    crest {dyn['crest']:.1f} dB   PSR {dyn['psr']:.1f} dB")
    worst = min(stereo["band_corr"], key=stereo["band_corr"].get)
    neg   = "   [SUSTAINED NEG]" if phase["sustained_neg"] else ""
    print(f"  Phase       avg {phase['avg']:.2f}   min {phase['min']:.2f}   worst {worst}{neg}")
    nf_str = f"{arts['noise_floor']:.0f} dB" if arts["noise_floor"] is not None else "n/a"
    print(f"  Artifacts   {arts['clicks']} clicks   DC {loud['dc']:+.6f}   noise floor {nf_str}")


def print_track_report(scores, ov, blockers, effort, raw, ref_bands=None):
    print(f"VERDICT: {verdict(ov)} — {ov:.0f}%")
    print()
    print(f"{'Category':<20} | {'Score':>5} | Status")
    print(f"{'-'*20}-+-{'-'*5}-+-{'-'*7}")
    for label, key in CATS:
        sc = scores[key]
        print(f"{label:<20} | {sc:>5.0f} | {status_of(sc)}")
    print()
    print_measured_data(raw, ref_bands)
    print()
    print("TOP BLOCKERS")
    if blockers:
        for b in blockers:
            print(f"  {b}")
    else:
        print("  None")
    print()
    print(f"FIX EFFORT: {effort}")


def print_ranking(results):
    cols = (28, 6, 20, 40, 6)
    header = f"{'Track':<{cols[0]}} | {'Score':>{cols[1]}} | {'Verdict':<{cols[2]}} | {'Main blocker':<{cols[3]}} | Effort"
    sep = "-+-".join("-" * c for c in cols)
    print("RANKING")
    print(header)
    print(sep)
    for r in sorted(results, key=lambda x: x["overall"], reverse=True):
        name = os.path.splitext(os.path.basename(r["file"]))[0]
        name = name[:cols[0]-1] if len(name) >= cols[0] else name
        blocker = (r["blockers"][0][:cols[3]-1] if r["blockers"] else "—")
        verd = verdict(r["overall"])[:cols[2]-1]
        print(f"{name:<{cols[0]}} | {r['overall']:>{cols[1]-1}.0f}% | "
              f"{verd:<{cols[2]}} | {blocker:<{cols[3]}} | {r['effort']}")
    print()


# ── Input resolution ─────────────────────────────────────────────────────────

def parse_extensions(spec):
    if not spec:
        return set(DEFAULT_EXTENSIONS)
    exts = set()
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        exts.add(part if part.startswith(".") else f".{part}")
    return exts


def _is_glob(entry):
    return any(ch in entry for ch in "*?[")


def _collect_dir(dir_path, recursive, extensions):
    matches = []
    if recursive:
        for root, _dirs, files in os.walk(dir_path):
            for name in files:
                if os.path.splitext(name)[1].lower() in extensions:
                    matches.append(os.path.join(root, name))
    else:
        for name in os.listdir(dir_path):
            full = os.path.join(dir_path, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in extensions:
                matches.append(full)
    return matches


def resolve_inputs(entries, recursive=False, extensions=None):
    """Expand a list of file/directory/glob entries into a sorted, deduplicated
    list of file paths. Returns (files, warnings) — warnings are human-readable
    messages for entries that resolved to nothing (missing path, empty
    directory, or glob with no matches); callers should surface them but keep
    going, only treating an empty final `files` list as fatal."""
    extensions = extensions if extensions is not None else DEFAULT_EXTENSIONS
    found, warnings = [], []

    for entry in entries:
        if _is_glob(entry):
            matches = glob.glob(entry, recursive=recursive)
            expanded = []
            for m in matches:
                if os.path.isdir(m):
                    expanded.extend(_collect_dir(m, recursive, extensions))
                elif os.path.isfile(m):
                    expanded.append(m)
            if not expanded:
                warnings.append(f"No files matched pattern: {entry}")
            found.extend(expanded)
        elif os.path.isdir(entry):
            matches = _collect_dir(entry, recursive, extensions)
            if not matches:
                warnings.append(f"No matching audio files in directory: {entry}")
            found.extend(matches)
        elif os.path.isfile(entry):
            found.append(entry)
        else:
            warnings.append(f"Not found: {entry}")

    deduped = {os.path.realpath(f): f for f in found}
    return sorted(deduped.values()), warnings


# ── Core ──────────────────────────────────────────────────────────────────────

def analyze(path, ref_bands=None):
    y, sr = load_audio(path)
    loud = measure_loudness(y, sr)
    bands = measure_bands(y, sr)
    stereo = measure_stereo(y, sr)
    phase = measure_phase(y, sr)
    dyn = measure_dynamics(y, sr, loud)
    arts = measure_artifacts(y, sr)

    t_s, t_i = score_technical(loud)
    f_s, f_i = score_frequency(bands, ref_bands)
    st_s, st_i = score_stereo(stereo, phase)
    d_s, d_i = score_dynamics(dyn, loud)
    a_s, a_i = score_artifacts(arts)
    g_s, g_i = score_genre(bands)

    scores = {"technical": t_s, "frequency": f_s, "stereo": st_s,
              "dynamics": d_s, "artifacts": a_s, "genre": g_s}
    ov = weighted_overall(scores)
    blockers = top_blockers(t_i + st_i + d_i + f_i + a_i + g_i)
    effort = fix_effort(scores, ov)

    return {"file": path, "scores": scores, "overall": ov,
            "blockers": blockers, "effort": effort,
            "raw": {"loudness": loud, "bands": bands, "stereo": stereo,
                    "phase": phase, "dynamics": dyn, "artifacts": arts}}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Track Analyzer — WAV release readiness for darksynth/industrial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+",
                        help="File path(s), directory path(s), and/or glob pattern(s)")
    parser.add_argument("--reference", "-r", metavar="WAV",
                        help="Reference WAV for frequency comparison")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subfolders when an input is a directory")
    parser.add_argument("--ext", metavar="EXTS",
                        help="Comma-separated extensions to match in directories "
                             f"(default: {','.join(sorted(DEFAULT_EXTENSIONS))})")
    args = parser.parse_args()

    extensions = parse_extensions(args.ext)
    paths, warnings = resolve_inputs(args.inputs, recursive=args.recursive, extensions=extensions)
    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)
    if not paths:
        print("Error: no matching audio files found", file=sys.stderr)
        sys.exit(1)

    ref_bands = None
    if args.reference:
        try:
            yr, sr_r = load_audio(args.reference)
            ref_bands = measure_bands(yr, sr_r)
        except FileNotFoundError:
            print(f"Error: reference not found: {args.reference}", file=sys.stderr)
            sys.exit(1)

    results = []
    for path in paths:
        try:
            results.append(analyze(path, ref_bands))
        except FileNotFoundError:
            print(f"Error: not found: {path}", file=sys.stderr)
        except Exception as e:
            print(f"Error analyzing {path}: {e}", file=sys.stderr)

    if not results:
        sys.exit(1)

    if args.as_json:
        out = results if len(results) > 1 else results[0]
        print(json.dumps(_clean(out), indent=2))
        return

    if len(results) > 1:
        print_ranking(results)
        for r in results:
            name = os.path.splitext(os.path.basename(r["file"]))[0]
            print(f"{'=' * 60}")
            print(name)
            print(f"{'=' * 60}")
            print_track_report(r["scores"], r["overall"], r["blockers"], r["effort"],
                               r["raw"], ref_bands)
            print()
    else:
        r = results[0]
        print_track_report(r["scores"], r["overall"], r["blockers"], r["effort"],
                           r["raw"], ref_bands)


if __name__ == "__main__":
    main()
