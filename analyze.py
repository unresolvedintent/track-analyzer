#!/usr/bin/env python3
"""Track Analyzer — release readiness for darksynth/industrial."""

import argparse, glob, json, os, re, sys
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal

DEFAULT_EXTENSIONS = {".wav", ".aiff", ".aif", ".flac"}

DEFAULT_RUBRIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "rubric_darksynth_v2.json")

RUBRIC = None
BANDS = None
GENRE_PROFILE = None
WEIGHTS = None


def load_rubric(path=DEFAULT_RUBRIC_PATH):
    with open(path) as f:
        return json.load(f)


def apply_rubric(data):
    """Install a loaded rubric as the active configuration for measurement
    and scoring. BANDS/GENRE_PROFILE/WEIGHTS are derived views kept for
    readability at call sites; RUBRIC is the source of truth."""
    global RUBRIC, BANDS, GENRE_PROFILE, WEIGHTS
    RUBRIC = data
    BANDS = [(b["name"], b["lo"], b["hi"]) for b in data["bands"]]
    GENRE_PROFILE = {k: tuple(v) for k, v in data["genre_profile"].items()}
    WEIGHTS = data["weights"]


apply_rubric(load_rubric())

# v2 spec's eight-band layout (rubric-v2-spec.md section 2). Kept as a plain
# constant rather than rubric-driven since only measurement code uses it so
# far — no v2 scoring function exists yet to own it.
BANDS_V2 = [
    ("sub",       20,    60),
    ("bass",      60,   120),
    ("low-mid",  120,   250),
    ("mud",      250,   500),
    ("mid",      500,  2000),
    ("presence", 2000, 5000),
    ("harsh",    5000, 8000),
    ("air",      8000, 16000),
]


def _gated_frames(gate_sig, n_fft, hop, gate_db):
    """Yield start indices of successive frames over `gate_sig`, skipping
    any frame whose RMS falls below `gate_db` dBFS (pass gate_db=None to
    disable gating)."""
    n = len(gate_sig)
    for start in range(0, n - n_fft + 1, hop):
        if gate_db is None:
            yield start
            continue
        g = gate_sig[start:start + n_fft]
        rms = float(np.sqrt(np.mean(g ** 2)))
        dbfs = 20 * np.log10(rms) if rms > 0 else -np.inf
        if dbfs >= gate_db:
            yield start


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


def measure_bands_v1(y, sr):
    """v1: 6-band mean dB energy (BANDS, from the rubric). Still used by
    v1 scoring (score_frequency, score_genre) — do not change."""
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


def measure_bands(y, sr):
    """v2 spec (section 2): 8192-sample Hann STFT, 50% overlap, frames
    below -60 dBFS excluded, L/R power averaged per frame, each of the
    eight BANDS_V2 reported as a percentage of total 20 Hz-16 kHz power.
    Informational only for now — no v2 scoring function consumes this yet."""
    n_fft, hop, gate_db = 8192, 4096, -60.0
    n = y.shape[1]
    if n < n_fft:
        return {name: None for name, *_ in BANDS_V2}
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    band_masks = [(name, (freqs >= lo) & (freqs < hi)) for name, lo, hi in BANDS_V2]
    total_mask = (freqs >= 20) & (freqs < 16000)
    y_mono = np.mean(y, axis=0)
    band_power = {name: 0.0 for name, *_ in BANDS_V2}
    total_power = 0.0
    for start in _gated_frames(y_mono, n_fft, hop, gate_db):
        specL = np.fft.rfft(y[0, start:start + n_fft] * window)
        specR = np.fft.rfft(y[1, start:start + n_fft] * window)
        power = (np.abs(specL) ** 2 + np.abs(specR) ** 2) / 2.0
        for name, mask in band_masks:
            band_power[name] += float(power[mask].sum())
        total_power += float(power[total_mask].sum())
    if total_power <= 0:
        return {name: None for name, *_ in BANDS_V2}
    return {name: float(100.0 * band_power[name] / total_power) for name, *_ in BANDS_V2}


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

    # v2 spec correlation bands (section 3): 20-120/120-500/500-1k Hz,
    # 4th-order Butterworth. Additive — band_corr above is untouched so
    # score_stereo() (v1) sees identical inputs.
    v2_bands = [("sub", 20, 120), ("low-mid", 120, 500), ("mid", 500, 1000)]
    band_corr_v2 = {}
    for name, lo, hi in v2_bands:
        lo_n, hi_n = lo / nyq, min(hi / nyq, .999)
        if lo_n <= 0 or lo_n >= hi_n:
            band_corr_v2[name] = 1.0
            continue
        try:
            b, a = signal.butter(4, [lo_n, hi_n], btype="band")
            Lf, Rf = signal.filtfilt(b, a, L), signal.filtfilt(b, a, R)
            c = float(np.corrcoef(Lf, Rf)[0, 1])
            band_corr_v2[name] = c if np.isfinite(c) else 1.0
        except Exception:
            band_corr_v2[name] = 1.0

    return {"width": float(width), "mono_db": float(mono_db), "band_corr": band_corr,
            "band_corr_v2": band_corr_v2}


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

    # Clicks: onset-driven candidates confirmed via a median-absolute-deviation
    # outlier test on the sample-difference signal, robust to loud but
    # legitimate transients (kicks, snares) that the old fixed threshold
    # (diff > 1.0) risked either missing or false-positiving on.
    diff = np.abs(np.diff(y_mono))
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med)))
    robust_std = mad * 1.4826  # normal-consistent MAD scaling
    clicks = 0
    if robust_std > 0:
        onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
        onset_samples = librosa.frames_to_samples(
            librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=False))
        win = max(int(.001 * sr), 1)  # 1ms search window around each onset
        outlier_thresh = med + 12.0 * robust_std  # conservative: confirmed click, not a normal transient
        confirmed = set()
        for center in onset_samples:
            lo, hi = max(0, center - win), min(len(diff), center + win)
            if hi > lo and np.max(diff[lo:hi]) > outlier_thresh:
                confirmed.add(int(np.argmax(diff[lo:hi]) + lo))
        clicks = len(confirmed)

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


def measure_format(path):
    """Container, subtype, sample rate, bit depth, channels, duration —
    via soundfile.info(), no ffprobe/FFmpeg dependency. Not scored."""
    info = sf.info(path)
    m = re.search(r"(\d+)", info.subtype or "")
    bit_depth = int(m.group(1)) if m else None
    duration_s = float(info.frames / info.samplerate) if info.samplerate else 0.0
    return {"container": info.format, "subtype": info.subtype,
            "sample_rate": int(info.samplerate), "bit_depth": bit_depth,
            "channels": int(info.channels), "duration_s": duration_s}


def measure_boundaries(y, sr):
    """First-sample amplitude, fade-in presence (first 5ms), final 50ms
    RMS, final fade slope, trailing silence length. Not scored — boundary
    conditions become hard gates in a later step."""
    y_mono = np.mean(y, axis=0)
    n = len(y_mono)

    first_amp = float(np.abs(y_mono[0])) if n else 0.0

    n5 = max(int(.005 * sr), 1)
    window5 = np.abs(y_mono[:n5])
    peak5 = float(window5.max()) if len(window5) else 0.0
    # Fade-in heuristic: first sample sits well below the peak reached
    # within the first 5ms (a ramp), rather than starting near it.
    fade_in = bool(peak5 > 0 and first_amp < 0.1 * peak5)

    n50 = max(int(.05 * sr), 1)
    tail = y_mono[-n50:] if n >= n50 else y_mono
    final_rms = float(np.sqrt(np.mean(tail ** 2))) if len(tail) else 0.0
    final_rms_db = float(20 * np.log10(final_rms)) if final_rms > 0 else -120.0

    # Fade slope: linear trend of RMS(dB) across 5 sub-frames of the tail.
    n_sub = 5
    sub_len = max(len(tail) // n_sub, 1)
    sub_dbs = []
    for i in range(n_sub):
        seg = tail[i * sub_len:(i + 1) * sub_len]
        if len(seg) == 0:
            continue
        r = float(np.sqrt(np.mean(seg ** 2)))
        sub_dbs.append(20 * np.log10(r) if r > 0 else -120.0)
    slope = float(np.polyfit(range(len(sub_dbs)), sub_dbs, 1)[0]) if len(sub_dbs) >= 2 else 0.0

    # Trailing silence: walk backward in 10ms frames while RMS stays below -60 dBFS.
    frame = max(int(.01 * sr), 1)
    trailing_silence = 0.0
    i = n
    while i - frame >= 0:
        seg = y_mono[i - frame:i]
        r = float(np.sqrt(np.mean(seg ** 2)))
        dbfs = 20 * np.log10(r) if r > 0 else -np.inf
        if dbfs < -60.0:
            trailing_silence += frame / sr
            i -= frame
        else:
            break

    return {"first_sample_amplitude": first_amp, "fade_in_present": fade_in,
            "final_50ms_rms_db": final_rms_db, "final_fade_slope_db_per_frame": slope,
            "trailing_silence_s": float(trailing_silence)}


def measure_balance(y, sr):
    """L/R RMS imbalance in dB, plus per-band side-channel energy (percentage
    of total side power, BANDS_V2) — to catch sub content leaking into side."""
    L, R = y[0], y[1]
    rms_l = float(np.sqrt(np.mean(L ** 2)))
    rms_r = float(np.sqrt(np.mean(R ** 2)))
    lr_imbalance_db = float(20 * np.log10(rms_l / rms_r)) if rms_l > 0 and rms_r > 0 else 0.0

    side = (L - R) * 0.5
    n_fft, hop, gate_db = 8192, 4096, -60.0
    n = len(side)
    side_band_pct = {name: None for name, *_ in BANDS_V2}
    if n >= n_fft:
        y_mono = np.mean(y, axis=0)
        window = np.hanning(n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        band_masks = [(name, (freqs >= lo) & (freqs < hi)) for name, lo, hi in BANDS_V2]
        total_mask = (freqs >= 20) & (freqs < 16000)
        band_power = {name: 0.0 for name, *_ in BANDS_V2}
        total_power = 0.0
        for start in _gated_frames(y_mono, n_fft, hop, gate_db):
            power = np.abs(np.fft.rfft(side[start:start + n_fft] * window)) ** 2
            for name, mask in band_masks:
                band_power[name] += float(power[mask].sum())
            total_power += float(power[total_mask].sum())
        if total_power > 0:
            side_band_pct = {name: float(100.0 * band_power[name] / total_power)
                              for name, *_ in BANDS_V2}

    return {"lr_imbalance_db": lr_imbalance_db, "side_band_pct": side_band_pct}


def measure_integrity(y, sr, path):
    """Lossy-content spectral cliff test (energy above 16kHz vs the 8-16kHz
    mean), dither presence on 16-bit files, momentary max LUFS (400ms)."""
    y_mono = np.mean(y, axis=0)
    n = len(y_mono)
    nyq = sr / 2.0

    cliff_db, cliff_sustained = None, None
    if nyq > 16000:
        n_fft, hop, gate_db = 8192, 4096, -60.0
        if n >= n_fft:
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            mask_mid = (freqs >= 8000) & (freqs < 16000)
            mask_high = (freqs >= 16000) & (freqs < nyq)
            window = np.hanning(n_fft)
            frame_ratios = []
            for start in _gated_frames(y_mono, n_fft, hop, gate_db):
                power = np.abs(np.fft.rfft(y_mono[start:start + n_fft] * window)) ** 2
                mid_p = float(power[mask_mid].mean()) if mask_mid.any() else 0.0
                high_p = float(power[mask_high].mean()) if mask_high.any() else 0.0
                if mid_p > 0:
                    frame_ratios.append(10 * np.log10(max(high_p, 1e-20) / mid_p))
            if frame_ratios:
                cliff_db = float(np.median(frame_ratios))
                cliff_sustained = bool(np.mean([r < -40 for r in frame_ratios]) >= 0.9)

    # Dither presence on 16-bit PCM: a balanced LSB distribution suggests
    # dithered quantization; a heavily skewed one suggests truncation.
    # Coarse heuristic, not a substitute for a real TPDF/noise-shaping test.
    dither_present = None
    try:
        info = sf.info(path)
        if info.subtype and "16" in info.subtype:
            raw, _ = sf.read(path, dtype="int16", always_2d=True)
            ones_ratio = float(np.mean(raw & 1))
            dither_present = bool(0.35 < ones_ratio < 0.65)
    except Exception:
        dither_present = None

    # Momentary max LUFS: 400ms windows, 100ms hop (EBU R128-style).
    meter = pyln.Meter(sr)
    win, hop = int(.4 * sr), int(.1 * sr)
    data = y.T
    mom_vals = []
    if data.shape[0] >= win:
        for i in range(0, data.shape[0] - win, hop):
            try:
                v = float(meter.integrated_loudness(data[i:i + win]))
                if np.isfinite(v):
                    mom_vals.append(v)
            except Exception:
                pass
    momentary_max_lufs = float(max(mom_vals)) if mom_vals else float("nan")

    return {"spectral_cliff_db": cliff_db, "spectral_cliff_sustained": cliff_sustained,
            "dither_present": dither_present, "momentary_max_lufs": momentary_max_lufs}


def measure_texture(y, sr):
    """HPSS percussive/harmonic ratio, kick-band energy (post-HPSS) vs sub
    energy, and spectral flatness — descriptive genre-fit inputs, not scored."""
    y_mono = np.mean(y, axis=0)
    harmonic, percussive = librosa.effects.hpss(y_mono)

    rms_h = float(np.sqrt(np.mean(harmonic ** 2)))
    rms_p = float(np.sqrt(np.mean(percussive ** 2)))
    if rms_h > 0:
        perc_harm_ratio = float(rms_p / rms_h)
    else:
        perc_harm_ratio = float("inf") if rms_p > 0 else 0.0

    # Kick-band: percussive energy in the kick fundamental/early-harmonic
    # range (40-100Hz), vs. the track's overall sub-band energy — flags a
    # sustained sub bassline masking the kick's transient.
    n_fft = 4096
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    Sp = np.abs(librosa.stft(percussive, n_fft=n_fft, hop_length=1024)) ** 2
    kick_mask = (freqs >= 40) & (freqs < 100)
    kick_energy = float(np.mean(Sp[kick_mask, :])) if kick_mask.any() else 0.0

    Sm = np.abs(librosa.stft(y_mono, n_fft=n_fft, hop_length=1024)) ** 2
    sub_mask = (freqs >= 20) & (freqs < 60)
    sub_energy = float(np.mean(Sm[sub_mask, :])) if sub_mask.any() else 0.0
    kick_vs_sub = float(kick_energy / sub_energy) if sub_energy > 0 else None

    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_mono)))

    return {"percussive_harmonic_ratio": perc_harm_ratio, "kick_vs_sub_energy": kick_vs_sub,
            "spectral_flatness": spectral_flatness}


# ── Scoring (each returns (0-100, [(priority, actionable_msg)]) ───────────────

def score_technical(loud):
    cfg = RUBRIC["technical"]
    s, issues = 100, []
    if loud["clips"] > 0:
        s -= cfg["clip_penalty"]
        issues.append((0, f"{loud['clips']} clipped sample(s) — reduce pre-limiter gain"))
    if loud["true_peak"] > cfg["true_peak_hard_db"]:
        s -= cfg["true_peak_hard_penalty"]
        issues.append((1, f"True peak {loud['true_peak']:.1f} dBTP — hard limit to -1.0 dBTP before export"))
    elif loud["true_peak"] > cfg["true_peak_soft_db"]:
        s -= cfg["true_peak_soft_penalty"]
        issues.append((6, f"True peak {loud['true_peak']:.1f} dBTP — tighten limiter ceiling to -1.5"))
    lufs = loud["integrated"]
    if np.isfinite(lufs):
        if lufs > cfg["lufs_hot_db"]:
            s -= cfg["lufs_hot_penalty"]
            issues.append((2, f"Integrated {lufs:.1f} LUFS too hot — lower limiter threshold"))
        elif lufs < cfg["lufs_quiet_db"]:
            s -= cfg["lufs_quiet_penalty"]
            issues.append((7, f"Integrated {lufs:.1f} LUFS too quiet — check gain staging"))
        elif lufs > cfg["lufs_warn_hot_db"] or lufs < cfg["lufs_warn_quiet_db"]:
            s -= cfg["lufs_warn_penalty"]
    else:
        s -= cfg["lufs_missing_penalty"]
    if loud["dc"] > cfg["dc_hard"]:
        s -= cfg["dc_hard_penalty"]
        issues.append((3, f"DC offset {loud['dc']:.4f} — apply DC filter before export"))
    elif loud["dc"] > cfg["dc_soft"]:
        s -= cfg["dc_soft_penalty"]
    return max(0, min(100, s)), issues


def score_frequency(bands, ref_bands=None):
    cfg = RUBRIC["frequency"]
    if ref_bands is None:
        s, issues = cfg["no_reference_base"], []
        sub, bass = bands.get("sub"), bands.get("bass")
        if sub is not None and bass is not None:
            if sub < bass - cfg["sub_vs_bass_gap_db"]:
                s -= cfg["sub_vs_bass_penalty"]
                issues.append((9, f"Sub very weak vs bass ({sub-bass:+.0f}dB) — check low-end content"))
        hm = bands.get("high-mid")
        if hm is not None and bass is not None and hm > bass - cfg["highmid_vs_bass_gap_db"]:
            s -= cfg["highmid_vs_bass_penalty"]
            issues.append((8, f"High-mid {hm-bass:+.0f}dB vs bass — may sound harsh/thin"))
        return max(0, min(100, s)), issues

    s, issues, deltas = cfg["reference_base"], [], {}
    for name, *_ in BANDS:
        t, r = bands.get(name), ref_bands.get(name)
        if t is None or r is None:
            continue
        d = t - r
        deltas[name] = d
        abd = abs(d)
        for th in cfg["delta_thresholds"]:
            if abd > th["gt_db"]:
                s -= th["penalty"]
                break

    hz_hint = {"sub": "40Hz shelf", "bass": "100Hz", "low-mid": "350Hz",
               "high-mid": "2kHz", "presence": "8kHz", "air": "14kHz shelf"}
    for name, d in sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:2]:
        if abs(d) > cfg["issue_threshold_db"]:
            action = "cut" if d > 0 else "boost"
            issues.append((4, f"{name} {d:+.1f}dB vs reference — {action} {hz_hint.get(name, name)}"))
    return max(0, min(100, s)), issues


def score_stereo(stereo, phase):
    cfg = RUBRIC["stereo"]
    s, issues = 100, []
    if phase["sustained_neg"]:
        s -= cfg["sustained_neg_penalty"]
        worst = min(stereo["band_corr"], key=stereo["band_corr"].get)
        issues.append((1, f"Phase correlation negative (min {phase['min']:.2f}) "
                          f"— check stereo widener, worst in {worst}"))
    sub_c = stereo["band_corr"].get("sub", 1.0)
    if sub_c < cfg["sub_corr_hard"]:
        s -= cfg["sub_corr_hard_penalty"]
        issues.append((2, f"Sub L/R correlation {sub_c:.2f} — collapse sub below 80Hz to mono"))
    elif sub_c < cfg["sub_corr_soft"]:
        s -= cfg["sub_corr_soft_penalty"]
        issues.append((6, f"Sub L/R correlation {sub_c:.2f} — consider mono filter at 80Hz"))
    if stereo["mono_db"] < cfg["mono_hard_db"]:
        s -= cfg["mono_hard_penalty"]
        issues.append((3, f"Mono compat {stereo['mono_db']:.1f}dB loss — anti-phase content present"))
    elif stereo["mono_db"] < cfg["mono_soft_db"]:
        s -= cfg["mono_soft_penalty"]
        issues.append((7, f"Mono compat {stereo['mono_db']:.1f}dB loss — verify mono playback"))
    if stereo["width"] > cfg["width_max"]:
        s -= cfg["width_penalty"]
        issues.append((8, f"Stereo width {stereo['width']:.2f} M/S ratio — may fold oddly on some systems"))
    return max(0, min(100, s)), issues


def score_dynamics(dyn, loud):
    cfg = RUBRIC["dynamics"]
    s, issues = 100, []
    lra = loud["lra"]
    if np.isfinite(lra):
        if lra < cfg["lra_severe"]:
            s -= cfg["lra_severe_penalty"]
            issues.append((3, f"LRA {lra:.1f} LU — severely over-compressed, ease limiter by 4+ dB"))
        elif lra < cfg["lra_over"]:
            s -= cfg["lra_over_penalty"]
            issues.append((5, f"LRA {lra:.1f} LU — over-compressed, ease limiter threshold"))
        elif lra > cfg["lra_loose"]:
            s -= cfg["lra_loose_penalty"]
            issues.append((7, f"LRA {lra:.1f} LU — very dynamic, may need limiting for streaming"))
    if dyn["crest"] < cfg["crest_hard"]:
        s -= cfg["crest_hard_penalty"]
        issues.append((4, f"Crest factor {dyn['crest']:.1f}dB — heavily limited, check limiter settings"))
    elif dyn["crest"] < cfg["crest_soft"]:
        s -= cfg["crest_soft_penalty"]
    if dyn["psr"] < cfg["psr_min"]:
        s -= cfg["psr_penalty"]
        issues.append((6, f"PSR {dyn['psr']:.1f}dB — over-limited, increase peak-to-loudness margin"))
    return max(0, min(100, s)), issues


def score_artifacts(arts):
    cfg = RUBRIC["artifacts"]
    s, issues = 100, []
    if arts["clicks"] > 0:
        s -= min(cfg["click_penalty_cap"], arts["clicks"] * cfg["click_penalty_per"])
        issues.append((2, f"{arts['clicks']} click(s) detected — check edit points and clip limiting"))
    nf = arts["noise_floor"]
    if nf is not None:
        if nf > cfg["noise_hard_db"]:
            s -= cfg["noise_hard_penalty"]
            issues.append((4, f"Noise floor {nf:.0f}dB — check source recordings for hum/hiss"))
        elif nf > cfg["noise_soft_db"]:
            s -= cfg["noise_soft_penalty"]
            issues.append((8, f"Noise floor {nf:.0f}dB — mild background noise"))
    if arts["silence_gaps"] > 0:
        s -= min(cfg["gap_penalty_cap"], arts["silence_gaps"] * cfg["gap_penalty_per"])
        issues.append((5, f"{arts['silence_gaps']} internal silence gap(s) — dead spots in arrangement"))
    if arts["zcr_spikes"] > cfg["zcr_spike_threshold"]:
        s -= cfg["zcr_spike_penalty"]
        issues.append((7, f"{arts['zcr_spikes']} zero-crossing anomalies — possible edit glitches"))
    return max(0, min(100, s)), issues


def score_genre(bands):
    cfg = RUBRIC["genre"]
    bass_val = bands.get("bass")
    if bass_val is None:
        return cfg["fallback_score"], []
    s, issues, deviations = 100, [], []
    for name, (lo, hi) in GENRE_PROFILE.items():
        if name == "bass":
            continue
        val = bands.get(name)
        if val is None:
            continue
        actual_delta = val - bass_val
        center = (lo + hi) / 2
        tolerance = (hi - lo) / 2 + cfg["tolerance_margin_db"]
        overshoot = max(0.0, abs(actual_delta - center) - tolerance)
        if overshoot > cfg["overshoot_hard"]:
            s -= cfg["overshoot_hard_penalty"]
            deviations.append((overshoot, name, actual_delta, center))
        elif overshoot > 0:
            s -= cfg["overshoot_soft_penalty"]
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
    cfg = RUBRIC["fix_effort"]
    if min(scores.values()) < cfg["high_min_score"] or ov < cfg["high_overall"]:
        return "High"
    if min(scores.values()) >= cfg["low_min_score"] and ov >= cfg["low_overall"]:
        return "Low"
    return "Medium"


def verdict(ov):
    cfg = RUBRIC["verdict"]
    return ("Ready for release" if ov > cfg["ready_over"]
            else ("Needs work" if ov >= cfg["needs_work_at_least"] else "Significant issues"))


def status_of(s):
    cfg = RUBRIC["status"]
    return "PASS" if s >= cfg["pass_at_least"] else ("WARNING" if s >= cfg["warning_at_least"] else "FAIL")


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

    def _na(v, fmt="{:.2f}"):
        return fmt.format(v) if v is not None else "n/a"

    if "format" in raw:
        fmt = raw["format"]
        print(f"  Format      {fmt['container']}/{fmt['subtype']}   {fmt['sample_rate']}Hz   "
              f"{_na(fmt['bit_depth'], '{:d}')}-bit   {fmt['channels']}ch   {fmt['duration_s']:.1f}s")
    if "boundaries" in raw:
        b = raw["boundaries"]
        fade = "  [fade-in]" if b["fade_in_present"] else ""
        print(f"  Boundaries  start amp {b['first_sample_amplitude']:.3f}{fade}   "
              f"end {b['final_50ms_rms_db']:.1f}dB   slope {b['final_fade_slope_db_per_frame']:+.2f}dB/frame   "
              f"trail silence {b['trailing_silence_s']:.2f}s")
    if "balance" in raw:
        bal = raw["balance"]
        side_row = "   ".join(f"{n} {_na(bal['side_band_pct'].get(n), '{:.1f}')}%"
                               for n, *_ in BANDS_V2)
        print(f"  Balance     L/R {bal['lr_imbalance_db']:+.2f}dB")
        print(f"  Side (%)    {side_row}")
    if "integrity" in raw:
        ig = raw["integrity"]
        cliff = _na(ig["spectral_cliff_db"], "{:.1f}") + "dB"
        cliff += "  [sustained]" if ig["spectral_cliff_sustained"] else ""
        print(f"  Integrity   16k+ cliff {cliff}   dither {ig['dither_present']}   "
              f"momentary max {_na(ig['momentary_max_lufs'], '{:.1f}')} LUFS")
    if "texture" in raw:
        tx = raw["texture"]
        print(f"  Texture     perc/harm {tx['percussive_harmonic_ratio']:.2f}   "
              f"kick/sub {_na(tx['kick_vs_sub_energy'])}   flatness {tx['spectral_flatness']:.3f}")
    if "bands_v2" in raw:
        bv2 = raw["bands_v2"]
        row = "   ".join(f"{n} {_na(bv2.get(n), '{:.1f}')}%" for n, *_ in BANDS_V2)
        print(f"  Freq v2 (%) {row}")
    if "band_corr_v2" in stereo:
        cv2 = stereo["band_corr_v2"]
        print(f"  Corr v2     sub(20-120) {cv2['sub']:.2f}   low-mid(120-500) {cv2['low-mid']:.2f}   "
              f"mid(500-1k) {cv2['mid']:.2f}")


def print_track_report(scores, ov, blockers, effort, raw, ref_bands=None, rubric_version=None):
    rv = f"   [rubric {rubric_version}]" if rubric_version else ""
    print(f"VERDICT: {verdict(ov)} — {ov:.0f}%{rv}")
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
    bands = measure_bands_v1(y, sr)
    bands_v2 = measure_bands(y, sr)
    stereo = measure_stereo(y, sr)
    phase = measure_phase(y, sr)
    dyn = measure_dynamics(y, sr, loud)
    arts = measure_artifacts(y, sr)
    fmt = measure_format(path)
    boundaries = measure_boundaries(y, sr)
    balance = measure_balance(y, sr)
    integrity = measure_integrity(y, sr, path)
    texture = measure_texture(y, sr)

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
            "blockers": blockers, "effort": effort, "rubric_version": RUBRIC["version"],
            "raw": {"loudness": loud, "bands": bands, "stereo": stereo,
                    "phase": phase, "dynamics": dyn, "artifacts": arts,
                    "bands_v2": bands_v2, "format": fmt, "boundaries": boundaries,
                    "balance": balance, "integrity": integrity, "texture": texture}}


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
    parser.add_argument("--rubric", metavar="JSON", default=DEFAULT_RUBRIC_PATH,
                        help=f"Rubric config to score against (default: {DEFAULT_RUBRIC_PATH})")
    args = parser.parse_args()

    try:
        apply_rubric(load_rubric(args.rubric))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: failed to load rubric {args.rubric}: {e}", file=sys.stderr)
        sys.exit(1)

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
            ref_bands = measure_bands_v1(yr, sr_r)
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
                               r["raw"], ref_bands, r["rubric_version"])
            print()
    else:
        r = results[0]
        print_track_report(r["scores"], r["overall"], r["blockers"], r["effort"],
                           r["raw"], ref_bands, r["rubric_version"])


if __name__ == "__main__":
    main()
