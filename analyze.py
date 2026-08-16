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


def _lerp_score(x, x100, x0):
    """Linear 0-100 score: `x100` maps to 100, `x0` maps to 0, clamped
    outside that range. Works for both directions (x100 > x0 or x100 < x0)."""
    if x100 == x0:
        return 100.0 if x == x100 else 0.0
    t = (x - x100) / (x0 - x100)
    t = max(0.0, min(1.0, t))
    return 100.0 * (1.0 - t)


def _interp_curve(x, points):
    """Piecewise-linear interpolation through sorted (x, y) points, clamped
    to the first/last y value outside the covered x range."""
    if x <= points[0][0]:
        return float(points[0][1])
    if x >= points[-1][0]:
        return float(points[-1][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return float(y0 + t * (y1 - y0))
    return float(points[-1][1])


def _has_confirmed_clip_run(y, min_run, thresh):
    """True if any sample position (any channel) sustains >=min_run
    consecutive samples at abs(x) >= thresh."""
    over = np.any(np.abs(y) >= thresh, axis=0).astype(np.int8)
    if not over.any():
        return False
    diffs = np.diff(np.concatenate(([0], over, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return False
    return bool(np.max(ends - starts) >= min_run)


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
    # True-peak oversampling factor per spec section 1: 4x for sr<=48kHz,
    # 2x for sr>=88.2kHz. Sample rates strictly between those (rare) fall
    # back to the safer 4x.
    tcfg = RUBRIC["technical"]
    if sr >= tcfg["tp_oversample_high_sr_min"]:
        os_factor = tcfg["tp_oversample_high_factor"]
    else:
        os_factor = tcfg["tp_oversample_low_factor"]
    tp = 0.0
    for ch in range(y.shape[0]):
        over = signal.resample_poly(y[ch], os_factor, 1)
        tp = max(tp, float(np.max(np.abs(over))))
    tp_dbtp = 20.0 * np.log10(tp) if tp > 0 else -120.0
    clips = int(np.sum(np.any(np.abs(y) >= 0.9999, axis=0)))
    confirmed_clip = _has_confirmed_clip_run(y, tcfg["clip_min_run"], tcfg["clip_confirmed_thresh"])
    dc = float(np.max(np.abs([np.mean(y[ch]) for ch in range(y.shape[0])])))
    return {"integrated": integrated, "short_term_max": float(st_max),
            "confirmed_clip": confirmed_clip,
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
        click_positions = sorted(confirmed)
    else:
        click_positions = []

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

    return {"clicks": clicks, "click_positions": click_positions, "noise_floor": noise_floor,
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

def score_technical(loud, fmt, integrity, path, final=False):
    """T = 0.40*TP + 0.25*CLIP + 0.15*LUFS + 0.10*DC + 0.10*FORMAT (spec section 1).
    Issues are (priority, warning_code_or_None, message). A None code means
    the condition is already accounted for by a hard gate (evaluate_gates())
    with an identical trigger, so compute_effort() must not double-count it."""
    cfg = RUBRIC["technical"]
    w = cfg["component_weights"]
    issues = []

    tp_db = loud["true_peak"]
    if tp_db <= cfg["tp_full_db"]:
        tp_score = 100.0
    else:
        tp_score = max(0.0, min(100.0, 100.0 - cfg["tp_slope_per_db"] * (tp_db - cfg["tp_full_db"])))
        # No code: identical trigger to the true_peak_exceeded gate.
        issues.append((1, None, f"True peak {tp_db:.1f} dBTP — hard limit to {cfg['tp_full_db']:.1f} dBTP before export"))

    if loud["confirmed_clip"]:
        clip_score = 0.0
        # No code: identical trigger to the confirmed_clipping gate.
        issues.append((0, None, f"Confirmed clipping ({loud['clips']} sample(s) at full scale) — reduce pre-limiter gain"))
    else:
        clip_score = 100.0

    lufs = loud["integrated"]
    if np.isfinite(lufs):
        raw = 100.0 - cfg["lufs_slope_per_lu"] * abs(lufs - cfg["lufs_target"])
        lufs_score = max(cfg["lufs_floor"], min(100.0, raw))
        if lufs_score < 100.0:
            issues.append((2, "lufs_off_target",
                            f"Integrated {lufs:.1f} LUFS vs {cfg['lufs_target']} target — adjust gain staging"))
    else:
        lufs_score = float(cfg["lufs_floor"])
        issues.append((2, "lufs_off_target", "Integrated loudness could not be measured"))

    dc = loud["dc"]
    if dc <= cfg["dc_full"]:
        dc_score = 100.0
    else:
        slope = (cfg["dc_mid_score"] - 100.0) / (cfg["dc_mid"] - cfg["dc_full"])
        dc_score = max(cfg["dc_floor"], 100.0 + slope * (dc - cfg["dc_full"]))
        issues.append((3, "dc_offset", f"DC offset {dc:.4f} — apply DC filter before export"))

    container = (fmt["container"] or "").upper()
    hidden_lossy = bool(integrity.get("spectral_cliff_sustained"))
    if hidden_lossy:
        format_score = float(cfg["format_hidden_lossy_score"])
        # No code when unconditional: identical trigger to the
        # lossy_content_detected gate (fires regardless of --final).
        issues.append((5, None, "Lossy encoding detected inside container "
                                 "(spectral cliff above 16kHz) — re-export from an uncompressed master"))
    elif container in ("WAV", "AIFF", "FLAC"):
        format_score = float(cfg["format_lossless_score"])
    else:
        try:
            size_bits = os.path.getsize(path) * 8
            kbps = (size_bits / fmt["duration_s"] / 1000.0) if fmt["duration_s"] > 0 else 0.0
        except OSError:
            kbps = 0.0
        if kbps >= cfg["format_mp3_bitrate_kbps"]:
            format_score = float(cfg["format_mp3_high_score"])
        else:
            format_score = float(cfg["format_mp3_low_score"])
        # No code when --final also gates this exact condition (lossy_container_final).
        code = None if final else "format_lossy"
        issues.append((5, code, f"Lossy container ({fmt['container']}, ~{kbps:.0f}kbps) — re-export from a lossless master"))

    t_score = (w["tp"] * tp_score + w["clip"] * clip_score + w["lufs"] * lufs_score +
               w["dc"] * dc_score + w["format"] * format_score)
    return max(0.0, min(100.0, t_score)), issues


def score_frequency(bands_v2):
    """F = weighted per-band dB-deviation score vs the stored calibration
    profile (spec section 2). Returns (score_or_None, issues); score is
    None (PROVISIONAL) until calibrate.py populates
    rubric['calibration']['frequency_profile']. --reference comparisons
    are reported separately by score_reference_delta() and never feed
    into this score (spec: "does not replace the rubric score or alter
    ranking")."""
    cfg = RUBRIC["frequency"]
    profile = RUBRIC["calibration"]["frequency_profile"]
    if not profile:
        return None, [(9, None, "Frequency: PROVISIONAL — no calibration profile yet (run calibrate.py)")]

    w = cfg["component_weights"]
    curve = cfg["deviation_curve"]
    deltas, total = {}, 0.0
    for name, weight in w.items():
        measured, target = bands_v2.get(name), profile.get(name)
        if measured is None or not target:
            band_score = 100.0  # nothing to compare; don't penalize for missing data
        else:
            dev_db = float(10 * np.log10(max(measured, 1e-9) / max(target, 1e-9)))
            deltas[name] = dev_db
            band_score = _interp_curve(abs(dev_db), curve)
        total += weight * band_score

    issues = []
    hz_hint = {"sub": "20-60Hz", "bass": "60-120Hz", "low-mid": "120-250Hz", "mud": "250-500Hz",
               "mid": "500Hz-2kHz", "presence": "2-5kHz", "harsh": "5-8kHz", "air": "8-16kHz"}
    for name, d in sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:2]:
        if abs(d) > 3:
            action = "cut" if d > 0 else "boost"
            issues.append((4, "frequency_band_deviation",
                            f"{name} {d:+.1f}dB vs calibration — {action} {hz_hint.get(name, name)}"))

    return max(0.0, min(100.0, total)), issues


def score_reference_delta(bands_v1, ref_bands_v1):
    """Informational only (spec section 2): v1-style per-band dB delta vs
    a user-supplied --reference track. Reported separately; never feeds
    score_frequency(), OVERALL, or ranking."""
    cfg = RUBRIC["frequency_reference"]
    s, issues, deltas = cfg["base"], [], {}
    for name, *_ in BANDS:
        t, r = bands_v1.get(name), ref_bands_v1.get(name)
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
    return max(0, min(100, s)), deltas, issues


def score_stereo(stereo, balance):
    """S = 0.30*SUB + 0.25*MONO + 0.20*LOWMID + 0.15*WIDTH + 0.10*BALANCE
    (spec section 3). Sustained negative correlation is a hard gate only
    (evaluate_gates()), not part of this formula."""
    cfg = RUBRIC["stereo"]
    w = cfg["component_weights"]
    v2 = stereo["band_corr_v2"]
    issues = []

    sub_c = v2["sub"]
    sub_score = _lerp_score(sub_c, cfg["sub_corr_full"], cfg["sub_corr_zero"])
    if sub_score < 100.0:
        issues.append((2, "stereo_sub_corr", f"Sub correlation (20-120Hz) {sub_c:.2f} — collapse sub to mono"))

    lm_c, mid_c = v2["low-mid"], v2["mid"]
    lm_score = _lerp_score(lm_c, cfg["lowmid_corr_full"], cfg["lowmid_corr_zero"])
    mid_score = _lerp_score(mid_c, cfg["lowmid_corr_full"], cfg["lowmid_corr_zero"])
    lowmid_score = (lm_score + mid_score) / 2.0
    if lowmid_score < 100.0:
        issues.append((6, "stereo_lowmid_corr",
                        f"Low-mid/mid correlation {lm_c:.2f}/{mid_c:.2f} — narrow width in that range"))

    mono_db = stereo["mono_db"]
    mono_score = _lerp_score(mono_db, cfg["mono_full_db"], cfg["mono_zero_db"])
    if mono_score < 100.0:
        # No code once the mono_sum_loss gate also fires (same trigger).
        code = "stereo_mono_loss" if mono_db > cfg["mono_zero_db"] else None
        issues.append((3, code, f"Mono compat {mono_db:.1f}dB loss — check phase alignment"))

    width = stereo["width"]
    width_score = _lerp_score(width, cfg["width_full"], cfg["width_zero"])
    if width_score < 100.0:
        issues.append((8, "stereo_width", f"Stereo width {width:.2f} M/S ratio — ease off the widener"))

    imbalance = abs(balance["lr_imbalance_db"])
    balance_score = _lerp_score(imbalance, cfg["balance_full_db"], cfg["balance_zero_db"])
    if balance_score < 100.0:
        issues.append((7, "stereo_balance", f"L/R imbalance {imbalance:.2f}dB — check pan/gain balance"))

    s_score = (w["sub"] * sub_score + w["mono"] * mono_score + w["lowmid"] * lowmid_score +
               w["width"] * width_score + w["balance"] * balance_score)
    return max(0.0, min(100.0, s_score)), issues


def score_dynamics(dyn, loud):
    """D = 0.60*LRA + 0.25*PSR + 0.15*CREST (spec section 4). LRA is
    asymmetric: under-range is penalized harder than over-range, since
    over-compression is harder to fix and more common in this genre."""
    cfg = RUBRIC["dynamics"]
    w = cfg["component_weights"]
    issues = []

    lra = loud["lra"]
    if np.isfinite(lra):
        lo, hi = cfg["lra_full_lo"], cfg["lra_full_hi"]
        if lo <= lra <= hi:
            lra_score = 100.0
        elif lra < lo:
            lra_score = max(0.0, 100.0 - cfg["lra_below_slope"] * (lo - lra))
            issues.append((3, "dynamics_lra", f"LRA {lra:.1f} LU — over-compressed, ease limiter threshold"))
        else:
            lra_score = max(0.0, 100.0 - cfg["lra_above_slope"] * (lra - hi))
            issues.append((7, "dynamics_lra", f"LRA {lra:.1f} LU — very dynamic, may need limiting for streaming"))
    else:
        lra_score = 0.0
        issues.append((3, "dynamics_lra", "LRA could not be measured"))

    psr = dyn["psr"]
    if psr >= cfg["psr_full_db"]:
        psr_score = 100.0
    else:
        psr_score = max(0.0, 100.0 - cfg["psr_slope_per_db"] * (cfg["psr_full_db"] - psr))
        issues.append((6, "dynamics_psr", f"PSR {psr:.1f}dB — over-limited, increase peak-to-loudness margin"))

    crest = dyn["crest"]
    if crest >= cfg["crest_full_db"]:
        crest_score = 100.0
    else:
        crest_score = max(0.0, 100.0 - cfg["crest_slope_per_db"] * (cfg["crest_full_db"] - crest))
        issues.append((4, "dynamics_crest", f"Crest factor {crest:.1f}dB — heavily limited, check limiter settings"))

    d_score = w["lra"] * lra_score + w["psr"] * psr_score + w["crest"] * crest_score
    return max(0.0, min(100.0, d_score)), issues


def score_artifacts(arts):
    """A = 0.70*CLICKS + 0.30*NOISE (spec section 5); renormalizes to
    A = CLICKS when noise floor is unmeasurable. Start/end boundaries and
    silence gaps are reported as data only (measure_boundaries()/arts) —
    they moved to hard gates and are never scored here."""
    cfg = RUBRIC["artifacts"]
    w = cfg["component_weights"]
    issues = []

    clicks = arts["clicks"]
    clicks_score = max(0.0, 100.0 - cfg["click_penalty_per"] * clicks)
    if clicks > 0:
        issues.append((2, "artifacts_clicks", f"{clicks} click(s) detected — check edit points and clip limiting"))

    nf = arts["noise_floor"]
    if nf is None:
        a_score = clicks_score
    else:
        if nf <= cfg["noise_full_db"]:
            noise_score = 100.0
        elif nf <= cfg["noise_mid_db"]:
            noise_score = float(cfg["noise_mid_score"])
            issues.append((8, "artifacts_noise", f"Noise floor {nf:.0f}dB — mild background noise"))
        else:
            noise_score = float(cfg["noise_high_score"])
            issues.append((4, "artifacts_noise", f"Noise floor {nf:.0f}dB — check source recordings for hum/hiss"))
        a_score = w["clicks"] * clicks_score + w["noise"] * noise_score

    return max(0.0, min(100.0, a_score)), issues


def score_genre(bands):
    """Descriptive genre fit vs the darksynth/industrial profile (spec
    section 9 scopes a fuller HPSS-based qualitative report; this keeps
    the existing v1-band-delta scoring as that numeric descriptive proxy).
    Never included in OVERALL — reported separately (see analyze())."""
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
        issues.append((9, None, f"{worst_name} {tag} for darksynth profile "
                                 f"({actual_delta:+.0f}dB vs bass, target {target:+.0f}dB)"))
    return max(0, min(100, s)), issues


# ── Aggregation ───────────────────────────────────────────────────────────────

def weighted_overall(scores):
    """OVERALL = 0.25*T + 0.24*F + 0.23*S + 0.18*D + 0.10*A (spec intro).
    Genre is never included (weight 0, excluded from WEIGHTS entirely).
    When frequency is None (uncalibrated, PROVISIONAL), its 0.24 weight is
    redistributed proportionally across the other scored categories."""
    freq = scores.get("frequency")
    if freq is None:
        remaining = {k: v for k, v in WEIGHTS.items() if k != "frequency"}
        total_w = sum(remaining.values())
        return sum(scores[k] * w for k, w in remaining.items()) / total_w
    return sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)


def top_blockers(all_issues, n=3):
    seen, out = set(), []
    for item in sorted(all_issues, key=lambda x: (x[0], x[2])):
        msg = item[2]
        if msg not in seen:
            out.append(msg)
            seen.add(msg)
        if len(out) == n:
            break
    return out


def verdict(ov, category_scores, gates):
    """Spec section 7. Gates can override the verdict regardless of OVERALL,
    so severity is checked most-severe-first. 'Easy'/'reconstruction-level'
    gates are derived from the same effort cost table (cost 1 / cost 3)
    rather than a separate hardcoded list, so there's one source of truth."""
    cfg = RUBRIC["verdict"]
    gate_costs = RUBRIC["effort"]["gate_costs"]
    easy_gates = {g for g, c in gate_costs.items() if c == 1}
    reconstruction_gates = {g for g, c in gate_costs.items() if c == 3}
    n_gates = len(gates)
    scored = {k: v for k, v in category_scores.items() if k in WEIGHTS and v is not None}

    if ov < cfg["significant_overall_max"] or n_gates >= 2 or any(g in reconstruction_gates for g in gates):
        return "Significant work"
    if ov >= cfg["ready_overall_min"] and all(v >= cfg["ready_category_min"] for v in scored.values()) and n_gates == 0:
        return "Ready"
    if n_gates == 1 and gates[0] in easy_gates:
        return "Minor work"
    if cfg["minor_overall_min"] <= ov <= cfg["minor_overall_max"]:
        return "Minor work"
    if cfg["needs_overall_min"] <= ov <= cfg["needs_overall_max"]:
        return "Needs work"
    return "Needs work"


def status_of(s):
    if s is None:
        return "N/A"
    cfg = RUBRIC["status"]
    return "PASS" if s >= cfg["pass_at_least"] else ("WARNING" if s >= cfg["warning_at_least"] else "FAIL")


def evaluate_gates(loud, phase, stereo, boundaries, integrity, fmt, arts, sr, n_samples, final=False):
    """Binary pass/fail hard gates, evaluated separately from the numeric
    score (spec section 6). Returns a list of triggered gate names — any
    non-empty list can override the verdict regardless of OVERALL."""
    cfg = RUBRIC["gates"]
    gates = []

    if loud["confirmed_clip"]:
        gates.append("confirmed_clipping")
    if loud["true_peak"] > cfg["true_peak_hard_db"]:
        gates.append("true_peak_exceeded")
    if phase["sustained_neg"]:
        gates.append("sustained_negative_correlation")
    if stereo["mono_db"] < cfg["mono_sum_loss_db"]:
        gates.append("mono_sum_loss")

    window = int(cfg["boundary_click_window_ms"] / 1000.0 * sr)
    if any(p < window or p > n_samples - window for p in arts["click_positions"]):
        gates.append("boundary_click")

    if (boundaries["first_sample_amplitude"] > cfg["hard_start_amp"]
            and not boundaries["fade_in_present"]):
        gates.append("hard_start")
    if (boundaries["final_50ms_rms_db"] > cfg["hard_end_rms_db"]
            and boundaries["final_fade_slope_db_per_frame"] >= 0):
        gates.append("hard_end")

    lossless = {"WAV", "AIFF", "FLAC"}
    if final and (fmt["container"] or "").upper() not in lossless:
        gates.append("lossy_container_final")
    if integrity.get("spectral_cliff_sustained"):
        gates.append("lossy_content_detected")
    if final and fmt["sample_rate"] not in cfg["final_sample_rates"]:
        gates.append("sample_rate_final")

    return gates


def compute_effort(gates, warning_codes):
    """effort_points as a pure table lookup from triggered gate names and
    non-gate warning codes (spec section 8) — no heuristics, no score
    thresholds. Codes not in either table (or None, meaning "already
    covered by a gate") contribute zero, by design."""
    cfg = RUBRIC["effort"]
    gate_costs, warning_costs = cfg["gate_costs"], cfg["warning_costs"]
    points = sum(gate_costs.get(g, 0) for g in gates)
    points += sum(warning_costs.get(c, 0) for c in warning_codes if c)
    if points <= cfg["low_max"]:
        bucket = "Low"
    elif points <= cfg["medium_max"]:
        bucket = "Medium"
    else:
        bucket = "High"
    return points, bucket


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


def print_measured_data(raw, reference_delta=None):
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
    if reference_delta:
        deltas = reference_delta["deltas"]
        def _delta(n):
            d = deltas.get(n)
            return f"{d:+.1f}" if d is not None else "n/a"
        print(f"  vs ref      {band_row(_delta)}   (informational only — score {reference_delta['score']:.0f}, "
              f"not part of OVERALL)")
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


def print_track_report(r):
    rv = f"   [rubric {r['rubric_version']}]" if r.get("rubric_version") else ""
    print(f"VERDICT: {r['verdict']} — {r['overall']:.0f}%{rv}")
    print()
    print(f"{'Category':<20} | {'Score':>5} | Status")
    print(f"{'-'*20}-+-{'-'*5}-+-{'-'*7}")
    for label, key in CATS:
        sc = r["scores"][key]
        sc_str = f"{sc:.0f}" if sc is not None else "N/A"
        print(f"{label:<20} | {sc_str:>5} | {status_of(sc)}")
    print()
    print_measured_data(r["raw"], r.get("reference_delta"))
    print()
    print("TOP BLOCKERS")
    if r["blockers"]:
        for b in r["blockers"]:
            print(f"  {b}")
    else:
        print("  None")
    print()
    print("HARD GATES")
    if r["gates"]:
        for g in r["gates"]:
            print(f"  {g}")
    else:
        print("  None")
    print()
    if r.get("genre_issues"):
        print("GENRE FIT (descriptive only, not in OVERALL)")
        for msg in r["genre_issues"]:
            print(f"  {msg}")
        print()
    print(f"FIX EFFORT: {r['effort']} ({r['effort_points']} point(s))")


def print_ranking(results):
    cols = (28, 6, 20, 40, 8)
    header = f"{'Track':<{cols[0]}} | {'Score':>{cols[1]}} | {'Verdict':<{cols[2]}} | {'Main blocker':<{cols[3]}} | Effort"
    sep = "-+-".join("-" * c for c in cols)
    print("RANKING")
    print(header)
    print(sep)
    # Ranking key: (effort_points ascending, OVERALL descending) — spec section 8.
    for r in sorted(results, key=lambda x: (x["effort_points"], -x["overall"])):
        name = os.path.splitext(os.path.basename(r["file"]))[0]
        name = name[:cols[0]-1] if len(name) >= cols[0] else name
        blocker = (r["blockers"][0][:cols[3]-1] if r["blockers"] else "—")
        verd = r["verdict"][:cols[2]-1]
        effort_str = f"{r['effort']} ({r['effort_points']})"
        print(f"{name:<{cols[0]}} | {r['overall']:>{cols[1]-1}.0f}% | "
              f"{verd:<{cols[2]}} | {blocker:<{cols[3]}} | {effort_str}")
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

def analyze(path, ref_bands=None, final=False):
    y, sr = load_audio(path)
    loud = measure_loudness(y, sr)
    bands_v1 = measure_bands_v1(y, sr)
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

    t_s, t_i = score_technical(loud, fmt, integrity, path, final=final)
    f_s, f_i = score_frequency(bands_v2)
    st_s, st_i = score_stereo(stereo, balance)
    d_s, d_i = score_dynamics(dyn, loud)
    a_s, a_i = score_artifacts(arts)
    g_s, g_i = score_genre(bands_v1)

    scores = {"technical": t_s, "frequency": f_s, "stereo": st_s,
              "dynamics": d_s, "artifacts": a_s, "genre": g_s}
    ov = weighted_overall(scores)

    n_samples = y.shape[1]
    gates = evaluate_gates(loud, phase, stereo, boundaries, integrity, fmt, arts, sr, n_samples, final=final)

    all_issues = t_i + f_i + st_i + d_i + a_i
    blockers = top_blockers(all_issues)
    warning_codes = [code for _, code, _ in all_issues]
    effort_points, effort_bucket = compute_effort(gates, warning_codes)
    v = verdict(ov, scores, gates)

    reference_delta = None
    if ref_bands is not None:
        rd_score, rd_deltas, rd_issues = score_reference_delta(bands_v1, ref_bands)
        reference_delta = {"score": rd_score, "deltas": rd_deltas,
                            "issues": [msg for _, msg in rd_issues]}

    return {"file": path, "scores": scores, "overall": ov, "verdict": v,
            "blockers": blockers, "gates": gates,
            "effort_points": effort_points, "effort": effort_bucket,
            "rubric_version": RUBRIC["version"],
            "genre_issues": [msg for _, _, msg in g_i],
            "reference_delta": reference_delta,
            "raw": {"loudness": loud, "bands": bands_v1, "stereo": stereo,
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
    parser.add_argument("--final", action="store_true",
                        help="Also enforce release-format gates: lossless container, "
                             "44.1/48kHz sample rate")
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
            results.append(analyze(path, ref_bands, final=args.final))
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
            print_track_report(r)
            print()
    else:
        print_track_report(results[0])


if __name__ == "__main__":
    main()
