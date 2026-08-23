#!/usr/bin/env python3
"""Measurement: audio loading and all measure_* functions. Pure signal
analysis — no scoring logic here.

Uses `import score` (not `from score import RUBRIC`) to read RUBRIC/BANDS,
since apply_rubric() (the --rubric flag) rebinds those globals at runtime;
a `from` import would freeze a stale reference taken at import time.
"""

import re
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal

import score

# v2 spec's eight-band layout (rubric-v2-spec.md section 2). Kept as a plain
# constant rather than rubric-driven since only measurement code uses it so
# far — no v2 scoring function consumes raw band edges directly.
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
    tcfg = score.RUBRIC["technical"]
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
    """v1: 6-band mean dB energy (score.BANDS, from the rubric). Still used
    by v1 scoring (score_frequency's --reference path, score_genre) — do
    not change."""
    y_mono = np.mean(y, axis=0)
    n_fft = 4096
    S = np.abs(librosa.stft(y_mono, n_fft=n_fft, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = S ** 2
    out = {}
    for name, lo, hi in score.BANDS:
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
    eight BANDS_V2 reported as a percentage of total 20 Hz-16 kHz power."""
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
    for name, lo, hi in score.BANDS:
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
    # any v1-style consumer sees identical inputs.
    #
    # Uses second-order-sections (sos) form + sosfiltfilt rather than the
    # transfer-function (b, a) form + filtfilt used above: at 4th order
    # with a band this narrow relative to Nyquist (e.g. 20-120Hz at
    # 44.1kHz), the (b, a) coefficients are numerically unstable and
    # filtfilt silently returns NaN, which the NaN-guard below then maps
    # to a fake "perfectly correlated" 1.0 — masking real anti-phase
    # content instead of detecting it. sos form doesn't have this problem.
    v2_bands = [("sub", 20, 120), ("low-mid", 120, 500), ("mid", 500, 1000)]
    band_corr_v2 = {}
    for name, lo, hi in v2_bands:
        lo_n, hi_n = lo / nyq, min(hi / nyq, .999)
        if lo_n <= 0 or lo_n >= hi_n:
            band_corr_v2[name] = 1.0
            continue
        try:
            sos = signal.butter(4, [lo_n, hi_n], btype="band", output="sos")
            Lf, Rf = signal.sosfiltfilt(sos, L), signal.sosfiltfilt(sos, R)
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
    # outlier test on the sample-difference signal. KNOWN FALSE-POSITIVE ISSUE:
    # this does not actually distinguish real clicks from loud legitimate
    # transients (kicks, snares, hi-hats) — verified against real darksynth/
    # industrial masters, where it flags essentially every hard percussive
    # attack. Scoring/gates ignore this output (see score.py) pending an
    # AR-prediction / interpolation-residual based detector. Values below
    # are kept for inspection only.
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
    RMS, final fade slope, trailing silence length. Not scored directly —
    feeds the boundary-related hard gates (evaluate_gates())."""
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
