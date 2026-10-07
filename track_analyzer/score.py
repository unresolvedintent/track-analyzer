#!/usr/bin/env python3
"""Scoring: LT_DARKSYNTH_V2 rubric — reads every threshold, weight, and
gate/effort table from rubric_darksynth_v2.json (or whatever --rubric
points at, via apply_rubric()). No numeric constants live in the scoring
functions themselves.

Other modules that need RUBRIC/BANDS/GENRE_PROFILE/WEIGHTS after import
time (e.g. measure.py, report.py, analyze.py) must access them via
`from . import score; score.RUBRIC[...]` rather than `from .score import RUBRIC`
— apply_rubric() rebinds these globals when a different rubric is loaded
(the --rubric flag), and a `from` import would freeze a stale reference
taken at import time.
"""

import json
import os
import numpy as np

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


# ── Scoring (each returns (0-100, [(priority, code_or_None, message)]) ─────────

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
    ranking").

    Tolerance scales per band with the calibration IQR: a band's IQR (in %
    energy) is converted to a dB width around its median, then normalized
    against the median IQR-in-dB across all bands, giving a per-band scale
    factor applied to deviation_curve's breakpoints. Bands where reference
    masters naturally vary more (e.g. sub-bass) get a proportionally wider
    curve; bands that are tightly consistent across masters (e.g. harsh)
    are scored more strictly. A band with missing/degenerate IQR data
    falls back to the unscaled curve (factor 1.0)."""
    cfg = RUBRIC["frequency"]
    profile = RUBRIC["calibration"]["frequency_profile"]
    profile_iqr = RUBRIC["calibration"].get("frequency_profile_iqr", {})
    if not profile:
        return None, [(9, None, "Frequency: PROVISIONAL — no calibration profile yet (run track-analyzer-calibrate)")]

    w = cfg["component_weights"]
    curve = cfg["deviation_curve"]

    iqr_db = {}
    for name in w:
        median, iqr = profile.get(name), profile_iqr.get(name)
        if not median or not iqr:
            continue
        lo, hi = median - iqr / 2, median + iqr / 2
        if lo <= 0 or hi <= lo:
            continue
        iqr_db[name] = float(10 * np.log10(hi / lo))
    typical_iqr_db = float(np.median(list(iqr_db.values()))) if iqr_db else None

    deltas, scales, total = {}, {}, 0.0
    for name, weight in w.items():
        measured, target = bands_v2.get(name), profile.get(name)
        if measured is None or not target:
            band_score = 100.0  # nothing to compare; don't penalize for missing data
        else:
            dev_db = float(10 * np.log10(max(measured, 1e-9) / max(target, 1e-9)))
            deltas[name] = dev_db
            scale = (iqr_db[name] / typical_iqr_db) if (typical_iqr_db and name in iqr_db) else 1.0
            scales[name] = scale
            band_curve = [(x * scale, y) for x, y in curve]
            band_score = _interp_curve(abs(dev_db), band_curve)
        total += weight * band_score

    issues = []
    hz_hint = {"sub": "20-60Hz", "bass": "60-120Hz", "low-mid": "120-250Hz", "mud": "250-500Hz",
               "mid": "500Hz-2kHz", "presence": "2-5kHz", "harsh": "5-8kHz", "air": "8-16kHz"}
    for name, d in sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:2]:
        if abs(d) > curve[1][0] * scales.get(name, 1.0):  # curve[1][0]: dB where the unscaled curve leaves 100
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
    they moved to hard gates and are never scored here.

    CLICKS is currently pinned to 100 via click_penalty_per=0 in the rubric:
    the detector false-positives on legitimate percussive transients (see
    measure_artifacts() in measure.py) and must not affect scoring until an
    AR-prediction based detector replaces it. Raw click data is still in
    arts for inspection; no per-track issue is raised for it."""
    cfg = RUBRIC["artifacts"]
    w = cfg["component_weights"]
    issues = []

    clicks = arts["clicks"]
    clicks_score = max(0.0, 100.0 - cfg["click_penalty_per"] * clicks)

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


def apply_gate_cap(ov, gates):
    """If any hard gate fired, cap the displayed OVERALL score — a track
    cannot show a near-100% score while a gate-level defect (clipping,
    hard start/end, mono loss, etc.) is present. Only ever lowers ov."""
    if not gates:
        return ov
    return min(ov, float(RUBRIC["verdict"]["gate_overall_cap"]))


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


def evaluate_gates(loud, phase, stereo, boundaries, integrity, fmt, final=False):
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
