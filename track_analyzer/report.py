#!/usr/bin/env python3
"""CLI reporting: text and JSON-adjacent formatting for analyze() results."""

import os
import numpy as np

from . import score
from .measure import BANDS_V2

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


def print_measured_data(raw, reference_delta=None):
    loud   = raw["loudness"]
    bands  = raw["bands"]
    stereo = raw["stereo"]
    phase  = raw["phase"]
    dyn    = raw["dynamics"]
    arts   = raw["artifacts"]

    # Recomputed per call (not cached at import time): score.BANDS can
    # change if a different --rubric is loaded mid-process.
    bnames = [b[0] for b in score.BANDS]

    def band_row(fn):
        return "   ".join(f"{_SHORT[n]} {fn(n)}" for n in bnames)

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
        print(f"{label:<20} | {sc_str:>5} | {score.status_of(sc)}")
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
    cols = (28, 6, 9, 20, 40, 8)
    header = (f"{'Track':<{cols[0]}} | {'Score':>{cols[1]}} | {'Mix score':>{cols[2]}} | "
              f"{'Verdict':<{cols[3]}} | {'Main blocker':<{cols[4]}} | Effort")
    sep = "-+-".join("-" * c for c in cols)
    print("RANKING")
    print(header)
    print(sep)
    # Ranking key: (effort_points ascending, uncapped OVERALL descending) — least
    # remaining work first; "Score" (gated) is display-only for this ordering,
    # since a boundary/gate fix doesn't change how much other work is left.
    for r in sorted(results, key=lambda x: (x["effort_points"], -x["overall_uncapped"])):
        name = os.path.splitext(os.path.basename(r["file"]))[0]
        name = name[:cols[0]-1] if len(name) >= cols[0] else name
        blocker = (r["blockers"][0][:cols[4]-1] if r["blockers"] else "—")
        verd = r["verdict"][:cols[3]-1]
        effort_str = f"{r['effort']} ({r['effort_points']})"
        print(f"{name:<{cols[0]}} | {r['overall']:>{cols[1]-1}.0f}% | "
              f"{r['overall_uncapped']:>{cols[2]-1}.0f}% | "
              f"{verd:<{cols[3]}} | {blocker:<{cols[4]}} | {effort_str}")
    print()
