#!/usr/bin/env python3
"""Track Analyzer — release readiness for darksynth/industrial.

Entry point, argument parsing, and orchestration only. Measurement lives
in measure.py, scoring (incl. rubric loading, gates, effort) in score.py,
CLI formatting in report.py.
"""

import argparse, glob, json, os, sys
import numpy as np

from .measure import (
    load_audio, measure_loudness, measure_bands_v1, measure_bands, measure_stereo,
    measure_phase, measure_dynamics, measure_artifacts, measure_format,
    measure_boundaries, measure_balance, measure_integrity, measure_texture,
)
from .score import (
    score_technical, score_frequency, score_reference_delta, score_stereo,
    score_dynamics, score_artifacts, score_genre, weighted_overall, apply_gate_cap,
    top_blockers, verdict, evaluate_gates, compute_effort, apply_rubric, load_rubric,
    DEFAULT_RUBRIC_PATH,
)
from . import score as _score  # qualified access to RUBRIC (rebound by --rubric; see score.py docstring)
from .report import print_track_report, print_ranking

DEFAULT_EXTENSIONS = {".wav", ".aiff", ".aif", ".flac"}


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
    ov_uncapped = weighted_overall(scores)

    gates = evaluate_gates(loud, phase, stereo, boundaries, integrity, fmt, final=final)
    ov = apply_gate_cap(ov_uncapped, gates)

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

    return {"file": path, "scores": scores, "overall": ov, "overall_uncapped": ov_uncapped,
            "verdict": v, "blockers": blockers, "gates": gates,
            "effort_points": effort_points, "effort": effort_bucket,
            "rubric_version": _score.RUBRIC["version"],
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
        description="Track Analyzer - WAV release readiness for darksynth/industrial",
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
                        help="Rubric config to score against "
                             f"(default: bundled {os.path.basename(DEFAULT_RUBRIC_PATH)})")
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

    print("Note: click/artifact detection is disabled in scoring and gates - the diff/MAD "
          "detector false-positives on legitimate percussive transients in this genre. Click "
          "counts shown below are raw candidate data for inspection only, pending an "
          "AR-prediction based detector.")
    print()

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
