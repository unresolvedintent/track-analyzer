#!/usr/bin/env python3
"""calibrate.py — build the frequency calibration profile (spec section 11).

Takes a folder of reference masters, runs measure_bands() (the v2,
percentage-of-total-power band measurement) on each, computes the per-band
median and IQR across the set, and writes the medians into the rubric JSON
as the target frequency profile that score_frequency() compares tracks
against. Requires at least rubric['calibrate']['min_files'] (default 8)
usable files — refuses to run on fewer, since a profile built from too few
masters isn't a reliable genre baseline.
Until this has been run, score_frequency() reports PROVISIONAL.
"""

import argparse
import datetime
import json
import sys

import numpy as np

from measure import load_audio, measure_bands, BANDS_V2
from score import DEFAULT_RUBRIC_PATH, load_rubric
from analyze import resolve_inputs, parse_extensions, DEFAULT_EXTENSIONS

def compute_profile(paths, min_files):
    band_names = [name for name, *_ in BANDS_V2]
    values = {name: [] for name in band_names}
    used = []
    for path in paths:
        try:
            y, sr = load_audio(path)
            bands = measure_bands(y, sr)
        except Exception as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)
            continue
        if any(bands.get(name) is None for name in band_names):
            print(f"Warning: skipping {path}: too short to measure all bands", file=sys.stderr)
            continue
        for name in band_names:
            values[name].append(bands[name])
        used.append(path)

    if len(used) < min_files:
        print(f"Error: only {len(used)} usable file(s), need at least {min_files}", file=sys.stderr)
        sys.exit(1)

    medians, iqrs = {}, {}
    for name in band_names:
        arr = np.array(values[name])
        medians[name] = float(np.median(arr))
        iqrs[name] = float(np.percentile(arr, 75) - np.percentile(arr, 25))

    return medians, iqrs, used


def main():
    parser = argparse.ArgumentParser(
        description="Build the darksynth/industrial frequency calibration profile "
                     "from a folder of reference masters.",
    )
    parser.add_argument("folder", help="Directory of reference master WAV/AIFF/FLAC files "
                                        "(also accepts a glob pattern or individual file paths)")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subfolders")
    parser.add_argument("--ext", metavar="EXTS",
                        help="Comma-separated extensions to match "
                             f"(default: {','.join(sorted(DEFAULT_EXTENSIONS))})")
    parser.add_argument("--rubric", metavar="JSON", default=DEFAULT_RUBRIC_PATH,
                        help=f"Rubric file to update (default: {DEFAULT_RUBRIC_PATH})")
    args = parser.parse_args()

    try:
        rubric = load_rubric(args.rubric)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: failed to load rubric {args.rubric}: {e}", file=sys.stderr)
        sys.exit(1)
    min_files = rubric["calibrate"]["min_files"]

    extensions = parse_extensions(args.ext)
    paths, warnings = resolve_inputs([args.folder], recursive=args.recursive, extensions=extensions)
    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)
    if not paths:
        print("Error: no matching audio files found", file=sys.stderr)
        sys.exit(1)
    if len(paths) < min_files:
        print(f"Error: found {len(paths)} file(s), need at least {min_files} to calibrate", file=sys.stderr)
        sys.exit(1)

    medians, iqrs, used = compute_profile(paths, min_files)

    calibrated_at = datetime.date.today().isoformat()
    rubric["calibration"] = {
        "frequency_profile": medians,
        "frequency_profile_iqr": iqrs,
        "file_count": len(used),
        "calibrated_at": calibrated_at,
    }
    with open(args.rubric, "w") as f:
        json.dump(rubric, f, indent=2)
        f.write("\n")

    print(f"Calibration written to {args.rubric}")
    print(f"  Files used: {len(used)}")
    print(f"  Date:       {calibrated_at}")
    print()
    print(f"{'Band':<10} | {'Median %':>9} | {'IQR':>6}")
    print(f"{'-'*10}-+-{'-'*9}-+-{'-'*6}")
    for name, *_ in BANDS_V2:
        print(f"{name:<10} | {medians[name]:>9.2f} | {iqrs[name]:>6.2f}")


if __name__ == "__main__":
    main()
