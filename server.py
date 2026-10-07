#!/usr/bin/env python3
"""MCP server: exposes WAV track analysis as a callable tool for Claude Code."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from analyze import analyze, load_audio, measure_bands_v1, _clean, resolve_inputs, parse_extensions

mcp = FastMCP(
    "track-analyzer",
    instructions=(
        "Analyzes WAV files for mix/mastering release readiness against the "
        "LT_DARKSYNTH_V2 rubric. Scores technical safety, frequency balance, "
        "stereo/phase, dynamics, and artifacts (weighted into OVERALL); genre "
        "fit is reported separately and never affects OVERALL. Also reports "
        "hard gates (binary pass/fail conditions that can override the verdict "
        "regardless of OVERALL) and effort_points (a table-driven estimate of "
        "fix cost). Returns numeric scores, measured data, hard gates, and "
        "actionable fix recommendations."
    ),
)


@mcp.tool()
def analyze_tracks(
    paths: list[str],
    reference: str | None = None,
    recursive: bool = False,
    extensions: list[str] | None = None,
    final: bool = False,
) -> dict:
    """
    Analyze WAV files for release readiness against the v2 rubric.

    Args:
        paths: One or more paths — each may be an absolute file path, a
            directory (non-recursive by default), or a glob pattern.
        reference: Optional reference WAV path for frequency comparison.
            Reported separately (reference_delta) — never affects the
            frequency score, OVERALL, or ranking.
        recursive: When a path is a directory, also scan its subfolders.
            Also enables recursive "**" expansion in glob patterns.
        extensions: Extensions to match when a path is a directory, e.g.
            [".wav", ".flac"]. Defaults to .wav/.aiff/.aif/.flac. Files
            matched directly by a glob pattern are used as-is regardless
            of this filter.
        final: Also enforce release-format gates: lossless container,
            44.1/48kHz sample rate.
    """
    ref_bands = None
    if reference:
        if not os.path.isfile(reference):
            return {"error": f"Reference file not found: {reference}"}
        try:
            yr, sr_r = load_audio(reference)
            ref_bands = measure_bands_v1(yr, sr_r)
        except Exception as e:
            return {"error": f"Failed to load reference: {e}"}

    ext_set = parse_extensions(",".join(extensions)) if extensions else None
    resolved, warnings = resolve_inputs(paths, recursive=recursive, extensions=ext_set)

    results, errors = [], list(warnings)
    for path in resolved:
        try:
            r = analyze(path, ref_bands, final=final)
            results.append(_clean(r))
        except Exception as e:
            errors.append(f"Error analyzing {os.path.basename(path)}: {e}")

    if not results:
        return {"error": "; ".join(errors) if errors else "No files analyzed"}

    if len(results) == 1:
        out = results[0]
    else:
        out = {
            "tracks": results,
            # Ranking key: (effort_points ascending, uncapped OVERALL descending) —
            # spec section 8; matches print_ranking() in report.py.
            "summary": sorted(
                [
                    {
                        "file": r["file"],
                        "overall": r["overall"],
                        "overall_uncapped": r["overall_uncapped"],
                        "verdict": r["verdict"],
                        "effort": r["effort"],
                        "effort_points": r["effort_points"],
                        "gates": r["gates"],
                        "rubric_version": r["rubric_version"],
                        "top_blocker": r["blockers"][0] if r["blockers"] else None,
                    }
                    for r in results
                ],
                key=lambda x: (x["effort_points"], -x["overall_uncapped"]),
            ),
        }

    if errors:
        out["errors"] = errors
    return out


if __name__ == "__main__":
    mcp.run()
