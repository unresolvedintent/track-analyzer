#!/usr/bin/env python3
"""MCP server: exposes WAV track analysis as a callable tool for Claude Code."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from analyze import analyze, load_audio, measure_bands, _clean, resolve_inputs, parse_extensions

mcp = FastMCP(
    "track-analyzer",
    instructions=(
        "Analyzes WAV files for mix/mastering release readiness. "
        "Scores technical safety, frequency balance, stereo/phase, dynamics, "
        "artifacts, and genre fit (darksynth/industrial). "
        "Returns numeric scores, measured data, and actionable fix recommendations."
    ),
)


def _verdict(ov: float) -> str:
    return "Ready for release" if ov > 80 else ("Needs work" if ov >= 60 else "Significant issues")


@mcp.tool()
def analyze_tracks(
    paths: list[str],
    reference: str | None = None,
    recursive: bool = False,
    extensions: list[str] | None = None,
) -> dict:
    """
    Analyze WAV files for release readiness.

    Args:
        paths: One or more paths — each may be an absolute file path, a
            directory (non-recursive by default), or a glob pattern.
        reference: Optional reference WAV path for frequency comparison.
        recursive: When a path is a directory, also scan its subfolders.
            Also enables recursive "**" expansion in glob patterns.
        extensions: Extensions to match when a path is a directory, e.g.
            [".wav", ".flac"]. Defaults to .wav/.aiff/.aif/.flac. Files
            matched directly by a glob pattern are used as-is regardless
            of this filter.
    """
    ref_bands = None
    if reference:
        if not os.path.isfile(reference):
            return {"error": f"Reference file not found: {reference}"}
        try:
            yr, sr_r = load_audio(reference)
            ref_bands = measure_bands(yr, sr_r)
        except Exception as e:
            return {"error": f"Failed to load reference: {e}"}

    ext_set = parse_extensions(",".join(extensions)) if extensions else None
    resolved, warnings = resolve_inputs(paths, recursive=recursive, extensions=ext_set)

    results, errors = [], list(warnings)
    for path in resolved:
        try:
            r = analyze(path, ref_bands)
            r["verdict"] = _verdict(r["overall"])
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
            "summary": sorted(
                [
                    {
                        "file": r["file"],
                        "overall": r["overall"],
                        "verdict": r["verdict"],
                        "effort": r["effort"],
                        "top_blocker": r["blockers"][0] if r["blockers"] else None,
                    }
                    for r in results
                ],
                key=lambda x: x["overall"],
                reverse=True,
            ),
        }

    if errors:
        out["errors"] = errors
    return out


if __name__ == "__main__":
    mcp.run()
