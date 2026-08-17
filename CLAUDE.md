# Track Analyzer

Python CLI tool for analyzing WAV files for mix/mastering readiness.

## Conventions
- Python 3.11+
- Use librosa for audio loading and spectral analysis
- Use pyloudnorm for LUFS and LRA measurement
- Use scipy for true peak detection and peak finding
- Use numpy for stereo/phase analysis
- No GUI, no matplotlib display, no web server
- Terminal output only, no color codes
- All frequency analysis in Hz, loudness in LUFS, peaks in dBFS
- Genre context: darksynth/industrial electronic (heavy sub-bass is intentional, compressed dynamics are normal, recessed air band is expected)
- Target reference: streaming platforms (Spotify -14 LUFS, Apple -16 LUFS)
- Output should be concise and actionable, not academic

## Structure
- `analyze.py` — entry point: CLI arg parsing, input resolution, orchestration (`analyze()`)
- `measure.py` — pure signal analysis (`measure_*` functions), no scoring logic
- `score.py` — rubric loading, all `score_*` functions, gates, effort, verdict; reads every
  threshold from `rubric_darksynth_v2.json` (or `--rubric`), no numeric constants in code
- `report.py` — CLI text/JSON-adjacent formatting (`print_track_report`, `print_ranking`)
- `calibrate.py` — builds the frequency calibration profile from a folder of reference masters
- `server.py` — MCP server exposing `analyze()` as a tool for Claude Code
- `tests/test_golden.py` — golden tests: synthetic WAVs with known defects (clipping,
  inverted phase, hard start), asserting expected gates and score ranges; run with `pytest`
- Modules needing `RUBRIC`/`BANDS`/etc. after import time must `import score` and access
  `score.RUBRIC[...]` rather than `from score import RUBRIC` — `apply_rubric()` (`--rubric`
  flag) rebinds those globals, and a `from` import freezes a stale reference
- Include --help, --json, and --reference flags

