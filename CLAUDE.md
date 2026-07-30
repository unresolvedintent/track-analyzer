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
- Single entry point: analyze.py
- Keep modular but in one file unless it exceeds 400 lines
- Include --help, --json, and --reference flags

