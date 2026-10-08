# track-analyzer

A command-line tool that measures WAV, AIFF and FLAC masters against a fixed, versioned scoring rubric and reports whether each one is ready to release. It is for producers who have several bounces of the same track and need to decide which one to release, and what each one still needs fixed.

The bundled rubric targets darksynth and industrial electronic music, where heavy sub-bass, compressed dynamics and a recessed air band are normal.

## How it differs from other tools

Loudness meters report LUFS, true peak and LRA. Feature-extraction libraries such as librosa report spectral and rhythmic descriptors. This tool uses both kinds of measurement, but its output is a decision rather than a set of readings. None of the ideas below is new on its own; the tool combines them.

- **Versioned genre rubric.** Every threshold, weight and penalty slope lives in one JSON file (`rubric_darksynth_v2.json`). Results carry the rubric version, so scores from the same version are comparable and a threshold change means a new version.
- **Hard gates separate from the score.** Defects such as confirmed clipping or a hard start are pass/fail gates. They are not averaged into a number where they could be hidden by good scores elsewhere.
- **Ranking by fix effort.** Multiple files are ranked by how much work remains (effort points from a fixed cost table), then by score. A clean track that needs EQ work can rank below a track that only needs a fade.
- **MCP server.** A Model Context Protocol server exposes the analysis as a tool, so an agent such as Claude Code can analyze files and read structured results.

## Install

Requires Python 3.12 or newer. The package is not on PyPI; install from a clone of this repository:

```
pip install .              # regular install
pip install -e ".[dev]"    # editable install with pytest, for development
```

This installs three commands: `track-analyzer`, `track-analyzer-calibrate` and `track-analyzer-mcp`.

## Usage

```
track-analyzer mix.wav                          # one file
track-analyzer bounces/                         # every matching file in a folder
track-analyzer bounces/ --recursive             # include subfolders
track-analyzer bounces/ --ext wav,flac          # extensions to match (default: .aif,.aiff,.flac,.wav)
track-analyzer "bounces/*_v3.wav"               # glob pattern
track-analyzer mix.wav --json                   # JSON output, including all raw measurements
track-analyzer mix.wav --reference ref.wav      # add a per-band comparison against a reference track
track-analyzer mix.wav --final                  # also require a lossless container and 44.1/48 kHz
track-analyzer mix.wav --rubric my_rubric.json  # score against a different rubric file
```

`--reference` output is informational only. It never changes the frequency score, the overall score or the ranking.

## Sample output

Two synthetic 10-second test signals: `bounce_a` is normalised to -14 LUFS with 50 ms fades; `bounce_b` is the same material pushed into clipping with no fade-in. Command: `track-analyzer bounce_a.wav bounce_b.wav`. The output below is the ranking plus the report for `bounce_b`; the report for `bounce_a` has the same layout and is left out for length. Synthetic tones are not music, so the frequency scores are low.

```
Note: click/artifact detection is disabled in scoring and gates - the diff/MAD detector false-positives on legitimate percussive transients in this genre. Click counts shown below are raw candidate data for inspection only, pending an AR-prediction based detector.

RANKING
Track                        |  Score | Mix score | Verdict              | Main blocker                             | Effort
-----------------------------+--------+-----------+----------------------+------------------------------------------+---------
bounce_a                     |    80% |       80% | Minor work           | LRA 1.1 LU - over-compressed, ease limi  | High (6)
bounce_b                     |    63% |       63% | Significant work     | Confirmed clipping (83586 sample(s) at   | High (11)
...
VERDICT: Significant work - 63%   [rubric v2]

Category             | Score | Status
---------------------+-------+--------
Technical safety     |    29 | FAIL
Frequency balance    |    54 | FAIL
Stereo / phase       |   100 | PASS
Dynamics             |    54 | FAIL
Artifacts            |   100 | PASS
Genre fit            |    95 | PASS

MEASURED DATA
  Loudness    -2.0 LUFS   1.30 dBTP   1.0 LU LRA   -2.0 LUFS ST max
  Freq (dB)   sub 53.9   bass 45.6   lo-mid 30.1   hi-mid 22.6   pres 11.2   air 11.2
  vs ref      no reference
  Stereo      width 0.00   mono +0.0 dB
  L/R corr    sub 1.00   bass 1.00   lo-mid 1.00   hi-mid 1.00   pres 1.00   air 1.00
  Dynamics    crest 2.4 dB   PSR 3.3 dB
  Phase       avg 1.00   min 1.00   worst high-mid
  Artifacts   0 clicks   DC +0.000365   noise floor n/a
  Format      WAV/PCM_24   44100Hz   24-bit   2ch   10.0s
  Boundaries  start amp 1.000   end -5.9dB   slope -3.84dB/frame   trail silence 0.00s
  Balance     L/R +0.00dB
  Side (%)    sub n/a%   bass n/a%   low-mid n/a%   mud n/a%   mid n/a%   presence n/a%   harsh n/a%   air n/a%
  Integrity   16k+ cliff 0.0dB   dither None   momentary max -2.0 LUFS
  Texture     perc/harm 0.11   kick/sub 0.00   flatness 0.012
  Freq v2 (%) sub 70.4%   bass 1.3%   low-mid 19.0%   mud 1.9%   mid 6.4%   presence 0.3%   harsh 0.2%   air 0.5%
  Corr v2     sub(20-120) 1.00   low-mid(120-500) 1.00   mid(500-1k) 1.00

TOP BLOCKERS
  Confirmed clipping (83586 sample(s) at full scale) - reduce pre-limiter gain
  True peak 1.3 dBTP - hard limit to -1.0 dBTP before export
  Integrated -2.0 LUFS vs -14 target - adjust gain staging

HARD GATES
  confirmed_clipping
  true_peak_exceeded
  hard_start

FIX EFFORT: High (11 point(s))
```

## How scoring works

Five categories are scored 0-100 and combined into OVERALL:

| Category | Weight | What it covers |
|---|---|---|
| Technical safety | 0.25 | True peak, confirmed clipping, LUFS vs -14 target, DC offset, format |
| Frequency balance | 0.24 | Eight bands vs a calibration profile from reference masters |
| Stereo / phase | 0.23 | Sub and low-mid correlation, mono-sum loss, width, L/R balance |
| Dynamics | 0.18 | LRA, peak-to-short-term-loudness ratio (PSR), crest factor |
| Artifacts | 0.10 | Clicks (currently disabled, see limitations) and noise floor |

Genre fit is also reported but has no weight and never affects OVERALL or ranking. If the frequency profile has not been calibrated, frequency is shown as N/A and its weight is redistributed across the other four categories.

**Hard gates** are checked separately: `confirmed_clipping`, `true_peak_exceeded` (above -1.0 dBTP), `sustained_negative_correlation` (500 ms or longer), `mono_sum_loss` (below -3 dB), `hard_start`, `hard_end`, `lossy_content_detected`, and with `--final`, `lossy_container_final` and `sample_rate_final`.

**Score vs mix score.** If any gate fires, the displayed score (OVERALL) is capped at 84, so a track cannot show a high score while it has a gate-level defect. The mix score is the uncapped weighted score. The ranking uses the mix score, because fixing a gate does not change how much other work remains.

**Effort points** come from a fixed table in the rubric: each gate and each warning has a cost of 1 (gain, limiter ceiling, fade, export), 2 (EQ, stereo narrowing, local repair) or 3 (remix or reconstruction). Totals map to Low (0-2), Medium (3-5) and High (6 or more). Multiple files are ranked by effort points ascending, then mix score descending.

**Verdict tiers**, checked in this order:

- **Significant work**: OVERALL below 60, two or more gates, or any cost-3 gate (sustained negative correlation, lossy content).
- **Ready**: OVERALL 85 or more, every scored category 70 or more, and no gates.
- **Minor work**: exactly one cost-1 gate, or OVERALL 75-84.
- **Needs work**: anything else.

Category status: PASS at 80 or more, WARNING at 60-79, FAIL below 60.

## Rubric changelog: v1 to v2

[v1](docs/rubric-v1-origin.md) began as a scoring prompt. [v2](docs/rubric-v2-spec.md) is the current specification, implemented in code with all numbers in the rubric JSON.

- **Weights.** Artifacts dropped from 0.20 to 0.10. Frequency went from 0.20 to 0.24, stereo from 0.20 to 0.23, dynamics from 0.15 to 0.18.
- **Boundaries became gates.** In v1, start and end behaviour made up 60% of the artifact score. Adding a fade is a cost-1 fix, and letting it dominate a category distorted the numeric score. In v2, a hard start (first sample above 0.02 with no fade-in) and a hard end (final 50 ms above -45 dBFS with no downward fade) are gates instead, and the artifact score covers only clicks and noise.
- **Frequency scoring.** v1 used fixed target percentage ranges per band with fixed penalties per percentage point. v2 scores each band by its dB deviation from a calibration profile (the per-band median of a set of reference masters): 100 within 3 dB, 80 at 6 dB, 50 at 9 dB, 0 at 15 dB or more.
- **IQR-scaled tolerance.** Reference masters vary much more in some bands than others. Each band's interquartile range (IQR) is converted to a dB width around its median and divided by the median width across all bands. The deviation curve for that band is stretched by that factor, so bands where masters naturally vary more get more tolerance and consistent bands are scored more strictly.
- **Gate cap.** New in v2: OVERALL is capped at 84 when any gate fires.
- **New gates.** Lossy content inside a lossless container (spectral cliff above 16 kHz), and with `--final`, lossy container and sample rate.
- **Dynamics.** LRA full marks moved from 5-8 LU to 4-9 LU and became asymmetric: 15 points per LU below, 8 per LU above, because over-compression is harder to fix and more common in this genre.
- **Loudness and correlation thresholds.** The LUFS penalty went from 5 to 15 points per LU from -14, with the floor lowered from 60 to 40. Full sub correlation now needs 0.95 (was 0.90), with zero at 0.75 (was 0.70).
- **Clicks disabled.** See known limitations.
- **No FFmpeg.** Format data comes from soundfile instead of ffprobe.

## Calibration

```
track-analyzer-calibrate references/ [--recursive] [--ext wav,flac] [--rubric my_rubric.json]
```

This measures every file's eight-band energy distribution and writes the per-band median and IQR into the rubric, along with the file count and date. It needs at least 8 usable files. Reference audio, filenames and paths are never stored. The bundled rubric ships with a profile calibrated from 8 reference masters.

By default this updates the bundled rubric. With an editable install that is the file in your checkout. With a regular install it is the copy inside the installed package, which is overwritten on reinstall. In that case, work on your own copy and pass it to both commands:

```
cp "$(python -c 'import track_analyzer.score as s; print(s.DEFAULT_RUBRIC_PATH)')" my_rubric.json
track-analyzer-calibrate references/ --rubric my_rubric.json
track-analyzer mix.wav --rubric my_rubric.json
```

## MCP server for Claude Code

`track-analyzer-mcp` runs a stdio MCP server with one tool, `analyze_tracks`. It accepts the same inputs as the CLI (`paths`, `reference`, `recursive`, `extensions`, `final`) and returns scores, gates, effort, verdict and raw measurements as structured data. For several files it adds a summary ranked the same way as the CLI.

Register it with Claude Code:

```
claude mcp add track-analyzer -- track-analyzer-mcp
```

If the command is not on your PATH (for example, it is inside a virtual environment), give the full path, such as `/path/to/venv/bin/track-analyzer-mcp`. The server can also be started with `python -m track_analyzer.server`.

## Known limitations

- **Click detection is disabled.** The detector uses onset detection plus a median-absolute-deviation outlier test on sample differences. It flagged percussive transients as clicks on every real track tested; waveform inspection at flagged positions showed continuous oscillations, not discontinuities. So the click penalty is pinned to 0 and no gate uses it. Raw click counts are still shown for inspection. The intended fix is a detector based on an autoregressive (AR) prediction residual.
- **Calibration from 8 masters gives wide IQRs.** For example, the bundled profile has sub-bass at a median of 11.7% of total energy with an IQR of 12.3 percentage points, and bass at 24.0% with an IQR of 16.0. The frequency score is only as reliable as that reference set.
- **Not validated against listening.** Scores have not been compared with listening tests or with expert judgement of the same tracks.
- **Genre-specific.** The bundled thresholds and calibration are for darksynth and industrial music. Other genres need their own rubric and calibration.

## Testing

```
pip install -e ".[dev]"
pytest
```

The tests generate synthetic WAV files with known defects (confirmed clipping, inverted-phase channels, a hard start) and a known-good signal, and check the gates and score ranges each one must produce. A rubric test checks that the category weights sum to 1.0 and that the effort cost tables match the gates and warning codes the code can emit.

## Licence

MIT. See [LICENSE](LICENSE).
