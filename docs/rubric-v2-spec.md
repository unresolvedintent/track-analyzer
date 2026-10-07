# LT_DARKSYNTH_V2 Rubric Specification

Merged specification for track-analyzer v2. Combines gap analysis findings
with external rubric input, with corrections applied where the external
input was wrong or self-contradictory.

## Weights

| Category | Weight | Notes |
|---|---|---|
| Technical (T) | 0.25 | |
| Frequency (F) | 0.24 | Absorbs weight freed from artifacts |
| Stereo/phase (S) | 0.23 | |
| Dynamics (D) | 0.18 | |
| Artifacts (A) | 0.10 | Start/end boundaries move to hard gates |
| Genre fit | 0 | Reported separately, never in OVERALL |

```
OVERALL = 0.25*T + 0.24*F + 0.23*S + 0.18*D + 0.10*A
```

## 1. Technical safety score

```
T = 0.40*TP + 0.25*CLIP + 0.15*LUFS + 0.10*DC + 0.10*FORMAT
```

### Subscores

- **TP**: 100 when dBTP <= -1.0. Above -1.0: `TP = clamp(100 - 40*(dBTP + 1))`.
  True-peak oversampling: 4x for sample rates <= 48 kHz, 2x for >= 88.2 kHz,
  per channel, report max.

- **CLIP**: Confirmed clipping = at least 3 consecutive decoded samples at
  `abs(x) >= 0.9999`, or a confirmed flat-topped run. `CLIP = 100` with no
  confirmed runs; `CLIP = 0` with any confirmed run. Isolated MP3 overs are
  warnings only, not automatically confirmed clipping.

- **LUFS**: Target -14 LUFS. `LUFS = clamp(100 - 15*abs(LUFS_I + 14))`, with
  a floor of 40. Unmeasurable integrated loudness scores the floor.

- **DC**: 100 at <= 0.001; 70 at 0.01; 40 above 0.01; interpolate continuously.

- **FORMAT**: 100 for WAV/AIFF/FLAC; 75 for MP3 >= 256 kbps; 60 for lower
  bitrate MP3. Additionally, detect lossy content inside lossless containers
  (energy above 16 kHz more than 40 dB below the 8-16 kHz mean in at least
  90% of gated frames) and score FORMAT 60 regardless of container.
  Only tested when Nyquist is above 16 kHz.

### Additional reported values (not scored)

- Sample rate, bit depth, channels, duration (from soundfile.info()).
- Momentary max LUFS (400 ms window).
- Dither presence on 16-bit exports.
- Post-normalisation preview at -14 LUFS (attenuation amount and whether
  Spotify's limiter would engage). *Not yet implemented.*

## 2. Frequency balance score

### Measurement

- 8192-sample Hann FFT, 50% overlap.
- Exclude frames below -60 dBFS.
- Average L/R power.
- Normalise band power against total 20 Hz - 16 kHz power.

### Eight bands

| Band | Range |
|---|---|
| Sub | 20-60 Hz |
| Bass | 60-120 Hz |
| Low-mid | 120-250 Hz |
| Mud | 250-500 Hz |
| Mid | 500 Hz - 2 kHz |
| Presence | 2-5 kHz |
| Harsh | 5-8 kHz |
| Air | 8-16 kHz |

### Scoring

Score each band by its dB deviation from the stored calibration profile
(per-band median energy percentage written by calibrate.py):

```
dev_db = 10*log10(measured_pct / median_pct)
```

Base deviation curve, interpolated continuously:
- <= 3 dB deviation: 100
- 6 dB: 80
- 9 dB: 50
- >= 15 dB: 0

The curve is scaled per band by the calibration IQR, so bands where the
reference masters naturally vary more get more tolerance:

1. Convert each band's IQR to a dB width around its median:
   `iqr_db = 10*log10((median + IQR/2) / (median - IQR/2))`.
2. Take the median `iqr_db` across all bands as the typical width.
3. `scale = iqr_db / typical_iqr_db`; every curve breakpoint's dB value is
   multiplied by `scale` for that band.

A band with missing or degenerate IQR data (lower bound <= 0) uses the
unscaled curve (`scale = 1.0`). A band that cannot be measured scores 100.
The two largest deviations are reported as issues when they exceed the
band's scaled 3 dB point.

Band weights:

```
F = 0.15*sub + 0.15*bass + 0.10*low_mid + 0.20*mud + 0.10*mid + 0.15*presence + 0.10*harsh + 0.05*air
```

No automatic 80-point ceiling. If the frequency profile has not been
calibrated (via calibrate.py), the frequency score is N/A (PROVISIONAL) and
its 0.24 weight is redistributed proportionally across the other
categories in OVERALL.

If reference tracks are supplied via --reference, report a separate
reference-delta score but do not replace the rubric score or alter ranking.

## 3. Stereo/phase score

### Measurements

- Sub correlation: 20-120 Hz (fourth-order Butterworth).
- Low-mid correlation: 120-500 Hz (fourth-order Butterworth).
- Mid correlation: 500 Hz - 1 kHz (fourth-order Butterworth).
- Sustained negative correlation in 100 ms frames.
- Mono-sum loss (mid RMS vs stereo RMS, dB).
- Side/mid RMS ratio.
- L/R RMS imbalance in dB.
- Per-band side-channel energy (to catch sub content leaking into side).

### Scoring

```
S = 0.30*SUB + 0.25*MONO + 0.20*LOWMID + 0.15*WIDTH + 0.10*BALANCE
```

- **SUB**: 100 at correlation >= 0.95; 0 at <= 0.75; interpolate.
- **LOWMID**: Average score for 120-500 Hz and 500 Hz - 1 kHz bands.
  100 at correlation >= 0.50; 0 at <= 0.20; interpolate.
- **MONO**: 100 at mono loss >= -1 dB; 0 at <= -3 dB; interpolate.
- **WIDTH**: 100 at side/mid <= 1.1; 0 at >= 1.6; interpolate.
- **BALANCE**: 100 at L/R imbalance <= 0.5 dB; 0 at >= 3 dB; interpolate.

Report global correlation but do not let it conceal unsafe
frequency-specific phase behaviour.

## 4. Dynamics score

```
D = 0.60*LRA + 0.25*PSR + 0.15*CREST
```

### LRA (asymmetric, genre-calibrated)

- 100 between 4 and 9 LU.
- Below 4: subtract 15 points per LU.
- Above 9: subtract 8 points per LU.
- Clamp to 0-100.

This is intentionally asymmetric: low LRA (over-compression) is penalised
more than high LRA (loose dynamics), because over-compression is harder
to fix and more common in the target genre.

### PSR

- 100 at >= 4 dB.
- Subtract 25 points per dB below 4.
- Clamp to 0-100.

### CREST

- 100 at >= 9 dB.
- Subtract 15 points per dB below 9.
- Clamp to 0-100.
- Report unusually high crest factor without penalising unless it exceeds
  the stated range.

## 5. Artifact score

```
A = 0.70*CLICKS + 0.30*NOISE
```

When noise floor is unmeasurable (track has no genuinely quiet frames),
mark noise as N/A and renormalise: `A = CLICKS`.

### Click detection (currently disabled)

Clicks are detected with `librosa.onset.onset_strength` plus a
median-absolute-deviation outlier test on the sample-difference signal
(threshold: median + 12 x MAD-scaled std, within 1 ms of an onset).

This detector false-positives on legitimate percussive transients in this
genre, so it is disabled in scoring and gates:

`CLICKS = clamp(100 - 0*confirmed_click_count)`, so CLICKS is always 100.

The penalty (`click_penalty_per`) is pinned to 0 in the rubric until an
AR-prediction based detector replaces it. Raw click counts are still
reported for inspection, and the CLI prints a note saying so.

### Noise floor

Measured as the median 10 ms RMS of frames more than 30 dB below the
loudest frame, only when such frames make up at least 1% of the track;
otherwise N/A.

- 100 at <= -50 dBFS.
- 80 between -50 and -40.
- 60 above -40.

### Boundaries (moved to hard gates, not scored)

Start and end conditions are hard gates, not part of the artifact score.
This prevents cost-1 fixes (adding a fade) from dominating a 10% category.

Report internal silence gaps separately as arrangement data. Do not
penalise unless discontinuities indicate an accidental edit.

## 6. Hard gates

Binary pass/fail conditions evaluated separately from the numeric score.
Any triggered gate can override the verdict regardless of OVERALL.

- Confirmed digital clipping.
- True peak above -1.0 dBTP.
- Sustained negative correlation for at least 500 ms (5 consecutive
  100 ms frames with negative L/R correlation).
- Mono-sum loss below -3 dB.
- Hard start: first sample > 0.02 without a fade-in, where a fade-in means
  the first sample is below 10% of the peak reached in the first 5 ms.
- Hard end: final 50 ms RMS > -45 dBFS without a downward fade (final
  fade slope >= 0).

Abrupt cuts are caught by the hard start and hard end gates. There is no
separate boundary-click gate while click detection is disabled.

### Gate score cap

If any hard gate fires, the displayed OVERALL is capped at 84, whatever the
weighted score. A track cannot show a near-100% score while a gate-level
defect is present. The uncapped weighted score is reported alongside it
as the mix score (`overall_uncapped`).
- Lossy container when running with --final flag.
- Lossy content detected (spectral cliff) regardless of container.
- Sample rate not 44.1 or 48 kHz when running with --final flag.

## 7. Verdict

- **Ready**: OVERALL >= 85, every category >= 70, and no hard gate.
- **Minor work**: OVERALL 75-84, or exactly one easily corrected gate
  (any gate with effort cost 1).

Any gate with effort cost 3 counts as reconstruction-level. Verdict uses
the capped OVERALL.
- **Needs work**: OVERALL 60-74.
- **Significant work**: OVERALL < 60, two or more hard gates, or any
  reconstruction-level problem.

## 8. Effort and ranking

Assign correction costs from a fixed lookup table (deterministic, no
heuristics):

- **Cost 1**: Gain adjustment, limiter ceiling, short fade, lossless export.
- **Cost 2**: EQ/dynamic-EQ correction, stereo narrowing, local artifact repair.
- **Cost 3**: Stem remix, dynamics restoration, phase reconstruction,
  arrangement editing.

Calculate total `effort_points` as a pure function of gate flags and
warning flags.

- **Low**: 0-2 points.
- **Medium**: 3-5 points.
- **High**: >= 6 points.

Ranking key: `(effort_points ascending, uncapped OVERALL descending)`.
The capped OVERALL is display-only for ordering, since a gate fix doesn't
change how much other work is left. Do not manually reorder tracks.

## 9. Genre fit (descriptive only, not in OVERALL)

Report as a separate INFERRED GENRE FIT section. Use HPSS-based measures:

- Percussive-to-harmonic energy ratio.
- Kick-band energy after harmonic removal vs sub energy (sub-masking-kick
  detection).
- Spectral flatness as a distortion/texture proxy.

Assess (with explicit uncertainty labelling):
- Low-end weight.
- Percussive strength.
- Metallic/noisy spectral texture.
- Section contrast.
- Repetition and arrangement variation.
- Potential fatigue from mud, harshness or density.

Do not claim to identify specific instruments, emotional effect, mix
pleasure or sound-design quality. Label all assessments as inference.

## 10. Configuration

All thresholds, weights, band edges, penalty slopes, gate conditions and
verdict boundaries live in `rubric_darksynth_v2.json`. The tool loads it
at startup and includes `rubric_version` in every result.

No numeric constants in the scoring functions. Any threshold change
creates a new version and requires rerunning every track for comparability.

## 11. Calibration

`calibrate.py` takes a folder of reference masters, runs measure_bands()
on each, computes the per-band median and IQR, and writes them into
rubric_darksynth_v2.json as the target frequency profile. Requires at
least `calibrate.min_files` (8) usable files. Records only the per-band
median, IQR, file count and date, never filenames or paths.

Until calibration is run, the frequency score is marked PROVISIONAL.

## 12. Measurement functions needed

- `measure_format()`: container, subtype, sample rate, bit depth, channels,
  duration. Uses soundfile.info(), no ffprobe/FFmpeg dependency.
- `measure_bands()`: rewritten to 8192-sample Hann, 50% overlap, -60 dBFS
  frame gate, eight bands, percentage of total 20 Hz - 16 kHz power.
- `measure_boundaries()`: first-sample amplitude, fade-in presence over
  first 5 ms, final 50 ms RMS, final fade slope, trailing silence length.
- `measure_balance()`: L/R RMS imbalance in dB, per-band side-channel energy.
- `measure_integrity()`: lossy-content spectral cliff test, dither presence
  on 16-bit, momentary max LUFS at 400 ms.
- `measure_texture()`: librosa.effects.hpss percussive/harmonic ratio,
  kick-band energy post-HPSS vs sub energy, spectral flatness.
- Click detection: librosa.onset.onset_strength + MAD outlier test
  (disabled in scoring, see section 5).

## 13. File structure

Split into modules when analyze.py exceeds 400 lines:
- `measure.py`: all measurement functions.
- `score.py`: all scoring functions, reading from rubric JSON.
- `report.py`: CLI formatting and output.
- `analyze.py`: entry point, argument parsing, orchestration.
- `calibrate.py`: reference calibration tool.
- `server.py`: MCP server.

## 14. Testing

Golden-file test with three synthetic WAVs:
- One with confirmed clipping.
- One with inverted-phase channels.
- One with a hard start at amplitude 0.6.

Assert exact expected scores and exact expected gate lists. This is what
enforces rubric consistency, not a prompt.
