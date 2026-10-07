> This is the original v1 scoring prompt (`LT_DARKSYNTH_V1`) that the v2 rubric in `rubric-v2-spec.md` was derived from. Kept for history; it does not describe current behaviour.

Analyze the uploaded audio tracks for production release-readiness using the fixed rubric below.

## Non-negotiable consistency rules

- Rubric version: `LT_DARKSYNTH_V1`.
- Apply identical measurements, thresholds, formulas and weights to every file.
- Never manually adjust scores or rankings.
- Do not reuse scores from previous chats.
- Separate measured results from inferred aesthetic judgement.
- Do not claim to have heard the music unless actual audio playback/listening is available.
- If audio cannot be decoded or measured, say so. Never invent values.
- Use lossless files where available. Flag MP3 analysis as potentially affected by codec overshoot.
- Define `clamp(x) = min(100, max(0, x))`.

## Measurement tools and methods

Use:

- `ffprobe`: codec, container, sample rate, channels, duration and bitrate.
- FFmpeg EBU-R128/BS.1770 analysis: integrated LUFS, LRA and true peak.
- Decode audio to floating-point PCM at its native sample rate.
- NumPy/SciPy or equivalent for sample statistics, FFT/STFT, filtering, correlation, M/S energy, RMS windows and artifact detection.
- For spectral analysis: 8192-sample Hann FFT, 50% overlap; exclude frames below -60 dBFS; average L/R power; normalise band power against total 20 Hz-16 kHz power.
- For band-specific stereo analysis: fourth-order Butterworth filters.

## Raw measurements required

For every track return:

- Format, codec, bitrate, sample rate, channels and duration.
- LUFS-I, dBTP and LRA.
- Sample peak, RMS, crest factor and PSR.
- Confirmed clipping runs and isolated near-full-scale samples.
- DC offset.
- Frequency-band power percentages.
- Global, sub, low-mid and mid correlation.
- M/S ratio, mono-sum loss and L/R imbalance.
- First-sample amplitude, first 5 ms behaviour, final 50 ms RMS, final fade slope and trailing silence.
- Click candidates, noise floor and internal silence gaps.
- Structural proxies: section-level range, spectral flux, percussive share, contrast and repetition.

## 1. Technical safety score

Calculate:

`T = 0.40*TP + 0.25*CLIP + 0.15*LUFS + 0.10*DC + 0.10*FORMAT`

Subscores:

- `TP = 100` when dBTP <= -1.0.
- Above -1.0: `TP = clamp(100 - 40*(dBTP + 1))`.
- Confirmed clipping means at least three consecutive decoded samples at `abs(x) >= 0.9999`, or a confirmed flat-topped run.
- `CLIP = 100` with no confirmed runs; `CLIP = 0` with any confirmed run.
- Isolated MP3 overs are warnings and do not alone count as confirmed clipping.
- `LUFS = clamp(100 - 5*abs(LUFS-I + 14))`, with a minimum of 60.
- `DC = 100` at <= 0.001; 70 at 0.01; 40 above 0.01; interpolate continuously.
- `FORMAT = 100` for WAV/AIFF/FLAC; 75 for MP3 >= 256 kbps; 60 for lower-bitrate MP3.

## 2. Frequency-balance score

Measure these power proportions:

- Sub, 20-60 Hz: target 10-18%.
- Bass, 60-120 Hz: 20-30%.
- Low-mid, 120-250 Hz: 18-26%.
- Mud, 250-500 Hz: 4-8%.
- Mid, 500 Hz-2 kHz: 15-25%.
- Presence, 2-5 kHz: 7-11%.
- Harsh, 5-8 kHz: 1.3-2.5%.
- Air, 8-16 kHz: 1-2%.

Each band receives 100 inside its range. Outside the range, subtract per percentage point outside:

- Sub: 8 points.
- Bass: 5.
- Low-mid: 7.
- Mud: 20.
- Mid: 10.
- Presence: 20.
- Harsh: 35.
- Air: 60.

Clamp each subscore to 0-100.

Calculate:

`F = 0.15*sub + 0.15*bass + 0.10*low_mid + 0.20*mud + 0.10*mid + 0.15*presence + 0.10*harsh + 0.05*air`

Do not impose an automatic 80-point ceiling.

If reference tracks are supplied, report a separate reference-delta score, but do not replace the fixed `LT_DARKSYNTH_V1` score or alter the ranking.

## 3. Stereo/phase score

Measure:

- Sub correlation across 20-120 Hz.
- Correlation across 120-500 Hz.
- Correlation across 500 Hz-1 kHz.
- Sustained negative correlation in 100 ms frames.
- Mono-sum loss.
- Side/mid RMS ratio.
- L/R RMS imbalance.

Calculate:

`S = 0.30*SUB + 0.25*MONO + 0.20*LOWMID + 0.15*WIDTH + 0.10*BALANCE`

Rules:

- `SUB = 100` at correlation >= 0.90; 0 at <= 0.70; interpolate.
- `LOWMID` is the average score for 120-500 Hz and 500 Hz-1 kHz: 100 at correlation >= 0.50; 0 at <= 0.20; interpolate.
- `MONO = 100` at mono loss >= -1 dB; 0 at <= -3 dB; interpolate.
- `WIDTH = 100` at side/mid <= 1.1; 0 at >= 1.6; interpolate.
- `BALANCE = 100` at L/R imbalance <= 0.5 dB; 0 at >= 3 dB; interpolate.
- Also report global correlation, but do not let it conceal unsafe frequency-specific phase behaviour.

## 4. Dynamics score

Calculate:

`D = 0.60*LRA + 0.25*PSR + 0.15*CREST`

Rules:

- `LRA = 100` between 5 and 8 LU; subtract 25 points per LU outside this range.
- `PSR = 100` at >= 4 dB; subtract 25 points per dB below 4.
- `CREST = 100` at >= 9 dB; subtract 15 points per dB below 9.
- Clamp all subscores to 0-100.
- Report unusually high crest factor or LRA without penalising unless it exceeds the stated range.

## 5. Artifact score

Calculate:

`A = 0.30*START + 0.30*END + 0.25*CLICKS + 0.15*NOISE`

Rules:

- `START = 100` if first-sample amplitude <= 0.02 or a clean fade-in of at least 5 ms exists; 70 between 0.02 and 0.10 without a fade; 20 above 0.10 without a fade.
- `END = 100` if final 50 ms RMS <= -45 dBFS or a clear fade reaches <= -60 dBFS; 60 between -45 and -30 without a proper fade; 20 above -30 without a fade.
- Detect clicks using both absolute sample discontinuity and a robust local outlier test. Do not rely only on `jump > 1.0`.
- `CLICKS = clamp(100 - 15*confirmed_click_count)`.
- `NOISE = 100` at <= -50 dBFS; 80 between -50 and -40; 60 above -40.
- If noise floor is unmeasurable because the track has no genuinely quiet frames, mark it `N/A` and re-normalise the remaining artifact weights.
- Report internal silence gaps separately. Do not penalise them unless discontinuities indicate an accidental edit.

## 6. Objective release-readiness score

Genre fit must not alter technical release-readiness.

Calculate:

`OVERALL = 0.25*T + 0.20*F + 0.20*S + 0.15*D + 0.20*A`

Hard release gates:

- Confirmed digital clipping.
- True peak above -1.0 dBTP.
- Sustained negative correlation for at least 500 ms.
- Mono-sum loss below -3 dB.
- Confirmed boundary click or abrupt cut.
- Lossy file when explicitly analysing a final distribution master.

Verdict:

- `Ready`: OVERALL >= 85, every category >= 70, and no hard gate.
- `Minor work`: OVERALL 75-84, or one easily corrected gate such as true peak/fade.
- `Needs work`: OVERALL 60-74.
- `Significant work`: OVERALL < 60, two or more hard gates, or any reconstruction-level problem.

## 7. Intervention effort and ranking

Assign correction costs:

- Cost 1: gain adjustment, limiter ceiling, short fade or lossless export.
- Cost 2: EQ/dynamic-EQ correction, stereo narrowing or local artifact repair.
- Cost 3: stem remix, dynamics restoration, phase reconstruction or arrangement editing.

Calculate total `effort_points`.

- Low: 0-2.
- Medium: 3-5.
- High: >= 6.

The purpose is to identify which version needs least work. Therefore rank using:

`rank_key = (effort_points ascending, OVERALL descending)`

Do not manually reorder tracks because one seems aesthetically preferable.

## 8. Genre and aesthetic judgement

Report this separately as `INFERRED GENRE FIT`; do not include it in OVERALL or ranking.

Assess only what the available audio analysis can support:

- Low-end weight.
- Percussive strength.
- Metallic/noisy spectral texture.
- Section contrast.
- Repetition and arrangement variation.
- Potential fatigue from mud, harshness or density.

Do not claim to identify guitars, synths, emotional effect, mix pleasure or sound-design quality reliably unless genuine audio listening is available. Explicitly label inference and uncertainty.

## Required output

Start with one short verdict.

Then provide one ranking table containing:

- Rank.
- Track.
- OVERALL.
- Technical.
- Frequency.
- Stereo.
- Dynamics.
- Artifacts.
- LUFS-I / dBTP / LRA.
- Hard gates.
- Main blocker.
- Effort points and Low/Medium/High.

After the table, provide a very short separate aesthetic section.

Show `Rubric: LT_DARKSYNTH_V1`.

Do not give production instructions unless requested. Keep explanations concise and report only the most important blocker per track.
