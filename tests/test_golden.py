"""Golden tests: synthetic WAVs with known defects, checked against the
gates and scores they must trigger."""

import numpy as np
import soundfile as sf

from track_analyzer.analyze import analyze

SR = 44100


def _fade_out(sig, sr, fade_s=0.15):
    n = len(sig)
    nf = int(fade_s * sr)
    out = sig.copy()
    out[n - nf:] *= np.linspace(1.0, 0.0, nf)
    return out


def _write_stereo(path, L, R):
    sf.write(path, np.stack([L, R]).T, SR, subtype="PCM_24")


def clipped_wav(path, dur=1.5, freq=5000, amp=3.0):
    """Hard-clipped sine, high enough in frequency that inter-sample
    peaks blow well past the true-peak gate too."""
    t = np.arange(int(dur * SR)) / SR
    sig = np.clip(amp * np.sin(2 * np.pi * freq * t), -0.9999, 0.9999)
    sig = _fade_out(sig, SR)
    _write_stereo(path, sig, sig)


def inverted_phase_wav(path, dur=1.5, freq=200, amp=0.4):
    """L and R are exact negations of each other — fully anti-phase."""
    t = np.arange(int(dur * SR)) / SR
    L = amp * np.sin(2 * np.pi * freq * t)
    _write_stereo(path, L, -L)


def hard_start_wav(path, dur=1.5, freq=300, amp=0.6):
    """Starts at full amplitude (cosine, no ramp-in) instead of at zero."""
    t = np.arange(int(dur * SR)) / SR
    sig = _fade_out(amp * np.cos(2 * np.pi * freq * t), SR)
    _write_stereo(path, sig, sig)


def test_confirmed_clipping_gates_and_technical_score(tmp_path):
    path = tmp_path / "clipped.wav"
    clipped_wav(str(path))
    r = analyze(str(path))

    assert r["gates"] == ["confirmed_clipping", "true_peak_exceeded"]
    assert r["scores"]["technical"] < 30


def test_inverted_phase_gates_and_stereo_score(tmp_path):
    path = tmp_path / "inverted.wav"
    inverted_phase_wav(str(path))
    r = analyze(str(path))

    assert r["gates"] == ["sustained_negative_correlation", "mono_sum_loss"]
    assert r["scores"]["stereo"] < 30


def test_hard_start_gate(tmp_path):
    path = tmp_path / "hard_start.wav"
    hard_start_wav(str(path))
    r = analyze(str(path))

    assert "hard_start" in r["gates"]
