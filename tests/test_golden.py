"""Golden tests: synthetic WAVs with known defects, checked against the
gates and scores they must trigger."""

import ast
import inspect

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from track_analyzer import score
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


def known_good_wav(path, dur=10.0, fade_s=0.05, target_lufs=-14.0):
    """Matched channels, 50ms linear fade in/out, normalised to -14 LUFS.
    Tones plus low-level broadband noise so every band has energy (a
    pure tone would trip the lossy-content spectral-cliff gate)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(0)
    sig = (0.5 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)
           + 0.15 * np.sin(2 * np.pi * 1000 * t) + 0.05 * rng.standard_normal(n))
    nf = int(fade_s * SR)
    sig[:nf] *= np.linspace(0.0, 1.0, nf)
    sig[-nf:] *= np.linspace(1.0, 0.0, nf)
    stereo = np.stack([sig, sig]).T
    meter = pyln.Meter(SR)
    stereo = pyln.normalize.loudness(stereo, meter.integrated_loudness(stereo), target_lufs)
    sf.write(path, stereo, SR, subtype="PCM_24")


def test_known_good_fires_no_gates(tmp_path):
    path = tmp_path / "known_good.wav"
    known_good_wav(str(path))
    r = analyze(str(path), final=True)  # final: also check format/sample-rate gates

    # Guard the fixture itself, so a passing gate check means something.
    loud = r["raw"]["loudness"]
    assert abs(loud["integrated"] - (-14.0)) < 0.5
    assert loud["true_peak"] < -3.0

    assert r["gates"] == []
    # Approximate: measured LUFS sits a hair off -14.000 and the LUFS penalty is linear from 0.
    assert r["scores"]["technical"] > 99.9
    assert r["scores"]["stereo"] > 99.9
    assert r["rubric_version"] == "v2"


# ── Rubric consistency ───────────────────────────────────────────────────────

def _emitted_gates():
    """Gate names evaluate_gates() can append, read from its source."""
    tree = ast.parse(inspect.getsource(score.evaluate_gates))
    return {node.args[0].value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "append"
            and node.args and isinstance(node.args[0], ast.Constant)}


def _emitted_warning_codes():
    """Warning codes the score_* functions can emit: the string second
    element of (int priority, code, message) issue tuples, plus codes chosen
    through a `code = "x" if ... else None` assignment."""
    tree = ast.parse(inspect.getsource(score))
    codes = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Tuple) and len(node.elts) == 3
                and isinstance(node.elts[0], ast.Constant) and isinstance(node.elts[0].value, int)
                and isinstance(node.elts[1], ast.Constant) and isinstance(node.elts[1].value, str)):
            codes.add(node.elts[1].value)
        if (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "code" for t in node.targets)
                and isinstance(node.value, ast.IfExp)):
            for branch in (node.value.body, node.value.orelse):
                if isinstance(branch, ast.Constant) and isinstance(branch.value, str):
                    codes.add(branch.value)
    return codes


def test_bundled_rubric_is_consistent():
    rubric = score.load_rubric(score.DEFAULT_RUBRIC_PATH)

    assert rubric.get("version"), "rubric has no version (reported as rubric_version)"
    assert abs(sum(rubric["weights"].values()) - 1.0) < 1e-9

    gate_costs = rubric["effort"]["gate_costs"]
    gates = _emitted_gates()
    assert gates, "found no gates in evaluate_gates()"
    assert set(gate_costs) - gates == set(), "cost table lists gates evaluate_gates() never fires"
    assert gates - set(gate_costs) == set(), "gates with no effort cost"

    warning_costs = rubric["effort"]["warning_costs"]
    codes = _emitted_warning_codes()
    assert codes, "found no warning codes in score.py"
    assert codes - set(warning_costs) == set(), "warning codes with no effort cost"
    assert set(warning_costs) - codes == set(), "cost table lists warning codes nothing emits"
