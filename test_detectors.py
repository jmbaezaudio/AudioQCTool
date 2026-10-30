"""
Detector tests on synthetic signals with known artifact counts.

These double as documentation: each test constructs a signal containing a
specific, countable defect and asserts the detector finds exactly that many.

Run:  python tests/test_detectors.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import detectors without pulling in pedalboard (not needed for these tests).
import types
for stub in ("pedalboard", "librosa", "soundfile", "tqdm"):
    if stub not in sys.modules:
        m = types.ModuleType(stub)
        if stub == "pedalboard":
            m.Pedalboard, m.load_plugin = object, lambda *a, **k: None
        if stub == "tqdm":
            m.tqdm = lambda *a, **k: None
        sys.modules[stub] = m

from batch_processor import detect_ticks, detect_plosives, detect_capture_type, SPEC

SR = 44100
RESULTS = []


def speech_like(duration=3.0, f0=120, sr=SR, seed=0):
    """Voiced-speech surrogate: harmonic stack with a syllabic amplitude envelope."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    harmonics = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 12))
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)) ** 2
    return harmonics * envelope * 0.08 + 0.001 * rng.standard_normal(len(t))


def add_plosives(sig, positions_s, sr=SR):
    """Low-frequency pressure burst with fast attack and exponential decay."""
    out = sig.copy()
    for pos in positions_s:
        i, dur = int(pos * sr), int(0.06 * sr)
        env = np.exp(-np.linspace(0, 6, dur))
        out[i:i + dur] += 0.5 * np.sin(2 * np.pi * 70 * np.arange(dur) / sr) * env
    return out


def add_ticks(sig, positions_s, sr=SR, seed=0):
    """Single-sample discontinuities -- the signature of dropped samples."""
    rng = np.random.default_rng(seed)
    out = sig.copy()
    for pos in positions_s:
        out[int(pos * sr)] += 0.6 * (1 if rng.random() > 0.5 else -1)
    return out


def check(name, got, expected):
    ok = got == expected
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, expected {expected}")


def run():
    print("\n--- Tick detection ---")
    clean = speech_like()
    ticks = detect_ticks(clean)
    check("clean speech -> no ticks", ticks["tick_events"], 0)

    sig = add_ticks(clean, [0.2, 0.9, 1.5, 2.0, 2.6])
    check("5 injected ticks", detect_ticks(sig)["tick_events"], 5)

    check("digital silence -> no ticks", detect_ticks(np.zeros(SR))["tick_events"], 0)

    print("\n--- Plosive detection ---")
    t_clean = detect_ticks(clean)
    check("clean speech -> no plosives",
          detect_plosives(clean, SR, t_clean["_starts"])["plosive_events"], 0)

    # A higher fundamental sits above the low band; must also stay clean.
    high = speech_like(f0=210)
    check("higher-f0 voice -> no plosives",
          detect_plosives(high, SR, detect_ticks(high)["_starts"])["plosive_events"], 0)

    plos = add_plosives(clean, [0.4, 1.3, 2.2])
    check("3 injected plosives",
          detect_plosives(plos, SR, detect_ticks(plos)["_starts"])["plosive_events"], 3)

    print("\n--- Cross-talk isolation ---")
    # Ticks are broadband and leak into the low band. The detectors must not
    # double-count one artifact as the other.
    both = add_ticks(add_plosives(clean, [0.4, 1.3, 2.2]), [0.7, 1.8], seed=1)
    tb = detect_ticks(both)
    check("mixed signal -> 2 ticks", tb["tick_events"], 2)
    check("mixed signal -> 3 plosives",
          detect_plosives(both, SR, tb["_starts"])["plosive_events"], 3)

    tick_only = add_ticks(clean, [0.2, 0.9, 1.5, 2.0, 2.6])
    check("ticks alone -> 0 plosives",
          detect_plosives(tick_only, SR, detect_ticks(tick_only)["_starts"])["plosive_events"], 0)

    print("\n--- Capture-type classification ---")
    check("broadband", detect_capture_type(12000, 44100), "broadband")
    check("native 8 kHz telephony", detect_capture_type(3400, 8000), "telephonic")
    # The important case: upsampled telephony lies about its rate but keeps its cutoff.
    check("telephony upsampled to 44.1 kHz", detect_capture_type(3400, 44100), "telephonic")

    print("\n" + "=" * 46)
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    print("=" * 46)
    return all(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
