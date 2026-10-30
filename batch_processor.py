"""
Speech Dataset QA & Conditioning Pipeline
=========================================

A QA-first batch pipeline for preparing speech audio (TTS/ASR training data)
against a defined technical specification.

Design philosophy
-----------------
Processing degrades signal. This pipeline therefore treats measurement as the
primary job and remediation as an exception path. Files are routed by what is
actually wrong with them:

    TIER 0  clean            -> conform loudness/format only (non-destructive)
    TIER 1  correctable      -> VST restoration chain + conform
    TIER 2  structural       -> quarantine for manual repair (never batch-processed)

Structural artifacts (plosives, digital ticks, clipping) alter waveform shape
rather than level. No batch process repairs them without collateral damage, so
they are flagged and held back for human repair, then re-verified through the
same QA gate.

At batch level, if the share of quarantined files exceeds a configured
threshold, the whole batch is recommended for rejection rather than
cherry-picking clean files out of a bad delivery.

Author: <your name>
License: MIT
"""

import os
import re
import json
import base64
import time
import warnings
import subprocess
import logging
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import librosa
import soundfile as sf
from scipy import signal as sps
from scipy.ndimage import median_filter, binary_closing
from pedalboard import Pedalboard, load_plugin
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================
# 1. RUN MODE
# =============================================================
# "process" : normal batch run over INPUT_DIR
# "verify"  : re-QA manually repaired files listed in a repair-queue manifest.
#             Skips the VST chain (the artifact was fixed by hand); still
#             conforms loudness/format and re-runs the full QA gate.
MODE = "process"          # "process" | "verify"


# =============================================================
# 2. PATHS
# =============================================================
INPUT_DIR    = r"./input"
OUTPUT_DIR   = r"./output"
REPAIRED_DIR = r"./repaired"      # where hand-repaired files are placed for verify mode
RPP_PATH     = r"./presets/chain.rpp"

# In verify mode, the repair queue manifest to read (produced by a prior run).
# Leave as None to auto-select the most recent repair_queue_*.json in OUTPUT_DIR.
REPAIR_QUEUE_PATH = None

VALID_EXTENSIONS    = ('.wav', '.m4a', '.mp3', '.flac', '.aiff', '.ogg')
NUM_WORKERS         = 8       # one per physical core
MAX_TASKS_PER_CHILD = 1000    # recycle workers to release VST C++ heap

PLUGIN_PATHS = {
    "pro_q":    r"C:\Program Files\Common Files\VST3\FabFilter Pro-Q 3.vst3",
    "rx8":      r"C:\Program Files\Common Files\VST3\iZotope\RX 8 Voice De-noise.vst3",
    "dxrevive": r"C:\Program Files\Common Files\VST3\Accentize-dxRevivePro.vst3",
}


# =============================================================
# 3. TECHNICAL SPECIFICATION
# Single source of truth. Every threshold and target lives here and is
# embedded verbatim into each delivery manifest, so any manifest is
# self-describing without reference to this file.
# =============================================================
SPEC = {
    # ---------- Output format ----------
    "target_sample_rate":        44100,      # Hz
    "target_bit_depth":          "PCM_16",   # PCM_16 | PCM_24 | PCM_32
    "target_channels":           1,          # mono

    # ---------- Loudness (EBU R128) ----------
    "target_lufs":               -24.0,      # integrated loudness
    "target_true_peak_dbtp":     -0.3,       # headroom for downstream codecs
    "target_lra_lu":             11.0,

    # ---------- Tier 1: correctable by processing ----------
    # Level/noise problems. Non-destructive to fix.
    "min_snr_db":                20.0,       # below -> LOW_SNR
    "max_noise_floor_db":        -40.0,      # above -> HIGH_NOISE_FLOOR

    # ---------- Tier 2: structural, manual repair only ----------
    # Waveform-shape damage. Batch remediation causes collateral degradation.
    "max_clipping_count":        10,         # samples at full scale
    "max_plosive_events":        0,          # any plosive -> quarantine
    "max_tick_events":           0,          # any digital tick -> quarantine

    # ---------- Segmentation sanity ----------
    "max_silence_ratio":         0.60,       # above -> SILENCE_HEAVY
    "min_silence_ratio":         0.03,       # below -> SILENCE_SPARSE

    # ---------- Output conformance tolerances ----------
    "lufs_tolerance_db":         1.0,
    "true_peak_tolerance_db":    0.5,

    # ---------- Capture-type detection ----------
    # G.711 telephony (ITU-T) is band-limited 300-3400 Hz at 8 kHz.
    # Upsampled files keep the cutoff, so bandwidth is checked as well as
    # the declared sample rate.
    "telephonic_bandwidth_hz":   4000,
    "narrowband_bandwidth_hz":   8000,

    # ---------- Batch-level rejection ----------
    # If more than this share of a delivery is quarantined, reject the batch
    # rather than repairing it file by file. Protects against absorbing the
    # cost of a vendor's bad capture setup.
    "batch_reject_threshold":    0.10,       # 10%

    # ---------- Analysis parameters ----------
    "frame_length_ms":           25,
    "hop_length_ms":             10,
    "clipping_threshold":        0.999,
    "noise_floor_percentile":    10,
    "speech_threshold_offset_db": 20.0,

    # ---------- Plosive detector ----------
    # A plosive is a LOW-FREQUENCY burst. Voiced speech (male f0 85-180 Hz)
    # already puts steady energy in that band, so absolute thresholds
    # false-fire. Detection is therefore based on local PROMINENCE: how far
    # low-band dominance rises above its own rolling baseline for this file.
    "plosive_band_hz":           160.0,   # low-band cutoff
    "plosive_ratio_prominence":  0.40,    # primary discriminator
    "plosive_prominence_db":     0.0,     # supporting LF energy prominence
    "plosive_baseline_ms":       500,     # rolling-median baseline window
    "plosive_min_frames":        1,
    "plosive_merge_gap_frames":  3,       # bridge gaps so one burst = one event

    # ---------- Tick / click detector ----------
    # A digital tick is a sample-level discontinuity with no acoustic cause.
    # Detected on the first difference with a robust (MAD) threshold; real
    # transients ramp over many samples, clicks span 1-4.
    "tick_sigma_threshold":      18.0,
    "tick_max_samples":          4,
}

# Flags that route a file to quarantine (manual repair).
TIER_2_FLAGS = {"CLIPPING", "PLOSIVE_DETECTED", "DIGITAL_TICKS"}

# Flags that route a file to the VST restoration chain.
TIER_1_FLAGS = {"LOW_SNR", "HIGH_NOISE_FLOOR"}


# =============================================================
# 4. LOGGING
# =============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_PATH        = os.path.join(OUTPUT_DIR, "processing.log")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, ".completed_files.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# =============================================================
# 5. RPP PLUGIN-STATE EXTRACTOR
# =============================================================
def extract_plugin_states(rpp_path: str) -> list:
    """
    Pull raw VST3 state bytes for each plugin in a Reaper session's master chain.

    Why this exists: Pedalboard's load_preset() only reads the VST3 .vstpreset
    standard. It cannot read FabFilter .ffp, iZotope .xml, or Accentize .dxrp.
    Rather than fight vendor formats, we read state from where Reaper already
    stores it fully dialed -- the .RPP session file itself.

    RPP layout per <VST> block:
        line 0   Reaper header (bypass flags, I/O config) -- not plugin state
        line 1+  base64 plugin state, concatenated
    The decoded blob carries an 8-byte Reaper prefix which is stripped, leaving
    raw VST3 state that Pedalboard accepts via plugin._raw_state.

    Returns states in chain order: [pro_q, rx8_1, rx8_2, dxrevive]
    """
    with open(rpp_path, "r", encoding="utf-8") as f:
        rpp = f.read()

    pattern = (r'<VST "VST3: ([^"]+)"[^\n]+\n'
               r'([\s\S]+?)(?=\n\s+(?:PRESETNAME|FLOATPOS|FXID|BYPASS|WAK|<VST|\s*>))')
    blocks = re.findall(pattern, rpp)

    if len(blocks) != 4:
        raise ValueError(
            f"Expected 4 VST plugins in MASTERFXLIST, found {len(blocks)}. "
            "Verify the full chain is loaded on the Master FX in the RPP."
        )

    states = []
    for name, content in blocks:
        lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
        raw = base64.b64decode("".join(l for l in lines[1:] if l))
        states.append(raw[8:])
        log.info(f"  Extracted state for '{name}': {len(raw[8:])} bytes")
    return states


# =============================================================
# 6. MEASUREMENT
# =============================================================
def _frame_rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Framed RMS. Local helper so detectors stay independent of librosa internals."""
    if len(y) < frame_length:
        return np.array([np.sqrt(np.mean(y ** 2))])
    n = (len(y) - frame_length) // hop_length + 1
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n)[:, None]
    return np.sqrt(np.mean(y[idx] ** 2, axis=1))


def _count_runs(mask: np.ndarray, min_len: int = 1, max_len=None):
    """Count contiguous True runs, optionally filtered by length. Returns (count, starts)."""
    if not np.any(mask):
        return 0, np.array([], dtype=int)
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    lengths = ends - starts
    keep = lengths >= min_len
    if max_len is not None:
        keep &= (lengths <= max_len)
    return int(np.sum(keep)), starts[keep]


def get_audio_metadata(file_path: str) -> dict:
    """Format-agnostic metadata via ffprobe (handles every input extension uniformly)."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", file_path]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout or "{}")
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    duration = audio.get("duration") or data.get("format", {}).get("duration") or 0
    return {
        "sample_rate":      int(audio.get("sample_rate", 0)),
        "channels":         int(audio.get("channels", 0)),
        "duration_seconds": round(float(duration), 3),
        "bit_depth":        audio.get("bits_per_raw_sample") or audio.get("bits_per_sample") or "unknown",
        "codec":            audio.get("codec_name", "unknown"),
    }


def measure_loudness_ffmpeg(file_path: str) -> dict:
    """
    EBU R128 metrics via ffmpeg's ebur128 filter.

    IMPORTANT: ebur128 streams a running per-frame reading on every output line
    during processing. Early frames report the -70 LUFS gate floor before the
    integrated value stabilises. Matching those yields wildly wrong numbers, so
    only the final "Summary:" block is parsed. The regex deliberately does not
    match inf/nan, so digitally silent files return None rather than a floor value.
    """
    cmd = ["ffmpeg", "-hide_banner", "-i", file_path,
           "-af", "ebur128=peak=true", "-f", "null", "-"]
    stderr = subprocess.run(cmd, capture_output=True, text=True).stderr

    idx = stderr.rfind("Summary:")
    if idx == -1:
        return {"integrated_lufs": None, "lra_lu": None, "true_peak_dbtp": None}
    summary = stderr[idx:]

    def _f(m):
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    return {
        "integrated_lufs": _f(re.search(r'I:\s+(-?[\d.]+)\s+LUFS',   summary)),
        "lra_lu":          _f(re.search(r'LRA:\s+(-?[\d.]+)\s+LU',    summary)),
        "true_peak_dbtp":  _f(re.search(r'Peak:\s+(-?[\d.]+)\s+dBFS', summary)),
    }


def detect_ticks(audio: np.ndarray) -> dict:
    """
    Digital tick / click detection.

    A click is a sample-to-sample discontinuity with no acoustic cause -- dropped
    samples, buffer underrun, clock desync. It is broadband and instantaneous,
    unlike a plosive (low-frequency, tens of milliseconds).

    Method: robust MAD-based threshold on the first difference. Real transients,
    even hard consonants, ramp over many samples; clicks span 1-4, so a run-length
    ceiling separates them.
    """
    d = np.diff(audio)
    if len(d) == 0:
        return {"tick_events": 0, "max_tick_sigma": 0.0, "_starts": np.array([], dtype=int)}

    med = np.median(d)
    sigma = 1.4826 * np.median(np.abs(d - med))      # robust std estimate
    if sigma < 1e-9:                                  # digital silence / DC
        return {"tick_events": 0, "max_tick_sigma": 0.0, "_starts": np.array([], dtype=int)}

    dev = np.abs(d - med) / sigma
    count, starts = _count_runs(dev > SPEC["tick_sigma_threshold"],
                                min_len=1, max_len=SPEC["tick_max_samples"])
    return {"tick_events": count, "max_tick_sigma": round(float(np.max(dev)), 1),
            "_starts": starts}


def detect_plosives(audio: np.ndarray, sr: int, tick_starts: np.ndarray) -> dict:
    """
    Plosive detection.

    A plosive is a burst of air on the capsule: low-frequency energy, fast onset,
    tens of milliseconds long. Naive low-band thresholding false-fires constantly
    because voiced speech (male f0 85-180 Hz) already sits in that band.

    Method: measure LOCAL PROMINENCE instead of absolute energy -- how far low-band
    dominance rises above its own rolling-median baseline within this file. Steady
    voiced energy lifts the baseline along with the signal and cancels out; a
    plosive spikes above it.

    Frames containing a detected digital tick are excluded: a click is broadband
    and leaks into the low band, but it is a different artifact with a different
    remedy, and double-counting it would distort the tier routing.
    """
    sos = sps.butter(4, SPEC["plosive_band_hz"], btype='low', fs=sr, output='sos')
    lf = sps.sosfilt(sos, audio)

    fl = int(sr * SPEC["frame_length_ms"] / 1000)
    hp = int(sr * SPEC["hop_length_ms"] / 1000)
    lf_rms, full_rms = _frame_rms(lf, fl, hp), _frame_rms(audio, fl, hp)
    n_frames = len(lf_rms)
    if n_frames < 5:
        return {"plosive_events": 0, "max_lf_prominence_db": 0.0}

    ratio = lf_rms / (full_rms + 1e-10)
    lf_db = 20 * np.log10(lf_rms + 1e-10)

    win = max(3, int(SPEC["plosive_baseline_ms"] / SPEC["hop_length_ms"]) | 1)
    lf_prom    = lf_db - median_filter(lf_db, size=win, mode='nearest')
    ratio_prom = ratio - median_filter(ratio, size=win, mode='nearest')

    mask = ((lf_prom > SPEC["plosive_prominence_db"]) &
            (ratio_prom > SPEC["plosive_ratio_prominence"]))

    # Bridge short gaps FIRST so one physical burst counts once.
    gap = SPEC["plosive_merge_gap_frames"]
    if gap > 0 and np.any(mask):
        mask = binary_closing(mask, structure=np.ones(gap + 1))

    # Exclude tick-contaminated frames AFTER closing -- otherwise closing
    # bridges straight over the frames just excluded.
    if len(tick_starts) > 0:
        tick_frames = np.zeros(n_frames, dtype=bool)
        for ts in tick_starts:
            lo = max(0, (ts - fl) // hp)
            hi = min(n_frames - 1, ts // hp + 1)
            tick_frames[lo:hi + 1] = True
        mask &= ~tick_frames

    count, _ = _count_runs(mask, min_len=SPEC["plosive_min_frames"])
    return {"plosive_events": count,
            "max_lf_prominence_db": round(float(np.max(lf_prom)), 1)}


def compute_signal_qa(audio: np.ndarray, sr: int) -> dict:
    """
    Signal-level QA metrics computed in numpy (no subprocess).

    Returns clipping count, noise floor, silence ratio, SNR, effective bandwidth,
    plus plosive and tick event counts.
    """
    fl = int(sr * SPEC["frame_length_ms"] / 1000)
    hp = int(sr * SPEC["hop_length_ms"] / 1000)

    clipping_count = int(np.sum(np.abs(audio) > SPEC["clipping_threshold"]))

    rms = _frame_rms(audio, fl, hp)
    rms_db = 20 * np.log10(rms + 1e-10)

    noise_floor_db = float(np.percentile(rms_db, SPEC["noise_floor_percentile"]))
    speech_mask = rms_db >= (noise_floor_db + SPEC["speech_threshold_offset_db"])
    silence_ratio = float(np.sum(~speech_mask) / len(rms_db))

    # SNR: mean power of speech frames vs silence frames.
    # None for degenerate files (all speech or all silence) rather than a fake number.
    if np.any(speech_mask) and np.any(~speech_mask):
        snr_db = float(10 * np.log10(np.mean(rms[speech_mask] ** 2) /
                                     (np.mean(rms[~speech_mask] ** 2) + 1e-10)))
    else:
        snr_db = None

    # Effective bandwidth: frequency below which 99% of spectral energy sits.
    # Survives upsampling, so it catches telephony re-encoded to 44.1 kHz.
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    cum_e = np.cumsum(fft ** 2)
    if cum_e[-1] > 0:
        idx = np.searchsorted(cum_e, 0.99 * cum_e[-1])
        effective_bandwidth = float(freqs[min(idx, len(freqs) - 1)])
    else:
        effective_bandwidth = 0.0

    # Ticks first -- their positions feed the plosive detector's exclusion mask.
    tick_res = detect_ticks(audio)
    plosive_res = detect_plosives(audio, sr, tick_res.pop("_starts"))

    return {
        "clipping_count":         clipping_count,
        "noise_floor_db":         round(noise_floor_db, 2),
        "silence_ratio":          round(silence_ratio, 3),
        "snr_db":                 round(snr_db, 2) if snr_db is not None else None,
        "effective_bandwidth_hz": round(effective_bandwidth, 1),
        **plosive_res,
        **tick_res,
    }


def detect_capture_type(effective_bandwidth_hz: float, native_sr: int) -> str:
    """
    Classify capture environment: 'telephonic' | 'narrowband' | 'broadband'.

    Two signals, because either alone is insufficient. A declared 8 kHz rate is
    definitive per G.711, but a file upsampled to 44.1 kHz upstream reports a
    full-band rate while retaining its ~3.4 kHz cutoff. Bandwidth catches those.
    """
    if native_sr <= 8000 or effective_bandwidth_hz < SPEC["telephonic_bandwidth_hz"]:
        return "telephonic"
    if effective_bandwidth_hz < SPEC["narrowband_bandwidth_hz"]:
        return "narrowband"
    return "broadband"


def generate_flags(qa: dict, capture_type: str) -> list:
    """Compare metrics to SPEC. Flags are independent and additive."""
    flags = []

    # Tier 2 -- structural
    if qa.get("clipping_count", 0) > SPEC["max_clipping_count"]:
        flags.append("CLIPPING")
    if qa.get("plosive_events", 0) > SPEC["max_plosive_events"]:
        flags.append("PLOSIVE_DETECTED")
    if qa.get("tick_events", 0) > SPEC["max_tick_events"]:
        flags.append("DIGITAL_TICKS")

    # Tier 1 -- correctable
    snr = qa.get("snr_db")
    if snr is not None and snr < SPEC["min_snr_db"]:
        flags.append("LOW_SNR")
    if qa.get("noise_floor_db", -99.0) > SPEC["max_noise_floor_db"]:
        flags.append("HIGH_NOISE_FLOOR")

    # Informational
    sil = qa.get("silence_ratio", 0)
    if sil > SPEC["max_silence_ratio"]:
        flags.append("SILENCE_HEAVY")
    elif sil < SPEC["min_silence_ratio"]:
        flags.append("SILENCE_SPARSE")

    if capture_type == "telephonic":
        flags.append("TELEPHONIC_CAPTURE")
    elif capture_type == "narrowband":
        flags.append("NARROWBAND_CAPTURE")

    return flags


def classify_tier(flags: list) -> int:
    """
    Route a file by severity.
        2 -> structural damage: quarantine for manual repair
        1 -> correctable: VST restoration chain
        0 -> clean: conform only
    """
    if TIER_2_FLAGS.intersection(flags):
        return 2
    if TIER_1_FLAGS.intersection(flags):
        return 1
    return 0


def verify_output_conformance(loudness: dict, meta: dict) -> dict:
    """Confirm the written file actually meets SPEC. Never assume the encoder complied."""
    lufs, tp = loudness.get("integrated_lufs"), loudness.get("true_peak_dbtp")
    lufs_delta = abs(lufs - SPEC["target_lufs"]) if lufs is not None else None
    tp_delta   = abs(tp - SPEC["target_true_peak_dbtp"]) if tp is not None else None
    return {
        "lufs_delta":       round(lufs_delta, 2) if lufs_delta is not None else None,
        "lufs_pass":        bool(lufs_delta is not None and lufs_delta <= SPEC["lufs_tolerance_db"]),
        "true_peak_pass":   bool(tp_delta   is not None and tp_delta   <= SPEC["true_peak_tolerance_db"]),
        "sample_rate_pass": meta.get("sample_rate") == SPEC["target_sample_rate"],
        "channels_pass":    meta.get("channels")    == SPEC["target_channels"],
    }


# =============================================================
# 7. LOUDNESS NORMALIZATION
# =============================================================
def normalize_loudness(input_wav: str, output_wav: str) -> bool:
    """
    Two-pass EBU R128 normalization.

    Single-pass loudnorm estimates loudness in one shot and drifts 1-3 LU on
    dynamic material like speech. Two-pass measures true values first
    (print_format=json), then re-runs with those values and linear=true so the
    filter applies a precise linear gain instead of a blind estimate.

    Falls back to single-pass if measurement fails, so no file is dropped.
    """
    measure_cmd = [
        "ffmpeg", "-hide_banner", "-i", input_wav,
        "-af", (f"loudnorm=I={SPEC['target_lufs']}"
                f":TP={SPEC['target_true_peak_dbtp']}"
                f":LRA={SPEC['target_lra_lu']}:print_format=json"),
        "-f", "null", "-",
    ]
    stderr = subprocess.run(measure_cmd, capture_output=True, text=True).stderr
    match = re.search(r'\{[^{}]*"input_i"[^{}]*\}', stderr, re.DOTALL)
    if not match:
        return _normalize_single_pass(input_wav, output_wav)

    m = json.loads(match.group(0))
    apply_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_wav,
        "-af", (f"loudnorm=I={SPEC['target_lufs']}"
                f":TP={SPEC['target_true_peak_dbtp']}"
                f":LRA={SPEC['target_lra_lu']}"
                f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
                f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
                f":linear=true"),
        "-ar", str(SPEC["target_sample_rate"]),
        "-ac", str(SPEC["target_channels"]),
        output_wav,
    ]
    return subprocess.run(apply_cmd, capture_output=True, text=True).returncode == 0


def _normalize_single_pass(input_wav: str, output_wav: str) -> bool:
    """Fallback used only when the measurement pass fails."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_wav,
        "-af", (f"loudnorm=I={SPEC['target_lufs']}"
                f":TP={SPEC['target_true_peak_dbtp']}"
                f":LRA={SPEC['target_lra_lu']}"),
        "-ar", str(SPEC["target_sample_rate"]),
        "-ac", str(SPEC["target_channels"]),
        output_wav,
    ]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


# =============================================================
# 8. CHECKPOINT & OUTPUT PATHS
# =============================================================
def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_PATH):
        return set()
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_checkpoint(file_path: str):
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")


def resolve_output_path(output_dir: str, base_name: str) -> str:
    """`{base}_processed.wav`, incrementing `_1`, `_2` ... on collision."""
    candidate = os.path.join(output_dir, f"{base_name}_processed.wav")
    if not os.path.exists(candidate):
        return candidate
    n = 1
    while True:
        candidate = os.path.join(output_dir, f"{base_name}_processed_{n}.wav")
        if not os.path.exists(candidate):
            return candidate
        n += 1


# =============================================================
# 9. WORKER
# =============================================================
_board = None


def worker_init(plugin_states):
    """
    Runs once per worker process (and again after each recycle).

    Every worker builds its OWN plugin instances from the extracted state bytes.
    Nothing is shared across processes, so there is no locking and no contention.
    """
    global _board
    if plugin_states is None:          # verify mode: no VST chain needed
        _board = None
        return
    pro_q_s, rx1_s, rx2_s, dx_s = plugin_states
    pro_q    = load_plugin(PLUGIN_PATHS["pro_q"])
    rx8_1    = load_plugin(PLUGIN_PATHS["rx8"])
    rx8_2    = load_plugin(PLUGIN_PATHS["rx8"])
    dxrevive = load_plugin(PLUGIN_PATHS["dxrevive"])
    pro_q._raw_state, rx8_1._raw_state = pro_q_s, rx1_s
    rx8_2._raw_state, dxrevive._raw_state = rx2_s, dx_s
    _board = Pedalboard([pro_q, rx8_1, rx8_2, dxrevive])


def _apply_vst_chain(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Run the restoration chain with latency compensation.

    One second of padding absorbs plugin lookahead so the tail is not truncated;
    afterwards the first non-silent sample locates the introduced latency and the
    signal is trimmed back to its original length.
    """
    original_length = len(audio)
    padded = np.pad(audio, (0, sr), mode='constant')
    processed = np.squeeze(_board.process(np.expand_dims(padded, axis=0), sr))
    non_silent = np.where(np.abs(processed) > 1e-4)[0]
    offset = non_silent[0] if len(non_silent) > 0 else 0
    return processed[offset:min(offset + original_length, len(processed))]


def process_file(args) -> dict:
    """
    Full per-file pipeline.

        1  input metadata (ffprobe)
        2  input loudness (ebur128)
        3  load + signal QA (incl. plosive & tick detection)
        4  capture-type detection
        5  flags -> tier classification
        6  route:  tier 2 -> quarantine (no processing)
                   tier 1 -> VST chain + conform
                   tier 0 -> conform only
        7  two-pass loudness normalization
        8  output verification against SPEC

    Returns a report dict. Exceptions are captured per file so one bad input
    never terminates the batch.
    """
    file_path, verify_mode = args
    t0 = time.time()
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]
    temp_wav = os.path.join(OUTPUT_DIR, f"temp_{os.getpid()}_{base_name}.wav")

    report = {
        "file": filename, "input_path": file_path, "output_path": None,
        "status": "failed", "tier": None, "action": None, "flags": [],
        "processing_time_seconds": None,
        "input":  {"metadata": {}, "capture_type": None, "qa_metrics": {}},
        "output": {"metadata": {}, "qa_metrics": {}, "spec_conformance": {}},
        "error": None,
    }

    try:
        report["input"]["metadata"] = get_audio_metadata(file_path)
        input_loudness = measure_loudness_ffmpeg(file_path)

        audio, _ = librosa.load(file_path, sr=SPEC["target_sample_rate"], mono=False)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        qa = compute_signal_qa(audio, SPEC["target_sample_rate"])
        capture_type = detect_capture_type(qa["effective_bandwidth_hz"],
                                           report["input"]["metadata"]["sample_rate"])
        report["input"]["capture_type"] = capture_type
        report["input"]["qa_metrics"] = {**qa, **input_loudness}

        flags = generate_flags(qa, capture_type)
        report["flags"] = flags
        tier = classify_tier(flags)
        report["tier"] = tier

        # --- Tier 2: structural damage. Hold back for manual repair. ---
        # Processing cannot fix waveform-shape damage without collateral loss,
        # so nothing is written and the file enters the repair queue.
        if tier == 2 and not verify_mode:
            report["status"] = "quarantined"
            report["action"] = "manual_repair_required"
            report["processing_time_seconds"] = round(time.time() - t0, 2)
            return report

        # --- Tier 1: correctable. Apply the restoration chain. ---
        if tier == 1 and not verify_mode and _board is not None:
            audio = _apply_vst_chain(audio, SPEC["target_sample_rate"])
            report["action"] = "vst_chain_applied"
        else:
            # Tier 0, or verify mode (artifact already fixed by hand).
            report["action"] = "conform_only"

        sf.write(temp_wav, audio, SPEC["target_sample_rate"], subtype=SPEC["target_bit_depth"])
        out_path = resolve_output_path(OUTPUT_DIR, base_name)

        if not normalize_loudness(temp_wav, out_path):
            raise RuntimeError("Loudness normalization failed.")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        out_loudness = measure_loudness_ffmpeg(out_path)
        out_meta = get_audio_metadata(out_path)
        conformance = verify_output_conformance(out_loudness, out_meta)

        report["output_path"] = out_path
        report["output"]["metadata"] = out_meta
        report["output"]["qa_metrics"] = out_loudness
        report["output"]["spec_conformance"] = conformance

        for key, flag in (("lufs_pass", "OUTPUT_LUFS_FAIL"),
                          ("true_peak_pass", "OUTPUT_TP_FAIL"),
                          ("sample_rate_pass", "OUTPUT_SR_FAIL"),
                          ("channels_pass", "OUTPUT_CHANNEL_FAIL")):
            if not conformance.get(key):
                report["flags"].append(flag)

        report["status"] = "ok_with_flags" if report["flags"] else "ok"
        save_checkpoint(file_path)

    except Exception as exc:
        report["error"] = str(exc)
        report["status"] = "failed"
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

    report["processing_time_seconds"] = round(time.time() - t0, 2)
    return report


# =============================================================
# 10. BATCH ASSESSMENT
# =============================================================
def assess_batch(reports: list) -> dict:
    """
    Batch-level accept/reject recommendation.

    Rationale: if structural artifacts are widespread, the problem is the
    vendor's capture setup, not the individual files. Repairing them one by one
    absorbs a cost that belongs upstream, so the delivery is returned instead.
    A numeric threshold makes that a policy rather than a per-batch judgement call.
    """
    total = len(reports)
    if total == 0:
        return {"recommendation": "NO_FILES", "quarantine_rate": 0.0}

    quarantined = sum(1 for r in reports if r["status"] == "quarantined")
    failed = sum(1 for r in reports if r["status"] == "failed")
    rate = quarantined / total
    threshold = SPEC["batch_reject_threshold"]

    if rate > threshold:
        recommendation = "REJECT_BATCH"
        rationale = (f"{rate:.1%} of files carry structural artifacts, exceeding the "
                     f"{threshold:.0%} threshold. Return to vendor; the capture setup "
                     f"should be corrected rather than the files repaired.")
    elif quarantined > 0:
        recommendation = "ACCEPT_WITH_MANUAL_REPAIR"
        rationale = (f"{quarantined} of {total} files ({rate:.1%}) need manual repair, "
                     f"within the {threshold:.0%} threshold. See the repair queue.")
    else:
        recommendation = "ACCEPT"
        rationale = "No structural artifacts detected."

    return {
        "recommendation": recommendation,
        "rationale": rationale,
        "quarantine_rate": round(rate, 4),
        "quarantine_threshold": threshold,
        "quarantined_count": quarantined,
        "failed_count": failed,
        "total_files": total,
    }


def count_flags(reports: list) -> dict:
    counts = {}
    for r in reports:
        for f in r.get("flags", []):
            counts[f] = counts.get(f, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def write_repair_queue(reports: list, timestamp: str) -> str:
    """
    Write the manual-repair queue.

    Quarantined files are tracked in their own manifest rather than by an ad-hoc
    flag, so the repair cycle is auditable: this file records exactly what was
    held back and why. After repair, set MODE = "verify" to re-run the same QA
    gate over the corrected files -- hand fixes are verified, not trusted.
    """
    quarantined = [r for r in reports if r["status"] == "quarantined"]
    if not quarantined:
        return None

    path = os.path.join(OUTPUT_DIR, f"repair_queue_{timestamp}.json")
    payload = {
        "generated_at": datetime.now().isoformat(),
        "count": len(quarantined),
        "instructions": (
            "Repair these files by hand (e.g. iZotope RX de-plosive / de-click), "
            f"place the corrected files in REPAIRED_DIR, set MODE='verify' and "
            "re-run to re-QA them against the same spec."
        ),
        "files": [
            {
                "file": r["file"],
                "input_path": r["input_path"],
                "flags": r["flags"],
                "plosive_events": r["input"]["qa_metrics"].get("plosive_events"),
                "tick_events":    r["input"]["qa_metrics"].get("tick_events"),
                "clipping_count": r["input"]["qa_metrics"].get("clipping_count"),
            }
            for r in quarantined
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# =============================================================
# 11. MAIN
# =============================================================
def _collect_process_files() -> list:
    if not os.path.isdir(INPUT_DIR):
        log.error(f"Input directory not found: {INPUT_DIR}")
        return []
    return [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(VALID_EXTENSIONS)]


def _collect_verify_files() -> list:
    """Resolve repaired files listed in the repair-queue manifest."""
    queue_path = REPAIR_QUEUE_PATH
    if queue_path is None:
        candidates = sorted(f for f in os.listdir(OUTPUT_DIR)
                            if f.startswith("repair_queue_") and f.endswith(".json"))
        if not candidates:
            log.error("No repair_queue_*.json found. Run in 'process' mode first.")
            return []
        queue_path = os.path.join(OUTPUT_DIR, candidates[-1])

    log.info(f"Reading repair queue: {queue_path}")
    with open(queue_path, "r", encoding="utf-8") as f:
        queue = json.load(f)

    files = []
    for entry in queue.get("files", []):
        repaired = os.path.join(REPAIRED_DIR, entry["file"])
        if os.path.exists(repaired):
            files.append(repaired)
        else:
            log.warning(f"Repaired file not yet present, skipping: {entry['file']}")
    return files


def main():
    verify_mode = (MODE == "verify")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("=" * 62)
    log.info(f"Speech Dataset QA Pipeline  |  MODE = {MODE}")
    log.info(f"Output:  {OUTPUT_DIR}")
    log.info(f"Workers: {NUM_WORKERS}  |  max tasks/child: {MAX_TASKS_PER_CHILD}")
    log.info("=" * 62)

    # Plugin states are only needed when the restoration chain may run.
    plugin_states = None
    if not verify_mode:
        if not os.path.exists(RPP_PATH):
            log.error(f"RPP not found: {RPP_PATH}")
            return
        log.info("Extracting plugin states from RPP...")
        try:
            plugin_states = extract_plugin_states(RPP_PATH)
        except Exception as exc:
            log.error(f"Failed to extract plugin states: {exc}")
            return

    files = _collect_verify_files() if verify_mode else _collect_process_files()
    if not files:
        log.warning("No files to process.")
        return

    completed = load_checkpoint()
    pending = [f for f in files if f not in completed]
    log.info(f"Found {len(files)} files  |  already done {len(completed)}  |  pending {len(pending)}")
    if not pending:
        log.info("Nothing to do.")
        return

    reports = []
    counters = {"ok": 0, "quarantined": 0, "failed": 0}

    with Pool(processes=NUM_WORKERS, initializer=worker_init,
              initargs=(plugin_states,), maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
        jobs = [(f, verify_mode) for f in pending]
        with tqdm(total=len(jobs), desc="Processing", unit="file", dynamic_ncols=True) as bar:
            for report in pool.imap_unordered(process_file, jobs):
                reports.append(report)
                status = report["status"]
                if status == "quarantined":
                    counters["quarantined"] += 1
                    log.warning(f"QUARANTINE  {report['file']}  {report['flags']}")
                elif status == "failed":
                    counters["failed"] += 1
                    log.error(f"FAIL        {report['file']}  {report['error']}")
                else:
                    counters["ok"] += 1
                    suffix = f"  {report['flags']}" if report["flags"] else ""
                    log.info(f"OK  T{report['tier']}  {report['file']}{suffix}")
                bar.update(1)
                bar.set_postfix(**counters)

    # ---- Batch assessment & artifacts ----
    assessment = assess_batch(reports)
    repair_queue_path = write_repair_queue(reports, timestamp) if not verify_mode else None

    manifest_path = os.path.join(OUTPUT_DIR, f"delivery_manifest_{timestamp}.json")
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "mode": MODE,
        "spec": SPEC,
        "batch_assessment": assessment,
        "summary": {
            "total": len(reports),
            "ok": counters["ok"],
            "quarantined": counters["quarantined"],
            "failed": counters["failed"],
            "flagged": sum(1 for r in reports if r["flags"]),
            "tier_breakdown": {
                "tier_0_clean":       sum(1 for r in reports if r["tier"] == 0),
                "tier_1_processed":   sum(1 for r in reports if r["tier"] == 1),
                "tier_2_quarantined": sum(1 for r in reports if r["tier"] == 2),
            },
            "flag_breakdown": count_flags(reports),
        },
        "files": reports,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log.info("=" * 62)
    log.info(f"Batch complete.  ok={counters['ok']}  "
             f"quarantined={counters['quarantined']}  failed={counters['failed']}")
    log.info(f"Assessment: {assessment['recommendation']}")
    log.info(f"  {assessment['rationale']}")
    log.info(f"Manifest -> {manifest_path}")
    if repair_queue_path:
        log.info(f"Repair queue -> {repair_queue_path}")
    log.info("=" * 62)


if __name__ == "__main__":
    main()
