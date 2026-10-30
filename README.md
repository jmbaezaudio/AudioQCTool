# Speech Dataset QA & Conditioning Pipeline

A parallel batch pipeline that validates and conditions speech audio against a
defined technical specification, for TTS/ASR training datasets.

It measures every file, routes it by what is actually wrong with it, applies the
minimum necessary processing, verifies the result, and emits a self-describing
delivery manifest.

---

## Why it's built this way

Processing degrades signal. A pipeline that runs every file through a
restoration chain "just in case" quietly damages the clean 90% to fix the
broken 10%.

So this is **QA-first**: measurement is the primary job and remediation is an
exception path. Files are routed by severity:

| Tier | Condition | Action |
|------|-----------|--------|
| **0** | Meets spec | Conform loudness/format only — non-destructive |
| **1** | Correctable (low SNR, high noise floor) | VST restoration chain, then conform |
| **2** | Structural (plosives, digital ticks, clipping) | **Quarantine** — never batch-processed |

Tier 2 artifacts alter waveform *shape*, not level. No batch process repairs
them without collateral damage, so they are held back for manual repair and
then re-verified through the same QA gate. Hand fixes are verified, not trusted.

At batch level, if the quarantine rate exceeds a configured threshold (default
10%), the pipeline recommends **rejecting the whole delivery** rather than
cherry-picking clean files out of it. Widespread structural damage is a capture
problem, and repairing it file by file absorbs a cost that belongs upstream.

---

## What it measures

**Format & loudness** — sample rate, bit depth, channel count, duration, codec;
integrated loudness (LUFS), loudness range (LRA), true peak (dBTP) via EBU R128.

**Signal quality** — SNR, noise floor, silence ratio, clipping count.

**Structural artifacts**

- **Digital ticks** — sample-level discontinuities from dropped samples, buffer
  underruns, or clock desync. Detected via a robust (MAD) threshold on the first
  difference. Real transients ramp over many samples; clicks span 1–4, so a
  run-length ceiling separates them.

- **Plosives** — low-frequency pressure bursts from air hitting the capsule.
  Naive low-band thresholding false-fires constantly, because voiced speech
  (male f0 85–180 Hz) already sits in that band. Detection instead measures
  **local prominence**: how far low-band dominance rises above its own rolling
  baseline *within that file*. Steady voiced energy lifts the baseline along with
  the signal and cancels out; a plosive spikes above it.

  Ticks are detected first and their frames excluded from plosive analysis —
  a click is broadband and leaks into the low band, but it is a different
  artifact with a different remedy.

**Capture type** — `broadband` / `narrowband` / `telephonic`, from effective
bandwidth (the frequency below which 99% of spectral energy sits) *and* the
declared sample rate. Two signals are needed because either alone is
insufficient: G.711 telephony upsampled to 44.1 kHz upstream reports a
full-band rate while retaining its ~3.4 kHz cutoff. Bandwidth catches those.

---

## Implementation notes

**Plugin settings are read from a Reaper session, not preset files.**
Pedalboard's `load_preset()` only reads the VST3 `.vstpreset` standard — not
FabFilter `.ffp`, iZotope `.xml`, or Accentize `.dxrp`. Rather than fight vendor
formats, the pipeline parses the `.RPP` session file and extracts raw VST3 state
bytes directly from the base64 blocks Reaper stores in `MASTERFXLIST`
(skipping Reaper's header line and stripping an 8-byte prefix).

Practical upshot: dial settings in Reaper, save the session, point the script at
it. No preset export step, and the session is the single source of truth.

**Loudness normalization is two-pass.** Single-pass `loudnorm` estimates
loudness in one shot and drifts 1–3 LU on dynamic material like speech. Pass one
measures true values (`print_format=json`); pass two re-runs with those values
and `linear=true` for a precise linear gain. Falls back to single-pass if
measurement fails, so no file is dropped.

**Loudness is parsed from the summary block only.** `ebur128` streams a running
per-frame reading on every output line; early frames report the −70 LUFS gate
floor before the integrated value stabilises. Matching those yields readings tens
of LU wrong. Only the final `Summary:` block is parsed.

**Parallelism.** One worker per physical core, each building its own plugin
instances from the extracted state — nothing shared across processes, so no
locking or contention. `maxtasksperchild` recycles workers periodically to
release the C++ heap that VST plugins accumulate and Python's GC cannot reach.

**Resilience.** Per-file exception capture (one bad input never kills the batch),
checkpointing for crash resume, and collision-safe output naming.

---

## Output

**`delivery_manifest_<timestamp>.json`** — the delivery artifact. Contains the
full spec used, a batch accept/reject assessment, tier and flag breakdowns, and
a per-file report with input metrics, capture type, action taken, output metrics,
and spec conformance. Embedding the spec makes any manifest self-describing
without reference to the code that produced it.

**`repair_queue_<timestamp>.json`** — quarantined files with their artifact
counts, and instructions for the repair cycle.

**`processing.log`** — full audit trail.

---

## Usage

```bash
pip install -r requirements.txt      # ffmpeg must also be on PATH
python batch_processor.py
```

Configuration lives in three blocks at the top of `batch_processor.py`:
paths, and the `SPEC` dictionary — every target, threshold, and detector
parameter, each commented with its rationale. Adapting to a different customer
spec is a one-block edit.

**Repair cycle**

1. Run in `MODE = "process"`. Tier 2 files are quarantined and listed in the
   repair queue.
2. Repair them by hand (e.g. iZotope RX de-plosive / de-click).
3. Place corrected files in `REPAIRED_DIR`, set `MODE = "verify"`, re-run.
   Verify mode skips the VST chain — the artifact is already fixed — but runs
   the same QA gate and conforms loudness/format.

---

## Tests

```bash
python tests/test_detectors.py
```

Twelve tests over synthetic signals with known artifact counts, covering clean
material at two fundamentals, injected plosives and ticks, cross-talk isolation
between the two detectors, and capture-type classification including the
upsampled-telephony case.

---

## Scope and limitations

- Detector thresholds are calibrated against **synthetic** test signals. They are
  a defensible starting point, not production-validated numbers — recalibrate
  against real material from your own corpus before relying on them.
- Requires licensed third-party VST3 plugins (FabFilter Pro-Q 3, iZotope RX 8
  Voice De-noise, Accentize dxRevive Pro). Plugin paths are configurable; the
  chain composition currently expects four plugins.
- Plugin state extraction targets Reaper's `.RPP` format specifically.
- Multilingual routing (per-language acoustic profiles) is designed but not yet
  implemented.

---

## Requirements

Python 3.10+, `ffmpeg` and `ffprobe` on PATH, and the packages in
`requirements.txt`.

---

## License

MIT — see [LICENSE](LICENSE).
