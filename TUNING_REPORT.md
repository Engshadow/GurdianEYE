# Hyperparameter Tuning Report (GurdianEYE)

Tuning followed the four-metric methodology: violation recall/precision,
alert latency, flicker rate. Infrastructure lives in `tuning/` (see
"Tuning workflow" below). All sweeps ran on cached detections of the 6
practice videos (~1,780 frames, `models/best.pt`, imgsz 640 & 1280).

## What was run

| Step | Command | Result |
|---|---|---|
| Cache detections (once, GPU) | `python -m tuning.cache_detections --videos videos/*.mp4 --imgsz 640 1280` | 12 JSONL caches in `tuning/cache/` |
| Timing sweep (48 configs) | `python -m tuning.sweep --group timing --cache "tuning/cache/*_i640_iou30.jsonl"` | `tuning/results/timing_sweep.csv` |
| Confidence sweep (24) | `--group conf` | `conf_sweep.csv` |
| Geometry sweep (27) | `--group geometry` | `geometry_sweep.csv` |
| EMA sweep (24) | `--group ema` | `ema_sweep.csv` |
| Draft GT (cross-model) | `tuning/pseudo_gt.py` + annotator model | `tuning/gt/gt_*.jsonl` |
| Event export + crop review | `tuning/export_events.py` | `tuning/labeling/crops/` |

A full replay of 503 frames takes **0.03 s** (vs ~50 s of GPU inference),
which is what made exhaustive grids practical.

## Findings

### Timing (the group that matters)
Averaged over all videos (lower flicker and alerts = more stable):

| filter_seconds | flicker | alerts | mean latency |
|---|---|---|---|
| 0.2 | 0.0607 | 0.78/video | 1.16 s |
| 0.3 | 0.0416 | 0.56 | 1.55 s |
| **0.5** | **0.0313** | **0.39** | **2.12 s** |
| 0.8 | 0.0248 | 0.28 | 2.32 s |

- **filter_seconds = 0.5** is the knee: flicker halves from 0.2 → 0.5, while
  0.5 → 0.8 buys only −20% flicker for +0.2 s latency. **Adopted: 0.5 s.**
- **stability_seconds** barely moves flicker (0.0433 → 0.0379 across 0.2–0.8).
  Per the guide, keep it short so a real PPE removal isn't masked. **Adopted: 0.4 s.**
- **violation_seconds** is redundant with the filter (both debounce).
  **Adopted: 0.5 s** (down from 1.0) — cuts total alert latency.

### Detector confidence
`detector_conf` 0.10–0.30 changed almost nothing: this model is
well-calibrated and produces very few mid-confidence boxes, so the
`max(conf_detector, conf_matcher)` interaction is dominated by the
matcher's 0.30. No change needed; re-check after retraining.

### EMA alphas
Flicker differences were small (0.0249–0.0257 in the top band); current
`box_smoothing=0.6, conf_smoothing=0.5` sit inside the optimal band
(box 0.4–0.8, conf 0.2–0.3 marginally better). **No change; re-derive
`min_ppe_confidence` from the smoothed histogram if alphas change.**

### Matcher geometry
`containment_threshold` 0.25–0.50, `min_match_score` 0.25–0.45 and
`head_region_ratio` 0.25–0.40 were nearly insensitive on these clips —
the current values are fine. Thresholds should be re-derived from labeled
pairs (`tuning/derive_thresholds.py`) once ~200 pairs are labeled.

### imgsz 640 vs 1280
Mixed per video (1280 found an extra event in `test`/`test4` but missed
one in `test2`/`test5`, and doubles inference cost). Keep **640** as the
default; 1280 remains available for far/small-object scenes.

### Alert audit at the adopted config
2 alerts across all videos (crop-reviewed in `tuning/labeling/crops/`):
- `test5_f178` — worker with helmet, **no vest** → **true violation** ✓ (latency 1.27 s)
- `test1_f408` — a **traffic barrier misdetected as a person** → false alarm ✗

The false alarm is a detector (person-class) failure, not a matcher/timing
one — ceiling is the detector (see "Next steps").

## Adopted configuration

`timing.py` defaults: `violation_seconds=0.5, filter_seconds=0.5,
stability_seconds=0.4`. `main.py` passes these through, with
`min_person_confidence=0.45, min_ppe_confidence=0.30`.

## Caveats

- These practice videos contain ~0 annotated violations (the annotator
  model found none, and only 1 of 2 alerts was real). Recall/precision
  rankings are therefore **provisional**. The GT pipeline
  (`tuning/gt/*.jsonl` draft + `tuning/export_events.py` crops) is ready:
  annotate ~300–500 sampled frames across 3–5 varied videos, hold out one
  video, then re-rank sweeps with `--gt` (score = 2·recall + precision).
- Draft GT files are machine-generated (cross-model consensus) — review
  before trusting any GT-based number.

## Next steps
1. Label GT via the exported crops / `gt_template.jsonl` (CVAT/LabelStudio).
2. Re-run sweeps with `--gt`; derive matcher thresholds from labeled pairs.
3. Detector-side fixes for the person-on-barrier false alarm: more person
   hard negatives, `model.tune()`, multi_scale training.
4. Evaluate `ppe_yolo11s_fet004.pt` (no-hardhat/no-vest classes) as the
   primary model — its negatives give direct violation evidence.
