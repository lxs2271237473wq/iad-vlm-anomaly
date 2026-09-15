# Stage24: repaired evidence baseline

This directory is independent of stages 1–23. Historical scripts, cached scores,
adapted datasets, and checkpoints are not overwritten.

## Current execution

- Server alias: `iad-vlm-4090`; project `/root/private_data/iad-vlm-anomaly`.
- Python: `/opt/conda/bin/python` (Anomalib 2.5.0; PyTorch 2.7.0+cu118).
- Run: `results/stage24_evidence/20260910_a`.
- `build_manifest.py`: original AD2 official train/validation/public, all 8 categories;
  VisA official 1cls split, with 10% normal training images reserved by fixed hash
  ordering for calibration. Every image has an SHA256, stable ID, and dimensions.
- `images.csv` contains no label column. `evaluation_labels.csv` is consumed only
  by metric evaluation after raw predictions and normal calibration are saved.
- Private AD2 data is not used. Filename-based scene groups are provisional
  evaluation groups, not independently verified physical-object metadata.
- Reference/detector training excludes calibration images.

## Fixed baseline configuration

- PatchCore: pretrained wide_resnet50_2, layer2/layer3, standard k-center greedy
  coreset ratio 0.1, 9 neighbors, seed 42, batch size 4.
- Full-field RGB resize to 256 x 256 and ImageNet normalization; no center crop.
  Aspect ratio is explicitly warped. This is the first repaired baseline, not a
  claim that 256 pixels is sufficient for all AD2 defects. Later resolution
  controls must apply equally to detector and proposed method.
- Direct Anomalib Torch model; no Lightning validation, label-dependent
  postprocessor, test min/max, or test-fitted decision threshold.
- CLIP: cached OpenAI ViT-B/32 weights, full-field 224 x 224 for each supplied view,
  fixed three normal and three anomalous category prompts. Both pretrained files
  are hashed in environment.json.
- Candidate map: >97th percentile, connected components of >=4 map pixels,
  top three ordered by peak and mean; integer half-open boxes mapped directly
  back to original image dimensions; context factor 1.5.
- If no component exists, use full-image CLIP and explicitly mark fallback.
- Raw scores are always retained. Fusion scales are fitted separately per category
  using only held-out normal calibration: median / (1.4826 MAD), std fallback,
  no clipping. Scores are not probabilities or guaranteed confidence estimates.
- Whole-image, tight top1, context top1 and context top3max CLIP scores are saved.
  Source folds must choose variants/weights; target metrics cannot choose them.
- Current outputs include detector, CLIP and naive fusion. Source-selected fixed
  weights, learned fusion and the proposed exchange evidence are later stages.

## Commands on the server

```bash
cd /root/private_data/iad-vlm-anomaly
/opt/conda/bin/python experiments/stage24_evidence/test_geometry.py
# Already launched; do not launch a duplicate. A flock prevents concurrent queues.
tail -n 30 results/stage24_evidence/20260910_a/queue.log
cat results/stage24_evidence/20260910_a/queue.pid
# Summarize only completed categories; incomplete datasets never receive a full macro.
/opt/conda/bin/python experiments/stage24_evidence/summarize_baselines.py \
  --out results/stage24_evidence/20260910_a/baselines
# Resume the same configuration after an observed failure / completed interruption:
# bash experiments/stage24_evidence/run_queue.sh
```

The queue writes `queue.exit` and `queue.finished` on exit; exit 0 plus
`baselines/queue_complete.json` means all requested categories finished. Each
category has an independent `complete.json`, bank hash, normal calibration,
raw and scaled scores, maps, ROI records, prompts/configuration and timings.
Timing fields for detector and CLIP exclude some image I/O; category wall time
includes training and inference. Do not report those fields as end-to-end savings.

## Next acceptance gates

1. Verify all 1,084 AD2 public and 2,162 VisA evaluation IDs are present.
2. Select fixed weights and simple learned fusion using source category folds;
   report held-out results, never training-fit AUROC as generalization.
3. Prototype bidirectional matched-normal ROI exchange on source folds first:
   one ROI, two references, four swapped views, normal-normal controls, no GT ROI.
4. Compare same-reference / same-view-budget controls, single-direction exchange,
   calibration removal and shuffled evidence before expanding the gate.
5. If source out-of-fold evidence adds no correction value, diagnose ROI coverage,
   reference mismatch and CLIP sensitivity; do not expand model capacity by default.

No GPU result from this baseline is evidence that the new exchange method works.
