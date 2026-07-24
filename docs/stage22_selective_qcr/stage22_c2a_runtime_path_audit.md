# Stage 22-C2a: Runtime Path Audit

## Purpose

Identify the existing VisA VLM inference path before
implementing actual selective invocation and timing.

- Python files scanned: `21`
- model execution: `none`
- GPU use: `none`

## prediction_outputs

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py` lines 14–30

```python
  14: OUT_DOC = DOC_DIR / "stage16_a0_qcru_inventory_and_ablation_plan.md"
  15: 
  16: SOURCE_PATHS = [
  17:     "results/stage9_qcr_u/stage9_a0_input_structure.csv",
  18:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv",
  19:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv",
  20:     "results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv",
  21:     "results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv",
  22:     "results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv",
  23:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv",
  24:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv",
  25:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv",
  26:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv",
  27:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv",
  28:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv",
  29:     "results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv",
  30:     "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py` lines 6–17

```python
   6: 
   7: 
   8: ROOT = Path(".").resolve()
   9: 
  10: IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"
  11: 
  12: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
  13: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  14: 
  15: OUT_PER_CONFIG = OUT_DIR / "stage16_b_adaptive_qcru_all_variants_per_config.csv"
  16: OUT_PER_CATEGORY = OUT_DIR / "stage16_b_adaptive_qcru_all_variants_per_category.csv"
  17: OUT_PRIMARY = OUT_DIR / "stage16_b_adaptive_qcru_primary_protocol_table.csv"
```

### `experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py` lines 6–17

```python
   6: 
   7: 
   8: ROOT = Path(".").resolve()
   9: 
  10: IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"
  11: IN_STAGE16D_DELTAS = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv"
  12: 
  13: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
  14: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  15: 
  16: OUT_CASES = OUT_DIR / "stage16_e_failure_boundary_case_inventory.csv"
  17: OUT_CATEGORY = OUT_DIR / "stage16_e_category_boundary_summary.csv"
```

### `experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py` lines 8–19

```python
   8: 
   9: 
  10: ROOT = Path(".").resolve()
  11: 
  12: IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"
  13: 
  14: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
  15: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  16: 
  17: OUT_PER_CONFIG = OUT_DIR / "stage16_a1_qcru_fixed_ablation_per_config.csv"
  18: OUT_PER_CATEGORY = OUT_DIR / "stage16_a1_qcru_fixed_ablation_per_category.csv"
  19: OUT_BEST = OUT_DIR / "stage16_a1_qcru_fixed_ablation_best_by_protocol.csv"
```

### `experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py` lines 6–17

```python
   6: 
   7: 
   8: ROOT = Path(".").resolve()
   9: 
  10: IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"
  11: 
  12: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
  13: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  14: 
  15: OUT_PER_CONFIG = OUT_DIR / "stage16_a3_adaptive_qcru_per_config.csv"
  16: OUT_DELTA = OUT_DIR / "stage16_a3_adaptive_qcru_delta_by_protocol.csv"
  17: OUT_FAILURES = OUT_DIR / "stage16_a3_adaptive_qcru_failure_cases.csv"
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 437–448

```python
 437:     out_root = Path(args.output_root)
 438:     category_root = out_root / "VisA" / category
 439:     category_root.mkdir(parents=True, exist_ok=True)
 440: 
 441:     image_csv = category_root / f"{args.backbone_model}_image_predictions.csv"
 442:     candidates_csv = category_root / "candidate_regions.csv"
 443: 
 444:     pd.DataFrame(image_records).to_csv(image_csv, index=False)
 445:     pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False)
 446: 
 447:     valid_candidates = pd.DataFrame(candidate_rows)
 448:     if len(valid_candidates):
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 460–471

```python
 460:         "num_candidate_rows": int(len(valid_candidates)),
 461:         "candidate_csv": str(candidates_csv),
 462:     }
 463: 
 464:     print(f"[DONE] Saved image predictions: {image_csv}")
 465:     print(f"[DONE] Saved candidate regions: {candidates_csv}")
 466: 
 467:     return coverage_row
 468: 
 469: 
 470: def main():
 471:     parser = argparse.ArgumentParser()
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 374–385

```python
 374:     out_root = Path(args.output_root)
 375:     category_root = out_root / "VisA" / category
 376:     category_root.mkdir(parents=True, exist_ok=True)
 377: 
 378:     image_csv = category_root / "patchcore_image_predictions.csv"
 379:     candidates_csv = category_root / "candidate_regions.csv"
 380: 
 381:     pd.DataFrame(image_records).to_csv(image_csv, index=False)
 382:     pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False)
 383: 
 384:     valid_candidates = pd.DataFrame(candidate_rows)
 385:     if len(valid_candidates):
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 397–408

```python
 397:         "num_candidate_rows": int(len(valid_candidates)),
 398:         "candidate_csv": str(candidates_csv),
 399:     }
 400: 
 401:     print(f"[DONE] Saved image predictions: {image_csv}")
 402:     print(f"[DONE] Saved candidate regions: {candidates_csv}")
 403: 
 404:     return coverage_row
 405: 
 406: 
 407: def main():
 408:     parser = argparse.ArgumentParser()
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 290–304

```python
 290:     return best_thr, best_f1, best_acc
 291: 
 292: 
 293: def evaluate_category(args, model, preprocess, tokenizer, device, category, strategy, eval_mode):
 294:     pred_csv = Path(args.patchcore_root) / "VisA" / category / "patchcore_image_predictions.csv"
 295: 
 296:     if not pred_csv.exists():
 297:         raise FileNotFoundError(f"Missing PatchCore image prediction CSV: {pred_csv}")
 298: 
 299:     df = pd.read_csv(pred_csv)
 300:     df = df[df["label"].isin(["normal", "anomaly"])].copy().reset_index(drop=True)
 301: 
 302:     boxes = load_candidate_boxes(args.patchcore_root, category, args.top_k)
 303:     text_features, prompt_row = build_text_features(
 304:         model=model,
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 466–490

```python
 466: 
 467:     mean_df = pd.DataFrame(mean_rows).sort_values(["best_f1", "auroc"], ascending=[False, False])
 468:     full_summary_df = pd.concat([summary_df, mean_df], ignore_index=True)
 469: 
 470:     predictions_csv = output_root / "visa_binary_prompt_predictions.csv"
 471:     summary_csv = output_root / "visa_binary_prompt_summary.csv"
 472:     mean_csv = output_root / "visa_binary_prompt_mean_summary.csv"
 473:     prompts_csv = output_root / "visa_binary_prompt_bank.csv"
 474: 
 475:     detail_df.to_csv(predictions_csv, index=False)
 476:     full_summary_df.to_csv(summary_csv, index=False)
 477:     mean_df.to_csv(mean_csv, index=False)
 478:     prompt_df.to_csv(prompts_csv, index=False)
 479: 
 480:     print("\n========== VisA Binary Prompt Mean Summary ==========")
 481:     print(mean_df.to_string(index=False))
 482: 
 483:     print(f"\n[DONE] Predictions saved to: {predictions_csv}")
 484:     print(f"[DONE] Summary saved to: {summary_csv}")
 485:     print(f"[DONE] Mean summary saved to: {mean_csv}")
 486:     print(f"[DONE] Prompt bank saved to: {prompts_csv}")
 487: 
 488: 
 489: if __name__ == "__main__":
 490:     main()
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 290–304

```python
 290:     return best_thr, best_f1, best_acc
 291: 
 292: 
 293: def evaluate_category(args, model, preprocess, tokenizer, device, category, strategy, eval_mode):
 294:     pred_csv = Path(args.candidate_root) / "VisA" / category / f"{args.backbone_model}_image_predictions.csv"
 295: 
 296:     if not pred_csv.exists():
 297:         raise FileNotFoundError(f"Missing PatchCore image prediction CSV: {pred_csv}")
 298: 
 299:     df = pd.read_csv(pred_csv)
 300:     df = df[df["label"].isin(["normal", "anomaly"])].copy().reset_index(drop=True)
 301: 
 302:     boxes = load_candidate_boxes(args.candidate_root, category, args.top_k)
 303:     text_features, prompt_row = build_text_features(
 304:         model=model,
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 475–499

```python
 475: 
 476:     mean_df = pd.DataFrame(mean_rows).sort_values(["best_f1", "auroc"], ascending=[False, False])
 477:     full_summary_df = pd.concat([summary_df, mean_df], ignore_index=True)
 478: 
 479:     predictions_csv = output_root / "visa_binary_prompt_predictions.csv"
 480:     summary_csv = output_root / "visa_binary_prompt_summary.csv"
 481:     mean_csv = output_root / "visa_binary_prompt_mean_summary.csv"
 482:     prompts_csv = output_root / "visa_binary_prompt_bank.csv"
 483: 
 484:     detail_df.to_csv(predictions_csv, index=False)
 485:     full_summary_df.to_csv(summary_csv, index=False)
 486:     mean_df.to_csv(mean_csv, index=False)
 487:     prompt_df.to_csv(prompts_csv, index=False)
 488: 
 489:     print("\n========== VisA Binary Prompt Mean Summary ==========")
 490:     print(mean_df.to_string(index=False))
 491: 
 492:     print(f"\n[DONE] Predictions saved to: {predictions_csv}")
 493:     print(f"[DONE] Summary saved to: {summary_csv}")
 494:     print(f"[DONE] Mean summary saved to: {mean_csv}")
 495:     print(f"[DONE] Prompt bank saved to: {prompts_csv}")
 496: 
 497: 
 498: if __name__ == "__main__":
 499:     main()
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 11–38

```python
  11: ROOT = Path(".").resolve()
  12: OUT_DIR = ROOT / "results" / "stage9_qcr_u"
  13: OUT_DIR.mkdir(parents=True, exist_ok=True)
  14: 
  15: OUT_PRED = OUT_DIR / "stage9_a1_qcr_u_fusion_predictions.csv"
  16: OUT_SUMMARY = OUT_DIR / "stage9_a1_qcr_u_fusion_summary.csv"
  17: OUT_REPORT = OUT_DIR / "stage9_a1_qcr_u_fusion_report.md"
  18: 
  19: 
  20: BACKBONE_CONFIGS = [
  21:     {
  22:         "backbone": "PatchCore",
  23:         "detector_root": ROOT / "results" / "stage7_generalization" / "visa_patchcore" / "VisA",
  24:         "vlm_predictions": ROOT / "results" / "stage7_generalization" / "visa_binary_prompt_reasoning" / "visa_binary_prompt_predictions.csv",
  25:         "detector_prediction_name": "patchcore_image_predictions.csv",
  26:     },
  27:     {
  28:         "backbone": "FastFlow",
  29:         "detector_root": ROOT / "results" / "stage7_generalization" / "visa_multibackbone" / "fastflow_12cls" / "VisA",
  30:         "vlm_predictions": ROOT / "results" / "stage7_generalization" / "visa_multibackbone" / "fastflow_binary_prompt_reasoning" / "visa_binary_prompt_predictions.csv",
  31:         "detector_prediction_name": "fastflow_image_predictions.csv",
  32:     },
  33: ]
  34: 
  35: 
  36: FUSION_WEIGHTS = {
  37:     "vlm_only": {"M": 1.00, "Q": 0.00, "K": 0.00, "D": 0.00},
  38:     "detector_only": {"M": 0.00, "Q": 0.00, "K": 0.00, "D": 1.00},
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 514–525

```python
 514:     lines.append("```")
 515:     lines.append("")
 516:     lines.append("## 3. Output Files")
 517:     lines.append("")
 518:     lines.append(f"- `{OUT_PRED.relative_to(ROOT)}`")
 519:     lines.append(f"- `{OUT_SUMMARY.relative_to(ROOT)}`")
 520:     lines.append(f"- `{OUT_REPORT.relative_to(ROOT)}`")
 521:     lines.append("")
 522:     lines.append("## 4. Best Overall Rows")
 523:     lines.append("")
 524:     lines.append("| Backbone | Strategy | Eval mode | Fusion | AUROC | AP | Best F1 | Delta AUROC vs VLM |")
 525:     lines.append("|---|---|---|---|---:|---:|---:|---:|")
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 567–582

```python
 567:     base = pd.concat(tables, ignore_index=True)
 568:     pred = add_fusion_scores(base)
 569:     summary = summarize(pred)
 570: 
 571:     pred.to_csv(OUT_PRED, index=False)
 572:     summary.to_csv(OUT_SUMMARY, index=False)
 573:     write_report(summary, pred)
 574: 
 575:     print("[DONE]", OUT_PRED)
 576:     print("[DONE]", OUT_SUMMARY)
 577:     print("[DONE]", OUT_REPORT)
 578:     print("prediction_rows:", len(pred))
 579:     print("summary_rows:", len(summary))
 580: 
 581:     print("\nTop QCR-U rows:")
 582:     show = (
```

### `experiments/stage9_qcr_u/run_stage9_a2_qcr_u_sanity_check.py` lines 8–19

```python
   8: import pandas as pd
   9: 
  10: 
  11: ROOT = Path(".").resolve()
  12: IN_PRED = ROOT / "results" / "stage9_qcr_u" / "stage9_a1_qcr_u_fusion_predictions.csv"
  13: IN_SUMMARY = ROOT / "results" / "stage9_qcr_u" / "stage9_a1_qcr_u_fusion_summary.csv"
  14: 
  15: OUT_DIR = ROOT / "results" / "stage9_qcr_u"
  16: OUT_PERCAT = OUT_DIR / "stage9_a2_qcr_u_per_category.csv"
  17: OUT_MACRO = OUT_DIR / "stage9_a2_qcr_u_macro_summary.csv"
  18: OUT_DIAG = OUT_DIR / "stage9_a2_qcr_u_signal_diagnostics.csv"
  19: OUT_REPORT = OUT_DIR / "stage9_a2_qcr_u_sanity_report.md"
```

### `experiments/stage9_qcr_u/run_stage9_a3_qcr_u_debias_check.py` lines 9–23

```python
   9: 
  10: 
  11: ROOT = Path(".").resolve()
  12: 
  13: IN_PRED = ROOT / "results" / "stage9_qcr_u" / "stage9_a1_qcr_u_fusion_predictions.csv"
  14: 
  15: OUT_DIR = ROOT / "results" / "stage9_qcr_u"
  16: OUT_PRED = OUT_DIR / "stage9_a3_qcr_u_debiased_predictions.csv"
  17: OUT_SUMMARY = OUT_DIR / "stage9_a3_qcr_u_debiased_summary.csv"
  18: OUT_PERCAT = OUT_DIR / "stage9_a3_qcr_u_debiased_per_category.csv"
  19: OUT_REPORT = OUT_DIR / "stage9_a3_qcr_u_debias_report.md"
  20: 
  21: 
  22: def to_binary_series(series: pd.Series) -> pd.Series:
  23:     def convert(x: object) -> int:
```

### `experiments/stage9_qcr_u/run_stage9_a3_qcr_u_debias_check.py` lines 423–434

```python
 423:     lines.append("The neutral value prevents no-candidate images from being automatically treated as normal through Q=0.")
 424:     lines.append("")
 425:     lines.append("## 3. Output Files")
 426:     lines.append("")
 427:     lines.append(f"- `{OUT_PRED.relative_to(ROOT)}`")
 428:     lines.append(f"- `{OUT_SUMMARY.relative_to(ROOT)}`")
 429:     lines.append(f"- `{OUT_PERCAT.relative_to(ROOT)}`")
 430:     lines.append(f"- `{OUT_REPORT.relative_to(ROOT)}`")
 431:     lines.append("")
 432:     lines.append("## 4. Best QCR-U / Debiased Rows")
 433:     lines.append("")
 434:     lines.append("| Backbone | Strategy | Eval mode | Method | AUROC | AP | Best F1 | ΔAUROC vs VLM |")
```

### `experiments/stage9_qcr_u/run_stage9_a3_qcr_u_debias_check.py` lines 484–500

```python
 484:     pred = build_debiased_scores(base)
 485:     summary = summarize(pred)
 486:     percat = per_category_summary(pred)
 487: 
 488:     pred.to_csv(OUT_PRED, index=False)
 489:     summary.to_csv(OUT_SUMMARY, index=False)
 490:     percat.to_csv(OUT_PERCAT, index=False)
 491:     write_report(summary, percat)
 492: 
 493:     print("[DONE]", OUT_PRED)
 494:     print("[DONE]", OUT_SUMMARY)
 495:     print("[DONE]", OUT_PERCAT)
 496:     print("[DONE]", OUT_REPORT)
 497:     print("prediction_rows:", len(pred))
 498:     print("summary_rows:", len(summary))
 499:     print("per_category_rows:", len(percat))
 500: 
```

## model_loading

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 176–232

```python
 176: 
 177: def collect_predictions(args, category, category_index=1, total_categories=1):
 178:     datamodule = build_datamodule(args, category)
 179: 
 180:     if args.backbone_model == "patchcore":
 181:         pre_processor = build_pre_processor_for_model(Patchcore, args)
 182:         model = Patchcore(
 183:             backbone=args.backbone,
 184:             layers=["layer2", "layer3"],
 185:             pre_trained=True,
 186:             coreset_sampling_ratio=args.coreset_sampling_ratio,
 187:             num_neighbors=args.num_neighbors,
 188:             pre_processor=pre_processor,
 189:         )
 190: 
 191:     elif args.backbone_model == "fastflow":
 192:         pre_processor = build_pre_processor_for_model(Fastflow, args)
 193:         model = Fastflow(
 194:             backbone="resnet18",
 195:             pre_trained=True,
 196:             flow_steps=8,
 197:             conv3x3_only=False,
 198:             hidden_ratio=1.0,
 199:             pre_processor=pre_processor,
 200:         )
 201: 
 202:     elif args.backbone_model == "reverse_distillation":
 203:         pre_processor = build_pre_processor_for_model(ReverseDistillation, args)
 204:         model = ReverseDistillation(
 205:             backbone="wide_resnet50_2",
 206:             layers=["layer1", "layer2", "layer3"],
 207:             pre_trained=True,
 208:             pre_processor=pre_processor,
 209:         )
 210: 
 211:     elif args.backbone_model == "stfpm":
 212:         pre_processor = build_pre_processor_for_model(Stfpm, args)
 213:         model = Stfpm(
 214:             backbone="resnet18",
 215:             layers=["layer1", "layer2", "layer3"],
 216:             pre_processor=pre_processor,
 217:         )
 218: 
 219:     elif args.backbone_model == "padim":
 220:         pre_processor = build_pre_processor_for_model(Padim, args)
 221:         model = Padim(
 222:             backbone="resnet18",
 223:             layers=["layer1", "layer2", "layer3"],
 224:             pre_trained=True,
 225:             pre_processor=pre_processor,
 226:         )
 227: 
 228:     else:
 229:         raise ValueError(f"Unknown backbone_model: {args.backbone_model}")
 230: 
 231:     work_dir = Path(args.work_root) / args.backbone_model / category
 232:     work_dir.mkdir(parents=True, exist_ok=True)
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 252–266

```python
 252:         limit_predict_batches=args.limit_predict_batches,
 253:     )
 254: 
 255:     print(f"[INFO] Fitting {args.backbone_model} on VisA category: {category}")
 256:     engine.fit(model=model, datamodule=datamodule)
 257: 
 258:     print(f"[INFO] Predicting {args.backbone_model} on VisA category: {category}")
 259:     predictions = engine.predict(model=model, datamodule=datamodule)
 260: 
 261:     return predictions
 262: 
 263: def evaluate_and_extract_candidates(args, category, predictions):
 264:     image_records = []
 265:     candidate_rows = []
 266: 
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 158–180

```python
 158: 
 159: def collect_predictions(args, category):
 160:     datamodule = build_datamodule(args, category)
 161: 
 162:     pre_processor = Patchcore.configure_pre_processor(
 163:         image_size=(args.image_size, args.image_size),
 164:         center_crop_size=(args.center_crop_size, args.center_crop_size),
 165:     )
 166: 
 167:     model = Patchcore(
 168:         backbone=args.backbone,
 169:         layers=["layer2", "layer3"],
 170:         pre_trained=True,
 171:         coreset_sampling_ratio=args.coreset_sampling_ratio,
 172:         num_neighbors=args.num_neighbors,
 173:         pre_processor=pre_processor,
 174:     )
 175: 
 176:     work_dir = Path(args.work_root) / category
 177:     work_dir.mkdir(parents=True, exist_ok=True)
 178: 
 179:     progress_callback = OneLineProgressCallback(category=category)
 180: 
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 188–202

```python
 188:         callbacks=[progress_callback],
 189:     )
 190: 
 191:     print(f"[INFO] Fitting PatchCore on VisA category: {category}")
 192:     engine.fit(model=model, datamodule=datamodule)
 193: 
 194:     print(f"[INFO] Predicting PatchCore on VisA category: {category}")
 195:     predictions = engine.predict(model=model, datamodule=datamodule)
 196: 
 197:     return predictions
 198: 
 199: 
 200: def evaluate_and_extract_candidates(args, category, predictions):
 201:     image_records = []
 202:     candidate_rows = []
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 300–311

```python
 300:     df = df[df["label"].isin(["normal", "anomaly"])].copy().reset_index(drop=True)
 301: 
 302:     boxes = load_candidate_boxes(args.patchcore_root, category, args.top_k)
 303:     text_features, prompt_row = build_text_features(
 304:         model=model,
 305:         tokenizer=tokenizer,
 306:         category=category,
 307:         strategy=strategy,
 308:         device=device,
 309:     )
 310: 
 311:     y_true = []
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 418–429

```python
 418:                 print(f"[INFO] Evaluating category={category}, strategy={strategy}, eval_mode={eval_mode}")
 419: 
 420:                 summary, details, prompt = evaluate_category(
 421:                     args=args,
 422:                     model=model,
 423:                     preprocess=preprocess,
 424:                     tokenizer=tokenizer,
 425:                     device=device,
 426:                     category=category,
 427:                     strategy=strategy,
 428:                     eval_mode=eval_mode,
 429:                 )
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 300–311

```python
 300:     df = df[df["label"].isin(["normal", "anomaly"])].copy().reset_index(drop=True)
 301: 
 302:     boxes = load_candidate_boxes(args.candidate_root, category, args.top_k)
 303:     text_features, prompt_row = build_text_features(
 304:         model=model,
 305:         tokenizer=tokenizer,
 306:         category=category,
 307:         strategy=strategy,
 308:         device=device,
 309:     )
 310: 
 311:     y_true = []
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 427–438

```python
 427:                 print(f"[INFO] Evaluating category={category}, strategy={strategy}, eval_mode={eval_mode}")
 428: 
 429:                 summary, details, prompt = evaluate_category(
 430:                     args=args,
 431:                     model=model,
 432:                     preprocess=preprocess,
 433:                     tokenizer=tokenizer,
 434:                     device=device,
 435:                     category=category,
 436:                     strategy=strategy,
 437:                     eval_mode=eval_mode,
 438:                 )
```

## inference

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py` lines 14–30

```python
  14: OUT_DOC = DOC_DIR / "stage16_a0_qcru_inventory_and_ablation_plan.md"
  15: 
  16: SOURCE_PATHS = [
  17:     "results/stage9_qcr_u/stage9_a0_input_structure.csv",
  18:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv",
  19:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv",
  20:     "results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv",
  21:     "results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv",
  22:     "results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv",
  23:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv",
  24:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv",
  25:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv",
  26:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv",
  27:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv",
  28:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv",
  29:     "results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv",
  30:     "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py` lines 234–248

```python
 234:         "",
 235:         "如果字段不够，先补：",
 236:         "",
 237:         "```text",
 238:         "Stage 16-A1-input: 构建统一 prediction table",
 239:         "```",
 240:         "",
 241:         "统一 prediction table 至少要包含：",
 242:         "",
 243:         "- category",
 244:         "- image_id 或 image_path",
 245:         "- gt_binary",
 246:         "- detector_score",
 247:         "- full_image_vlm_score",
 248:         "- crop_topk_vlm_score",
```

### `experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py` lines 6–17

```python
   6: 
   7: 
   8: ROOT = Path(".").resolve()
   9: 
  10: IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"
  11: 
  12: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
  13: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  14: 
  15: OUT_PER_CONFIG = OUT_DIR / "stage16_b_adaptive_qcru_all_variants_per_config.csv"
  16: OUT_PER_CATEGORY = OUT_DIR / "stage16_b_adaptive_qcru_all_variants_per_category.csv"
  17: OUT_PRIMARY = OUT_DIR / "stage16_b_adaptive_qcru_primary_protocol_table.csv"
```

### `experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py` lines 6–17

```python
   6: 
   7: 
   8: ROOT = Path(".").resolve()
   9: 
  10: IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"
  11: IN_STAGE16D_DELTAS = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv"
  12: 
  13: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
  14: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  15: 
  16: OUT_CASES = OUT_DIR / "stage16_e_failure_boundary_case_inventory.csv"
  17: OUT_CATEGORY = OUT_DIR / "stage16_e_category_boundary_summary.csv"
```

### `experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py` lines 467–478

```python
 467:         "## 1. Purpose",
 468:         "",
 469:         "Stage 16-D created the paper-facing main comparison. Stage 16-E explains method boundaries.",
 470:         "",
 471:         "This stage does not train models or rerun VLM inference. It mines the existing Stage 9 prediction table for representative boundary cases.",
 472:         "",
 473:         "## 2. Primary Scope",
 474:         "",
 475:         "The case inventory uses the QCR primary protocol:",
 476:         "",
 477:         "```text",
 478:         "dataset = VisA",
```

### `experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py` lines 8–19

```python
   8: 
   9: 
  10: ROOT = Path(".").resolve()
  11: 
  12: IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"
  13: 
  14: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
  15: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  16: 
  17: OUT_PER_CONFIG = OUT_DIR / "stage16_a1_qcru_fixed_ablation_per_config.csv"
  18: OUT_PER_CATEGORY = OUT_DIR / "stage16_a1_qcru_fixed_ablation_per_category.csv"
  19: OUT_BEST = OUT_DIR / "stage16_a1_qcru_fixed_ablation_best_by_protocol.csv"
```

### `experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py` lines 177–188

```python
 177:     missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
 178:     if missing:
 179:         raise RuntimeError(f"Missing required columns: {missing}")
 180: 
 181:     # Stage 9 prediction table has many duplicated image rows because fusion weights/methods vary.
 182:     # Base signals are image-level and should be identical across those rows, so we deduplicate.
 183:     base_cols = [
 184:         "backbone",
 185:         "dataset",
 186:         "category",
 187:         "strategy",
 188:         "eval_mode",
```

### `experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py` lines 334–345

```python
 334:         "# Stage 16-A1 QCR-U Fixed-Protocol Ablation",
 335:         "",
 336:         "## 1. Purpose",
 337:         "",
 338:         "This stage evaluates fixed, non-tuned QCR-U ablation variants using the existing Stage 9 prediction table.",
 339:         "",
 340:         "It does not train models, rerun VLM inference, or tune weights on the test set.",
 341:         "",
 342:         "## 2. Input",
 343:         "",
 344:         f"- source: `{IN_PRED.relative_to(ROOT)}`",
 345:         f"- deduplicated base rows: `{len(base)}`",
```

### `experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py` lines 6–17

```python
   6: 
   7: 
   8: ROOT = Path(".").resolve()
   9: 
  10: IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"
  11: 
  12: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
  13: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  14: 
  15: OUT_PER_CONFIG = OUT_DIR / "stage16_a3_adaptive_qcru_per_config.csv"
  16: OUT_DELTA = OUT_DIR / "stage16_a3_adaptive_qcru_delta_by_protocol.csv"
  17: OUT_FAILURES = OUT_DIR / "stage16_a3_adaptive_qcru_failure_cases.csv"
```

### `experiments/stage7_generalization/progress_utils.py` lines 68–79

```python
  68:         if self.stage == "fit":
  69:             epoch_frac = (self.epoch_index - 1 + frac) / max(float(self.max_epochs), 1.0)
  70:             return self.fit_weight * max(0.0, min(1.0, epoch_frac))
  71: 
  72:         if self.stage == "predict":
  73:             return self.fit_weight + (1.0 - self.fit_weight) * frac
  74: 
  75:         return frac
  76: 
  77:     def _overall_eta(self, done, total):
  78:         now = time.time()
  79:         total_elapsed = now - self.run_start_time
```

### `experiments/stage7_generalization/progress_utils.py` lines 148–160

```python
 148: 
 149:     def on_train_epoch_end(self, trainer, pl_module):
 150:         self._finish()
 151: 
 152:     def on_predict_start(self, trainer, pl_module):
 153:         total = getattr(trainer, "num_predict_batches", 1)
 154:         self._start("predict", total)
 155: 
 156:     def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
 157:         self._print(batch_idx + 1, self.total)
 158: 
 159:     def on_predict_end(self, trainer, pl_module):
 160:         self._finish()
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 173–184

```python
 173: 
 174:     return datamodule
 175: 
 176: 
 177: def collect_predictions(args, category, category_index=1, total_categories=1):
 178:     datamodule = build_datamodule(args, category)
 179: 
 180:     if args.backbone_model == "patchcore":
 181:         pre_processor = build_pre_processor_for_model(Patchcore, args)
 182:         model = Patchcore(
 183:             backbone=args.backbone,
 184:             layers=["layer2", "layer3"],
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 248–282

```python
 248:         enable_model_summary=False,
 249:         callbacks=[progress_callback],
 250:         max_epochs=args.max_epochs,
 251:         limit_train_batches=args.limit_train_batches,
 252:         limit_predict_batches=args.limit_predict_batches,
 253:     )
 254: 
 255:     print(f"[INFO] Fitting {args.backbone_model} on VisA category: {category}")
 256:     engine.fit(model=model, datamodule=datamodule)
 257: 
 258:     print(f"[INFO] Predicting {args.backbone_model} on VisA category: {category}")
 259:     predictions = engine.predict(model=model, datamodule=datamodule)
 260: 
 261:     return predictions
 262: 
 263: def evaluate_and_extract_candidates(args, category, predictions):
 264:     image_records = []
 265:     candidate_rows = []
 266: 
 267:     image_labels = []
 268:     image_scores = []
 269: 
 270:     pixel_labels_all = []
 271:     pixel_scores_all = []
 272: 
 273:     per_image_cache = []
 274: 
 275:     for batch in predictions:
 276:         image_paths = get_field(batch, "image_path")
 277:         gt_masks = get_field(batch, "gt_mask")
 278:         anomaly_maps = get_field(batch, "anomaly_map")
 279: 
 280:         batch_size = len(image_paths) if isinstance(image_paths, (list, tuple)) else 1
 281: 
 282:         for i in range(batch_size):
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 437–448

```python
 437:     out_root = Path(args.output_root)
 438:     category_root = out_root / "VisA" / category
 439:     category_root.mkdir(parents=True, exist_ok=True)
 440: 
 441:     image_csv = category_root / f"{args.backbone_model}_image_predictions.csv"
 442:     candidates_csv = category_root / "candidate_regions.csv"
 443: 
 444:     pd.DataFrame(image_records).to_csv(image_csv, index=False)
 445:     pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False)
 446: 
 447:     valid_candidates = pd.DataFrame(candidate_rows)
 448:     if len(valid_candidates):
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 460–471

```python
 460:         "num_candidate_rows": int(len(valid_candidates)),
 461:         "candidate_csv": str(candidates_csv),
 462:     }
 463: 
 464:     print(f"[DONE] Saved image predictions: {image_csv}")
 465:     print(f"[DONE] Saved candidate regions: {candidates_csv}")
 466: 
 467:     return coverage_row
 468: 
 469: 
 470: def main():
 471:     parser = argparse.ArgumentParser()
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 475–486

```python
 475:     default="fastflow",
 476:     choices=["patchcore", "fastflow", "reverse_distillation", "stfpm", "padim"],)
 477:     parser.add_argument("--max_epochs", type=int, default=1)
 478:     parser.add_argument("--limit_train_batches", type=float, default=1.0)
 479:     parser.add_argument("--limit_predict_batches", type=float, default=1.0)
 480:     parser.add_argument("--progress_refresh_interval", type=float, default=1.0)
 481:     parser.add_argument("--data_root", type=str, default="datasets/VisA_anomalib_1cls")
 482:     parser.add_argument("--categories", nargs="+", default=VISA_CATEGORIES)
 483:     parser.add_argument("--output_root", type=str, default="results/stage7_generalization/visa_multibackbone")
 484:     parser.add_argument("--work_root", type=str, default="runs/stage7_generalization/visa_multibackbone")
 485:     parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
 486:     parser.add_argument("--coreset_sampling_ratio", type=float, default=0.1)
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 501–516

```python
 501:     metric_rows = []
 502:     coverage_rows = []
 503: 
 504:     for category_index, category in enumerate(args.categories, start=1):
 505:         predictions = collect_predictions(args, category, category_index, len(args.categories))
 506:         metric_row, image_records, candidate_rows = evaluate_and_extract_candidates(
 507:             args=args,
 508:             category=category,
 509:             predictions=predictions,
 510:         )
 511:         coverage_row = save_category_outputs(
 512:             args=args,
 513:             category=category,
 514:             metric_row=metric_row,
 515:             image_records=image_records,
 516:             candidate_rows=candidate_rows,
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 155–166

```python
 155: 
 156:     return datamodule
 157: 
 158: 
 159: def collect_predictions(args, category):
 160:     datamodule = build_datamodule(args, category)
 161: 
 162:     pre_processor = Patchcore.configure_pre_processor(
 163:         image_size=(args.image_size, args.image_size),
 164:         center_crop_size=(args.center_crop_size, args.center_crop_size),
 165:     )
 166: 
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 190–219

```python
 190: 
 191:     print(f"[INFO] Fitting PatchCore on VisA category: {category}")
 192:     engine.fit(model=model, datamodule=datamodule)
 193: 
 194:     print(f"[INFO] Predicting PatchCore on VisA category: {category}")
 195:     predictions = engine.predict(model=model, datamodule=datamodule)
 196: 
 197:     return predictions
 198: 
 199: 
 200: def evaluate_and_extract_candidates(args, category, predictions):
 201:     image_records = []
 202:     candidate_rows = []
 203: 
 204:     image_labels = []
 205:     image_scores = []
 206: 
 207:     pixel_labels_all = []
 208:     pixel_scores_all = []
 209: 
 210:     per_image_cache = []
 211: 
 212:     for batch in predictions:
 213:         image_paths = get_field(batch, "image_path")
 214:         gt_masks = get_field(batch, "gt_mask")
 215:         anomaly_maps = get_field(batch, "anomaly_map")
 216: 
 217:         batch_size = len(image_paths) if isinstance(image_paths, (list, tuple)) else 1
 218: 
 219:         for i in range(batch_size):
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 374–385

```python
 374:     out_root = Path(args.output_root)
 375:     category_root = out_root / "VisA" / category
 376:     category_root.mkdir(parents=True, exist_ok=True)
 377: 
 378:     image_csv = category_root / "patchcore_image_predictions.csv"
 379:     candidates_csv = category_root / "candidate_regions.csv"
 380: 
 381:     pd.DataFrame(image_records).to_csv(image_csv, index=False)
 382:     pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False)
 383: 
 384:     valid_candidates = pd.DataFrame(candidate_rows)
 385:     if len(valid_candidates):
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 397–408

```python
 397:         "num_candidate_rows": int(len(valid_candidates)),
 398:         "candidate_csv": str(candidates_csv),
 399:     }
 400: 
 401:     print(f"[DONE] Saved image predictions: {image_csv}")
 402:     print(f"[DONE] Saved candidate regions: {candidates_csv}")
 403: 
 404:     return coverage_row
 405: 
 406: 
 407: def main():
 408:     parser = argparse.ArgumentParser()
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 428–443

```python
 428:     metric_rows = []
 429:     coverage_rows = []
 430: 
 431:     for category in args.categories:
 432:         predictions = collect_predictions(args, category)
 433:         metric_row, image_records, candidate_rows = evaluate_and_extract_candidates(
 434:             args=args,
 435:             category=category,
 436:             predictions=predictions,
 437:         )
 438:         coverage_row = save_category_outputs(
 439:             args=args,
 440:             category=category,
 441:             metric_row=metric_row,
 442:             image_records=image_records,
 443:             candidate_rows=candidate_rows,
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 100–112

```python
 100: 
 101: def encode_prompt_set(model, tokenizer, prompts, device):
 102:     tokens = tokenizer(prompts).to(device)
 103: 
 104:     with torch.no_grad():
 105:         features = model.encode_text(tokens)
 106:         features = features / features.norm(dim=-1, keepdim=True)
 107: 
 108:     feature = features.mean(dim=0, keepdim=True)
 109:     feature = feature / feature.norm(dim=-1, keepdim=True)
 110:     return feature
 111: 
 112: 
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 237–252

```python
 237: 
 238:     raise ValueError(f"Unknown eval mode: {eval_mode}")
 239: 
 240: 
 241: def encode_images(model, preprocess, images, device):
 242:     batch = torch.cat([preprocess(img).unsqueeze(0) for img in images], dim=0).to(device)
 243: 
 244:     with torch.no_grad():
 245:         features = model.encode_image(batch)
 246:         features = features / features.norm(dim=-1, keepdim=True)
 247: 
 248:     return features
 249: 
 250: 
 251: def safe_auroc(y_true, scores):
 252:     if len(np.unique(y_true)) < 2:
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 290–304

```python
 290:     return best_thr, best_f1, best_acc
 291: 
 292: 
 293: def evaluate_category(args, model, preprocess, tokenizer, device, category, strategy, eval_mode):
 294:     pred_csv = Path(args.patchcore_root) / "VisA" / category / "patchcore_image_predictions.csv"
 295: 
 296:     if not pred_csv.exists():
 297:         raise FileNotFoundError(f"Missing PatchCore image prediction CSV: {pred_csv}")
 298: 
 299:     df = pd.read_csv(pred_csv)
 300:     df = df[df["label"].isin(["normal", "anomaly"])].copy().reset_index(drop=True)
 301: 
 302:     boxes = load_candidate_boxes(args.patchcore_root, category, args.top_k)
 303:     text_features, prompt_row = build_text_features(
 304:         model=model,
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 321–332

```python
 321:             fallback_count += 1
 322:         else:
 323:             covered_count += 1
 324: 
 325:         image_features = encode_images(model, preprocess, eval_images, device)
 326:         sims_matrix = (image_features @ text_features.T).detach().cpu().numpy()
 327: 
 328:         # text index 0 = normal, 1 = anomaly.
 329:         # for top-k crops, use max anomaly margin over crops.
 330:         margins = sims_matrix[:, 1] - sims_matrix[:, 0]
 331:         anomaly_score = float(np.max(margins))
 332: 
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 344–355

```python
 344:                 "image_path": row["image_path"],
 345:                 "canonical_image_path": canonical_path(row["image_path"]),
 346:                 "label": row["label"],
 347:                 "is_anomaly": true,
 348:                 "vlm_anomaly_score": anomaly_score,
 349:                 "fallback": int(fallback),
 350:                 "num_eval_images": len(eval_images),
 351:             }
 352:         )
 353: 
 354:     y_true_np = np.asarray(y_true).astype(int)
 355:     scores_np = np.asarray(anomaly_scores).astype(float)
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 466–490

```python
 466: 
 467:     mean_df = pd.DataFrame(mean_rows).sort_values(["best_f1", "auroc"], ascending=[False, False])
 468:     full_summary_df = pd.concat([summary_df, mean_df], ignore_index=True)
 469: 
 470:     predictions_csv = output_root / "visa_binary_prompt_predictions.csv"
 471:     summary_csv = output_root / "visa_binary_prompt_summary.csv"
 472:     mean_csv = output_root / "visa_binary_prompt_mean_summary.csv"
 473:     prompts_csv = output_root / "visa_binary_prompt_bank.csv"
 474: 
 475:     detail_df.to_csv(predictions_csv, index=False)
 476:     full_summary_df.to_csv(summary_csv, index=False)
 477:     mean_df.to_csv(mean_csv, index=False)
 478:     prompt_df.to_csv(prompts_csv, index=False)
 479: 
 480:     print("\n========== VisA Binary Prompt Mean Summary ==========")
 481:     print(mean_df.to_string(index=False))
 482: 
 483:     print(f"\n[DONE] Predictions saved to: {predictions_csv}")
 484:     print(f"[DONE] Summary saved to: {summary_csv}")
 485:     print(f"[DONE] Mean summary saved to: {mean_csv}")
 486:     print(f"[DONE] Prompt bank saved to: {prompts_csv}")
 487: 
 488: 
 489: if __name__ == "__main__":
 490:     main()
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 100–112

```python
 100: 
 101: def encode_prompt_set(model, tokenizer, prompts, device):
 102:     tokens = tokenizer(prompts).to(device)
 103: 
 104:     with torch.no_grad():
 105:         features = model.encode_text(tokens)
 106:         features = features / features.norm(dim=-1, keepdim=True)
 107: 
 108:     feature = features.mean(dim=0, keepdim=True)
 109:     feature = feature / feature.norm(dim=-1, keepdim=True)
 110:     return feature
 111: 
 112: 
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 237–252

```python
 237: 
 238:     raise ValueError(f"Unknown eval mode: {eval_mode}")
 239: 
 240: 
 241: def encode_images(model, preprocess, images, device):
 242:     batch = torch.cat([preprocess(img).unsqueeze(0) for img in images], dim=0).to(device)
 243: 
 244:     with torch.no_grad():
 245:         features = model.encode_image(batch)
 246:         features = features / features.norm(dim=-1, keepdim=True)
 247: 
 248:     return features
 249: 
 250: 
 251: def safe_auroc(y_true, scores):
 252:     if len(np.unique(y_true)) < 2:
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 290–304

```python
 290:     return best_thr, best_f1, best_acc
 291: 
 292: 
 293: def evaluate_category(args, model, preprocess, tokenizer, device, category, strategy, eval_mode):
 294:     pred_csv = Path(args.candidate_root) / "VisA" / category / f"{args.backbone_model}_image_predictions.csv"
 295: 
 296:     if not pred_csv.exists():
 297:         raise FileNotFoundError(f"Missing PatchCore image prediction CSV: {pred_csv}")
 298: 
 299:     df = pd.read_csv(pred_csv)
 300:     df = df[df["label"].isin(["normal", "anomaly"])].copy().reset_index(drop=True)
 301: 
 302:     boxes = load_candidate_boxes(args.candidate_root, category, args.top_k)
 303:     text_features, prompt_row = build_text_features(
 304:         model=model,
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 321–332

```python
 321:             fallback_count += 1
 322:         else:
 323:             covered_count += 1
 324: 
 325:         image_features = encode_images(model, preprocess, eval_images, device)
 326:         sims_matrix = (image_features @ text_features.T).detach().cpu().numpy()
 327: 
 328:         # text index 0 = normal, 1 = anomaly.
 329:         # for top-k crops, use max anomaly margin over crops.
 330:         margins = sims_matrix[:, 1] - sims_matrix[:, 0]
 331:         anomaly_score = float(np.max(margins))
 332: 
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 344–355

```python
 344:                 "image_path": row["image_path"],
 345:                 "canonical_image_path": canonical_path(row["image_path"]),
 346:                 "label": row["label"],
 347:                 "is_anomaly": true,
 348:                 "vlm_anomaly_score": anomaly_score,
 349:                 "fallback": int(fallback),
 350:                 "num_eval_images": len(eval_images),
 351:             }
 352:         )
 353: 
 354:     y_true_np = np.asarray(y_true).astype(int)
 355:     scores_np = np.asarray(anomaly_scores).astype(float)
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 475–499

```python
 475: 
 476:     mean_df = pd.DataFrame(mean_rows).sort_values(["best_f1", "auroc"], ascending=[False, False])
 477:     full_summary_df = pd.concat([summary_df, mean_df], ignore_index=True)
 478: 
 479:     predictions_csv = output_root / "visa_binary_prompt_predictions.csv"
 480:     summary_csv = output_root / "visa_binary_prompt_summary.csv"
 481:     mean_csv = output_root / "visa_binary_prompt_mean_summary.csv"
 482:     prompts_csv = output_root / "visa_binary_prompt_bank.csv"
 483: 
 484:     detail_df.to_csv(predictions_csv, index=False)
 485:     full_summary_df.to_csv(summary_csv, index=False)
 486:     mean_df.to_csv(mean_csv, index=False)
 487:     prompt_df.to_csv(prompts_csv, index=False)
 488: 
 489:     print("\n========== VisA Binary Prompt Mean Summary ==========")
 490:     print(mean_df.to_string(index=False))
 491: 
 492:     print(f"\n[DONE] Predictions saved to: {predictions_csv}")
 493:     print(f"[DONE] Summary saved to: {summary_csv}")
 494:     print(f"[DONE] Mean summary saved to: {mean_csv}")
 495:     print(f"[DONE] Prompt bank saved to: {prompts_csv}")
 496: 
 497: 
 498: if __name__ == "__main__":
 499:     main()
```

### `experiments/stage9_qcr_u/inspect_stage9_inputs.py` lines 22–33

```python
  22:     "candidate",
  23:     "region",
  24:     "bbox",
  25:     "box",
  26:     "prediction",
  27:     "prompt",
  28:     "reasoning",
  29:     "binary",
  30:     "clip",
  31:     "patchcore",
  32:     "fastflow",
  33:     "visa",
```

### `experiments/stage9_qcr_u/inspect_stage9_inputs.py` lines 72–83

```python
  72:     if "candidate" in text or "region" in text or "bbox" in col_text or "box" in col_text:
  73:         return "candidate_or_region"
  74:     if "prompt" in text or "reasoning" in text or "clip" in text or "normal_score" in col_text or "anomaly_score" in col_text:
  75:         return "vlm_reasoning"
  76:     if "prediction" in text or "metric" in text or "summary" in text or "auroc" in col_text:
  77:         return "detector_or_summary"
  78:     return "unknown"
  79: 
  80: 
  81: def has_group(columns: List[str], group_terms: List[str]) -> bool:
  82:     lower_cols = [c.lower() for c in columns]
  83:     for c in lower_cols:
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 11–38

```python
  11: ROOT = Path(".").resolve()
  12: OUT_DIR = ROOT / "results" / "stage9_qcr_u"
  13: OUT_DIR.mkdir(parents=True, exist_ok=True)
  14: 
  15: OUT_PRED = OUT_DIR / "stage9_a1_qcr_u_fusion_predictions.csv"
  16: OUT_SUMMARY = OUT_DIR / "stage9_a1_qcr_u_fusion_summary.csv"
  17: OUT_REPORT = OUT_DIR / "stage9_a1_qcr_u_fusion_report.md"
  18: 
  19: 
  20: BACKBONE_CONFIGS = [
  21:     {
  22:         "backbone": "PatchCore",
  23:         "detector_root": ROOT / "results" / "stage7_generalization" / "visa_patchcore" / "VisA",
  24:         "vlm_predictions": ROOT / "results" / "stage7_generalization" / "visa_binary_prompt_reasoning" / "visa_binary_prompt_predictions.csv",
  25:         "detector_prediction_name": "patchcore_image_predictions.csv",
  26:     },
  27:     {
  28:         "backbone": "FastFlow",
  29:         "detector_root": ROOT / "results" / "stage7_generalization" / "visa_multibackbone" / "fastflow_12cls" / "VisA",
  30:         "vlm_predictions": ROOT / "results" / "stage7_generalization" / "visa_multibackbone" / "fastflow_binary_prompt_reasoning" / "visa_binary_prompt_predictions.csv",
  31:         "detector_prediction_name": "fastflow_image_predictions.csv",
  32:     },
  33: ]
  34: 
  35: 
  36: FUSION_WEIGHTS = {
  37:     "vlm_only": {"M": 1.00, "Q": 0.00, "K": 0.00, "D": 0.00},
  38:     "detector_only": {"M": 0.00, "Q": 0.00, "K": 0.00, "D": 1.00},
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 231–243

```python
 231:     agg = add_group_minmax(agg, "candidate_quality", "candidate_quality_norm", ["category"])
 232:     return agg
 233: 
 234: 
 235: def read_detector_predictions(detector_root: Path, prediction_name: str) -> pd.DataFrame:
 236:     files = sorted(detector_root.glob(f"*/{prediction_name}"))
 237:     frames = []
 238: 
 239:     for path in files:
 240:         df = pd.read_csv(path)
 241:         if df.empty:
 242:             continue
 243: 
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 278–305

```python
 278: 
 279:     return det
 280: 
 281: 
 282: def read_vlm_predictions(path: Path, backbone: str) -> pd.DataFrame:
 283:     if not path.exists():
 284:         raise FileNotFoundError(path)
 285: 
 286:     df = pd.read_csv(path)
 287:     if df.empty:
 288:         raise ValueError(f"Empty VLM prediction file: {path}")
 289: 
 290:     df = df.copy()
 291:     df["backbone"] = backbone
 292:     df["source_vlm_csv"] = str(path.relative_to(ROOT))
 293:     df["image_key"] = df.get("canonical_image_path", df["image_path"]).map(canonicalize_path)
 294: 
 295:     if "vlm_anomaly_score" not in df.columns:
 296:         raise ValueError(f"Missing vlm_anomaly_score in {path}")
 297: 
 298:     df["vlm_anomaly_score"] = pd.to_numeric(df["vlm_anomaly_score"], errors="coerce").fillna(0.0)
 299: 
 300:     if "strategy" not in df.columns:
 301:         df["strategy"] = "unknown_strategy"
 302:     if "eval_mode" not in df.columns:
 303:         df["eval_mode"] = "unknown_eval_mode"
 304:     if "used_mode" not in df.columns:
 305:         df["used_mode"] = "unknown_used_mode"
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 309–325

```python
 309: 
 310: def build_backbone_table(config: Dict[str, object]) -> pd.DataFrame:
 311:     backbone = str(config["backbone"])
 312:     detector_root = Path(config["detector_root"])
 313:     vlm_path = Path(config["vlm_predictions"])
 314:     prediction_name = str(config["detector_prediction_name"])
 315: 
 316:     candidates = read_candidate_quality(detector_root)
 317:     detector = read_detector_predictions(detector_root, prediction_name)
 318:     vlm = read_vlm_predictions(vlm_path, backbone)
 319: 
 320:     merged = vlm.merge(
 321:         detector,
 322:         on=["category", "image_key"],
 323:         how="left",
 324:         suffixes=("_vlm", "_detector"),
 325:     )
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 361–372

```python
 361:     merged["has_candidate"] = merged["num_candidates"] > 0
 362: 
 363:     merged = add_group_minmax(
 364:         merged,
 365:         "vlm_anomaly_score",
 366:         "vlm_score_norm",
 367:         ["backbone", "strategy", "eval_mode", "category"],
 368:     )
 369: 
 370:     merged["high_high_consistency"] = (
 371:         (1.0 - (merged["vlm_score_norm"] - merged["detector_score_norm"]).abs()).clip(0.0, 1.0)
 372:         * (0.5 * (merged["vlm_score_norm"] + merged["detector_score_norm"]))
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 389–400

```python
 389:         "is_anomaly_final",
 390:         "fallback",
 391:         "has_candidate",
 392:         "num_candidates",
 393:         "vlm_anomaly_score",
 394:         "vlm_score_norm",
 395:         "detector_image_score",
 396:         "detector_score_norm",
 397:         "candidate_quality_norm",
 398:         "high_high_consistency",
 399:     ]
 400:     base_cols = [c for c in base_cols if c in df.columns]
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 497–509

```python
 497:     lines.append("# Stage 9-A1 QCR-U Fusion Report")
 498:     lines.append("")
 499:     lines.append("## 1. Purpose")
 500:     lines.append("")
 501:     lines.append("This stage implements fixed-weight QCR-U fusion on existing VisA predictions.")
 502:     lines.append("It reads existing detector predictions, candidate regions, and VLM binary prompt scores.")
 503:     lines.append("It does not train models, rerun CLIP, or regenerate anomaly maps.")
 504:     lines.append("")
 505:     lines.append("## 2. Score Definition")
 506:     lines.append("")
 507:     lines.append("```text")
 508:     lines.append("M = normalized VLM anomaly score")
 509:     lines.append("Q = normalized candidate-region quality")
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 574–585

```python
 574: 
 575:     print("[DONE]", OUT_PRED)
 576:     print("[DONE]", OUT_SUMMARY)
 577:     print("[DONE]", OUT_REPORT)
 578:     print("prediction_rows:", len(pred))
 579:     print("summary_rows:", len(summary))
 580: 
 581:     print("\nTop QCR-U rows:")
 582:     show = (
 583:         summary[summary["fusion_method"].str.contains("qcr_u", regex=False)]
 584:         .sort_values("auroc", ascending=False)
 585:         .head(10)
```

### `experiments/stage9_qcr_u/run_stage9_a2_qcr_u_sanity_check.py` lines 8–19

```python
   8: import pandas as pd
   9: 
  10: 
  11: ROOT = Path(".").resolve()
  12: IN_PRED = ROOT / "results" / "stage9_qcr_u" / "stage9_a1_qcr_u_fusion_predictions.csv"
  13: IN_SUMMARY = ROOT / "results" / "stage9_qcr_u" / "stage9_a1_qcr_u_fusion_summary.csv"
  14: 
  15: OUT_DIR = ROOT / "results" / "stage9_qcr_u"
  16: OUT_PERCAT = OUT_DIR / "stage9_a2_qcr_u_per_category.csv"
  17: OUT_MACRO = OUT_DIR / "stage9_a2_qcr_u_macro_summary.csv"
  18: OUT_DIAG = OUT_DIR / "stage9_a2_qcr_u_signal_diagnostics.csv"
  19: OUT_REPORT = OUT_DIR / "stage9_a2_qcr_u_sanity_report.md"
```

### `experiments/stage9_qcr_u/run_stage9_a2_qcr_u_sanity_check.py` lines 363–374

```python
 363:     lines.append("")
 364:     lines.append("## 1. Purpose")
 365:     lines.append("")
 366:     lines.append("This stage checks whether Stage 9-A1 QCR-U fusion can be used as a paper-level module.")
 367:     lines.append("It reads existing Stage 9-A1 predictions only. It does not train models, rerun CLIP, or regenerate anomaly maps.")
 368:     lines.append("")
 369:     lines.append("## 2. Why This Check Is Necessary")
 370:     lines.append("")
 371:     lines.append("Stage 9-A1 shows strong QCR-U performance, but candidate_quality_only is also extremely strong.")
 372:     lines.append("Therefore, the current result must be treated carefully: QCR-U may be a useful calibration module, but candidate quality may already encode most anomaly evidence.")
 373:     lines.append("")
 374:     lines.append("## 3. Output Files")
```

### `experiments/stage9_qcr_u/run_stage9_a2_qcr_u_sanity_check.py` lines 461–472

```python
 461: 
 462: 
 463: def main() -> None:
 464:     if not IN_PRED.exists():
 465:         raise FileNotFoundError(f"Missing Stage 9-A1 prediction file: {IN_PRED}")
 466: 
 467:     pred = pd.read_csv(IN_PRED)
 468: 
 469:     required_cols = [
 470:         "backbone",
 471:         "strategy",
 472:         "eval_mode",
```

### `experiments/stage9_qcr_u/run_stage9_a2_qcr_u_sanity_check.py` lines 493–504

```python
 493:     print("[DONE]", OUT_PERCAT)
 494:     print("[DONE]", OUT_MACRO)
 495:     print("[DONE]", OUT_DIAG)
 496:     print("[DONE]", OUT_REPORT)
 497:     print("prediction_rows:", len(pred))
 498:     print("per_category_rows:", len(percat))
 499:     print("macro_rows:", len(macro))
 500:     print("diagnostic_rows:", len(diag))
 501: 
 502:     print("\nTop QCR-U macro rows:")
 503:     show = (
 504:         macro[macro["fusion_method"].isin(["qcr_u_fixed", "qcr_u_detector_aware"])]
```

### `experiments/stage9_qcr_u/run_stage9_a3_qcr_u_debias_check.py` lines 9–23

```python
   9: 
  10: 
  11: ROOT = Path(".").resolve()
  12: 
  13: IN_PRED = ROOT / "results" / "stage9_qcr_u" / "stage9_a1_qcr_u_fusion_predictions.csv"
  14: 
  15: OUT_DIR = ROOT / "results" / "stage9_qcr_u"
  16: OUT_PRED = OUT_DIR / "stage9_a3_qcr_u_debiased_predictions.csv"
  17: OUT_SUMMARY = OUT_DIR / "stage9_a3_qcr_u_debiased_summary.csv"
  18: OUT_PERCAT = OUT_DIR / "stage9_a3_qcr_u_debiased_per_category.csv"
  19: OUT_REPORT = OUT_DIR / "stage9_a3_qcr_u_debias_report.md"
  20: 
  21: 
  22: def to_binary_series(series: pd.Series) -> pd.Series:
  23:     def convert(x: object) -> int:
```

### `experiments/stage9_qcr_u/run_stage9_a3_qcr_u_debias_check.py` lines 114–125

```python
 114: 
 115:     return float(best_f1), float(best_acc), float(best_thr)
 116: 
 117: 
 118: def get_base_predictions() -> pd.DataFrame:
 119:     if not IN_PRED.exists():
 120:         raise FileNotFoundError(f"Missing input file: {IN_PRED}")
 121: 
 122:     pred = pd.read_csv(IN_PRED)
 123: 
 124:     required = [
 125:         "backbone",
```

### `experiments/stage9_qcr_u/run_stage9_a3_qcr_u_debias_check.py` lines 410–421

```python
 410:     lines.append("## 1. Purpose")
 411:     lines.append("")
 412:     lines.append("Stage 9-A2 showed that candidate_quality_only is extremely strong.")
 413:     lines.append("This stage removes candidate-existence bias by assigning a neutral Q value to images without candidates.")
 414:     lines.append("It reads Stage 9-A1 predictions only and does not train models or regenerate anomaly maps.")
 415:     lines.append("")
 416:     lines.append("## 2. Debias Setting")
 417:     lines.append("")
 418:     lines.append("```text")
 419:     lines.append("q_original = candidate_quality_norm")
 420:     lines.append("q_neutral = candidate_quality_norm if candidate exists else 0.5")
 421:     lines.append("```")
```

### `experiments/stage9_qcr_u/run_stage9_a3_qcr_u_debias_check.py` lines 479–490

```python
 479: 
 480: def main() -> None:
 481:     OUT_DIR.mkdir(parents=True, exist_ok=True)
 482: 
 483:     base = get_base_predictions()
 484:     pred = build_debiased_scores(base)
 485:     summary = summarize(pred)
 486:     percat = per_category_summary(pred)
 487: 
 488:     pred.to_csv(OUT_PRED, index=False)
 489:     summary.to_csv(OUT_SUMMARY, index=False)
 490:     percat.to_csv(OUT_PERCAT, index=False)
```

### `experiments/stage9_qcr_u/run_stage9_a3_qcr_u_debias_check.py` lines 493–504

```python
 493:     print("[DONE]", OUT_PRED)
 494:     print("[DONE]", OUT_SUMMARY)
 495:     print("[DONE]", OUT_PERCAT)
 496:     print("[DONE]", OUT_REPORT)
 497:     print("prediction_rows:", len(pred))
 498:     print("summary_rows:", len(summary))
 499:     print("per_category_rows:", len(percat))
 500: 
 501:     cols = [
 502:         "backbone",
 503:         "strategy",
 504:         "eval_mode",
```

## iteration

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py` lines 210–221

```python
 210:         "| Variant | Detector | Crop VLM | Quality | Consistency | Unknown | Purpose |",
 211:         "|---|---:|---:|---:|---:|---:|---|",
 212:     ]
 213: 
 214:     for _, r in plan.iterrows():
 215:         lines.append(
 216:             f"| {r['variant']} | "
 217:             f"{int(bool(r['uses_detector_score']))} | "
 218:             f"{int(bool(r['uses_crop_vlm']))} | "
 219:             f"{int(bool(r['uses_quality']))} | "
 220:             f"{int(bool(r['uses_consistency']))} | "
 221:             f"{int(bool(r['uses_unknown']))} | "
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py` lines 144–155

```python
 144: 
 145:     failures["failure_reason"] = ""
 146:     reasons = []
 147: 
 148:     for _, r in failures.iterrows():
 149:         rs = []
 150:         if not bool(r["v5_beats_naive"]):
 151:             rs.append("V5_not_better_than_naive")
 152:         if not bool(r["v5_beats_quality_only"]):
 153:             rs.append("V5_not_better_than_quality_only")
 154:         if not bool(r["v5_beats_detector"]):
 155:             rs.append("V5_not_better_than_detector")
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py` lines 177–188

```python
 177:         "| Check | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |",
 178:         "|---|---:|---:|---:|---:|---:|---:|---:|",
 179:     ]
 180: 
 181:     for _, r in summary.iterrows():
 182:         lines.append(
 183:             f"| {r['check']} | {int(r['wins'])} | {int(r['total_protocols'])} | "
 184:             f"{r['win_rate']:.4f} | {r['mean_delta']:.4f} | {r['median_delta']:.4f} | "
 185:             f"{r['min_delta']:.4f} | {r['max_delta']:.4f} |"
 186:         )
 187: 
 188:     lines += [
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py` lines 193–204

```python
 193:         "|---|---|---|---:|---:|---:|---:|---:|",
 194:     ]
 195: 
 196:     display = delta.sort_values("delta_v5_minus_v3_naive", ascending=False)
 197:     for _, r in display.iterrows():
 198:         lines.append(
 199:             f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | "
 200:             f"{r['V5']:.4f} | {r['V3']:.4f} | {r['V4']:.4f} | "
 201:             f"{r['delta_v5_minus_v3_naive']:+.4f} | {r['delta_v5_minus_v4_quality']:+.4f} |"
 202:         )
 203: 
 204:     lines += [
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py` lines 213–224

```python
 213:         lines += [
 214:             "| Backbone | Strategy | Eval Mode | V5-V3 | V5-V4 | V5-V0 | V5-V2 | Reason |",
 215:             "|---|---|---|---:|---:|---:|---:|---|",
 216:         ]
 217:         for _, r in failures.iterrows():
 218:             lines.append(
 219:                 f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | "
 220:                 f"{r['delta_v5_minus_v3_naive']:+.4f} | "
 221:                 f"{r['delta_v5_minus_v4_quality']:+.4f} | "
 222:                 f"{r['delta_v5_minus_v0_detector']:+.4f} | "
 223:                 f"{r['delta_v5_minus_v2_crop']:+.4f} | "
 224:                 f"{r['failure_reason']} |"
```

### `experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py` lines 403–414

```python
 403:         "| Backbone | Variant | AUROC | AP | Best F1 | Best Acc |",
 404:         "|---|---|---:|---:|---:|---:|",
 405:     ]
 406: 
 407:     for _, r in primary.iterrows():
 408:         lines.append(
 409:             f"| {r['backbone']} | {r['variant']} | "
 410:             f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['best_accuracy']:.4f} |"
 411:         )
 412: 
 413:     lines += [
 414:         "",
```

### `experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py` lines 417–428

```python
 417:         "| Scope | Comparison | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |",
 418:         "|---|---|---:|---:|---:|---:|---:|---:|---:|",
 419:     ]
 420: 
 421:     for _, r in decision.iterrows():
 422:         lines.append(
 423:             f"| {r['scope']} | {r['comparison']} | {int(r['wins'])} | {int(r['num_protocols'])} | "
 424:             f"{r['win_rate']:.4f} | {r['mean_delta']:+.4f} | {r['median_delta']:+.4f} | "
 425:             f"{r['min_delta']:+.4f} | {r['max_delta']:+.4f} |"
 426:         )
 427: 
 428:     if not decision.empty:
```

### `experiments/stage16_qcru_ablation/build_stage16_c_final_method_claims.py` lines 305–316

```python
 305:         "| Claim ID | Type | Claim | Paper Status |",
 306:         "|---|---|---|---|",
 307:     ]
 308: 
 309:     for _, r in out.iterrows():
 310:         lines.append(
 311:             f"| {r['claim_id']} | {r['claim_type']} | {r['claim']} | {r['paper_status']} |"
 312:         )
 313: 
 314:     lines += [
 315:         "",
 316:         "## 6. Safe Contribution Wording",
```

### `experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py` lines 61–72

```python
  61:     if mean.empty:
  62:         raise RuntimeError("Stage 15-E has no MEAN rows.")
  63: 
  64:     rows = []
  65:     for _, r in mean.iterrows():
  66:         method = str(r["method"])
  67:         fairness_tag = str(r.get("fairness_tag", ""))
  68:         protocol = str(r.get("protocol", ""))
  69: 
  70:         if "same-set" in method:
  71:             paper_role = "upper_bound_diagnostic_only"
  72:             use_in_main_claim = False
```

### `experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py` lines 132–143

```python
 132:     df = primary.copy()
 133:     df = to_num(df, ["auroc", "ap", "best_f1", "best_accuracy", "best_threshold"])
 134: 
 135:     rows = []
 136:     for _, r in df.iterrows():
 137:         display_name, paper_role, use_in_main_claim = rename_qcr_variant(
 138:             str(r["variant_id"]),
 139:             str(r["variant"]),
 140:         )
 141: 
 142:         rows.append(
 143:             {
```

### `experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py` lines 240–251

```python
 240:         )
 241: 
 242:     # Include Stage 16-B decision rows as evidence, but not as final table entries.
 243:     if not decision.empty:
 244:         for _, r in decision.iterrows():
 245:             rows.append(
 246:                 {
 247:                     "delta_type": "stage16b_decision_summary",
 248:                     "scope": r.get("scope", ""),
 249:                     "comparison": r.get("comparison", ""),
 250:                     "left_score": "",
 251:                     "right_score": "",
```

### `experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py` lines 327–338

```python
 327:         "| Rank | Method | Mean Image AUROC | Role | Fairness Tag |",
 328:         "|---:|---|---:|---|---|",
 329:     ]
 330: 
 331:     for _, r in system_table.iterrows():
 332:         lines.append(
 333:             f"| {int(r['rank_by_mean_image_auroc'])} | {r['method']} | "
 334:             f"{float(r['mean_image_auroc']):.4f} | {r['paper_role']} | {r['fairness_tag']} |"
 335:         )
 336: 
 337:     lines += [
 338:         "",
```

### `experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py` lines 347–358

```python
 347:         "| Backbone | Method | Variant | Image AUROC | AP | Best F1 | Role |",
 348:         "|---|---|---|---:|---:|---:|---|",
 349:     ]
 350: 
 351:     for _, r in qcr_table.iterrows():
 352:         lines.append(
 353:             f"| {r['backbone']} | {r['method']} | {r['variant_id']} | "
 354:             f"{float(r['image_auroc']):.4f} | {float(r['image_ap']):.4f} | "
 355:             f"{float(r['best_f1']):.4f} | {r['paper_role']} |"
 356:         )
 357: 
 358:     lines += [
```

### `experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py` lines 368–379

```python
 368:         "| Scope | Comparison | Left Score | Right Score | Delta | Interpretation |",
 369:         "|---|---|---:|---:|---:|---|",
 370:     ]
 371: 
 372:     for _, r in deltas.iterrows():
 373:         left = r["left_score"]
 374:         right = r["right_score"]
 375:         delta = r["delta"]
 376: 
 377:         left_s = "" if left == "" else f"{float(left):.4f}"
 378:         right_s = "" if right == "" else f"{float(right):.4f}"
 379:         try:
```

### `experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py` lines 398–409

```python
 398:         lines += [
 399:             "| Claim ID | Type | Claim | Status |",
 400:             "|---|---|---|---|",
 401:         ]
 402:         for _, r in claims.iterrows():
 403:             lines.append(
 404:                 f"| {r['claim_id']} | {r['claim_type']} | {r['claim']} | {r['paper_status']} |"
 405:             )
 406: 
 407:     lines += [
 408:         "",
 409:         "## 7. Safe Main Claim",
```

### `experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py` lines 240–251

```python
 240:         "delta_adaptive_minus_quality",
 241:         "delta_adaptive_minus_fixed",
 242:         "detector_vlm_disagreement",
 243:     ]
 244:     cols += [c for c in ["image_path", "gt_label", "defect_type", "has_candidate", "num_candidates", "fallback"] if c in g.columns]
 245:     cols = [c for c in cols if c in g.columns]
 246: 
 247:     out = g.sort_values(sort_col, ascending=ascending).head(n)[cols].copy()
 248:     out.insert(0, "case_type", case_type)
 249:     out.insert(1, "selection_metric", sort_col)
 250:     out.insert(2, "selection_order", "ascending" if ascending else "descending")
 251:     return out
```

### `experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py` lines 485–496

```python
 485:         "| Backbone | Category | V3 Naive | V4 Quality | V5 Fixed Q+C | V6 Adaptive | V4-V3 | V6-V4 | V5-V4 | Boundary Label |",
 486:         "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
 487:     ]
 488: 
 489:     for _, r in category_summary.iterrows():
 490:         lines.append(
 491:             f"| {r['backbone']} | {r['category']} | "
 492:             f"{r['V3']:.4f} | {r['V4']:.4f} | {r['V5']:.4f} | {r['V6']:.4f} | "
 493:             f"{r['delta_v4_quality_minus_v3_naive']:+.4f} | "
 494:             f"{r['delta_v6_adaptive_minus_v4_quality']:+.4f} | "
 495:             f"{r['delta_v5_fixed_minus_v4_quality']:+.4f} | "
 496:             f"{r['boundary_label']} |"
```

### `experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py` lines 523–545

```python
 523:             "",
 524:             "| Case Type | Count |",
 525:             "|---|---:|",
 526:         ]
 527:         for _, r in counts.iterrows():
 528:             lines.append(f"| {r['case_type']} | {int(r['count'])} |")
 529: 
 530:     lines += [
 531:         "",
 532:         "## 5. Boundary Decisions",
 533:         "",
 534:         "| Decision ID | Topic | Decision | Paper Action |",
 535:         "|---|---|---|---|",
 536:     ]
 537: 
 538:     for _, r in decision.iterrows():
 539:         lines.append(
 540:             f"| {r['decision_id']} | {r['topic']} | {r['decision']} | {r['paper_action']} |"
 541:         )
 542: 
 543:     lines += [
 544:         "",
 545:         "## 6. Paper Interpretation",
```

### `experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py` lines 250–264

```python
 250:         {
 251:             "claim_id": "P9",
 252:             "claim_category": "segmentation_boundary",
 253:             "paper_claim": "Do not claim pixel-level segmentation SOTA.",
 254:             "allowed_wording": "Pixel-level/localization signals are used to generate candidate evidence for image-level anomaly recognition.",
 255:             "forbidden_wording": "The method achieves pixel-level segmentation SOTA.",
 256:             "evidence_files": "stage15_modern_detector_baselines; stage16_qcru_ablation",
 257:             "evidence_summary": "The current method is evaluated and framed primarily for image-level anomaly recognition and candidate reasoning.",
 258:             "support_level": "strong_as_restriction",
 259:             "paper_section": "Limitations",
 260:             "caveat": "Pixel metrics may be reported only as auxiliary detector evidence, not as the main claim.",
 261:             "status": "reject",
 262:         },
 263:     ]
 264: 
```

### `experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py` lines 368–379

```python
 368:         "| Claim ID | Category | Paper Claim | Support | Status | Section |",
 369:         "|---|---|---|---|---|---|",
 370:     ]
 371: 
 372:     for _, r in claim_map.iterrows():
 373:         lines.append(
 374:             f"| {r['claim_id']} | {r['claim_category']} | {r['paper_claim']} | "
 375:             f"{r['support_level']} | {r['status']} | {r['paper_section']} |"
 376:         )
 377: 
 378:     lines += [
 379:         "",
```

### `experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py` lines 382–419

```python
 382:         "| Claim ID | Evidence Summary | Caveat |",
 383:         "|---|---|---|",
 384:     ]
 385: 
 386:     for _, r in claim_map.iterrows():
 387:         lines.append(
 388:             f"| {r['claim_id']} | {r['evidence_summary']} | {r['caveat']} |"
 389:         )
 390: 
 391:     lines += [
 392:         "",
 393:         "## 5. Rejected / Forbidden Claims",
 394:         "",
 395:         "| Claim ID | Forbidden Wording | Allowed Replacement |",
 396:         "|---|---|---|",
 397:     ]
 398: 
 399:     for _, r in rejected.iterrows():
 400:         lines.append(
 401:             f"| {r['claim_id']} | {r['forbidden_wording']} | {r['allowed_wording']} |"
 402:         )
 403: 
 404:     lines += [
 405:         "",
 406:         "## 6. Paper Readiness Status",
 407:         "",
 408:         "| Status Group | Claim IDs | Summary |",
 409:         "|---|---|---|",
 410:     ]
 411: 
 412:     for _, r in status.iterrows():
 413:         lines.append(
 414:             f"| {r['status_group']} | {r['claim_ids']} | {r['summary']} |"
 415:         )
 416: 
 417:     lines += [
 418:         "",
 419:         "## 7. Safe Abstract-level Wording",
```

### `experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py` lines 366–377

```python
 366: 
 367:     if best_protocol.empty:
 368:         lines.append("| - | - | - | - | - | - | - | - |")
 369:     else:
 370:         for _, r in best_protocol.head(20).iterrows():
 371:             lines.append(
 372:                 f"| {int(r['rank_by_v5_auroc'])} | {r['backbone']} | {r['dataset']} | "
 373:                 f"{r['strategy']} | {r['eval_mode']} | "
 374:                 f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} |"
 375:             )
 376: 
 377:     lines += [
```

### `experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py` lines 396–407

```python
 396:             "| Variant | AUROC | AP | Best F1 | Best Accuracy |",
 397:             "|---|---:|---:|---:|---:|",
 398:         ]
 399: 
 400:         for _, r in comp.iterrows():
 401:             lines.append(
 402:                 f"| {r['variant']} | {r['auroc']:.4f} | {r['ap']:.4f} | "
 403:                 f"{r['best_f1']:.4f} | {r['best_accuracy']:.4f} |"
 404:             )
 405: 
 406:         v = dict(zip(comp["variant"], comp["auroc"]))
 407:         naive = v.get("naive_detector_crop_fusion")
```

### `experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py` lines 384–395

```python
 384:         "| Check | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |",
 385:         "|---|---:|---:|---:|---:|---:|---:|---:|",
 386:     ]
 387: 
 388:     for _, r in summary.iterrows():
 389:         lines.append(
 390:             f"| {r['check']} | {int(r['wins'])} | {int(r['total_protocols'])} | "
 391:             f"{r['win_rate']:.4f} | {r['mean_delta']:.4f} | {r['median_delta']:.4f} | "
 392:             f"{r['min_delta']:.4f} | {r['max_delta']:.4f} |"
 393:         )
 394: 
 395:     lines += [
```

### `experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py` lines 399–410

```python
 399:         "| Rank | Backbone | Strategy | Eval Mode | V6 AUROC | AP | Best F1 |",
 400:         "|---:|---|---|---|---:|---:|---:|",
 401:     ]
 402: 
 403:     for _, r in best.iterrows():
 404:         lines.append(
 405:             f"| {int(r['rank_by_v6_auroc'])} | {r['backbone']} | {r['strategy']} | {r['eval_mode']} | "
 406:             f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} |"
 407:         )
 408: 
 409:     lines += [
 410:         "",
```

### `experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py` lines 414–425

```python
 414:         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
 415:     ]
 416: 
 417:     display = delta.sort_values("delta_v6_minus_v3_naive", ascending=False)
 418:     for _, r in display.iterrows():
 419:         lines.append(
 420:             f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | "
 421:             f"{r['V3']:.4f} | {r['V4']:.4f} | {r['V5']:.4f} | {r['V6']:.4f} | "
 422:             f"{r['delta_v6_minus_v3_naive']:+.4f} | "
 423:             f"{r['delta_v6_minus_v4_quality']:+.4f} | "
 424:             f"{r['delta_v6_minus_v5_fixed_qc']:+.4f} |"
 425:         )
```

### `experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py` lines 461–472

```python
 461:         | (~delta["v6_beats_fixed_qc"])
 462:     ].copy()
 463: 
 464:     reasons = []
 465:     for _, r in failures.iterrows():
 466:         rs = []
 467:         if not bool(r["v6_beats_naive"]):
 468:             rs.append("V6_not_better_than_naive")
 469:         if not bool(r["v6_beats_quality"]):
 470:             rs.append("V6_not_better_than_quality")
 471:         if not bool(r["v6_beats_fixed_qc"]):
 472:             rs.append("V6_not_better_than_fixed_qc")
```

### `experiments/stage7_generalization/build_visa_manifest.py` lines 52–63

```python
  52: 
  53:     rows = []
  54:     missing_rows = []
  55: 
  56:     for idx, row in df.iterrows():
  57:         category = str(row["object"])
  58:         split = str(row["split"])
  59:         label = str(row["label"])
  60: 
  61:         image_rel = normalize_path_text(row["image"])
  62:         mask_rel = normalize_path_text(row["mask"])
  63: 
```

### `experiments/stage7_generalization/prepare_visa_anomalib_view.py` lines 96–107

```python
  96: 
  97:     rows = []
  98:     errors = []
  99: 
 100:     for _, row in df.iterrows():
 101:         category = row["category"]
 102:         split = row["split"]
 103:         label = row["label"]
 104: 
 105:         image_src = Path(row["image_path"])
 106:         if not image_src.exists():
 107:             errors.append(
```

### `experiments/stage7_generalization/progress_utils.py` lines 152–160

```python
 152:     def on_predict_start(self, trainer, pl_module):
 153:         total = getattr(trainer, "num_predict_batches", 1)
 154:         self._start("predict", total)
 155: 
 156:     def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
 157:         self._print(batch_idx + 1, self.total)
 158: 
 159:     def on_predict_end(self, trainer, pl_module):
 160:         self._finish()
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 125–136

```python
 125:     return float(average_precision_score(y_true, y_score))
 126: 
 127: 
 128: def extract_score(batch, index, anomaly_map):
 129:     for field in ["pred_score", "pred_scores", "anomaly_score", "image_score", "score"]:
 130:         try:
 131:             values = get_field(batch, field)
 132:             value = take_item(values, index)
 133:             if hasattr(value, "detach"):
 134:                 value = value.detach().cpu().item()
 135:             elif hasattr(value, "item"):
 136:                 value = value.item()
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 163–176

```python
 163:         normal_dir="train/good",
 164:         abnormal_dir=["test/anomaly"],
 165:         normal_test_dir="test/good",
 166:         mask_dir=["ground_truth/anomaly"],
 167:         train_batch_size=args.train_batch_size,
 168:         eval_batch_size=args.eval_batch_size,
 169:         num_workers=args.num_workers,
 170:         seed=args.seed,
 171:         val_split_mode="same_as_test",
 172:     )
 173: 
 174:     return datamodule
 175: 
 176: 
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 276–289

```python
 276:         image_paths = get_field(batch, "image_path")
 277:         gt_masks = get_field(batch, "gt_mask")
 278:         anomaly_maps = get_field(batch, "anomaly_map")
 279: 
 280:         batch_size = len(image_paths) if isinstance(image_paths, (list, tuple)) else 1
 281: 
 282:         for i in range(batch_size):
 283:             image_path = str(take_item(image_paths, i))
 284:             y_image = infer_label_from_path(image_path)
 285: 
 286:             anomaly_map = normalize_map(take_item(anomaly_maps, i))
 287:             gt_mask = mask_to_2d(take_item(gt_masks, i))
 288: 
 289:             if anomaly_map is None:
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 329–340

```python
 329: 
 330:     pixel_auroc = safe_roc_auc(pixel_labels_np, pixel_scores_np)
 331:     pixel_ap = safe_ap(pixel_labels_np, pixel_scores_np)
 332: 
 333:     for item in per_image_cache:
 334:         pred_image_label = int(item["image_score"] >= image_thr)
 335: 
 336:         image_records.append(
 337:             {
 338:                 "dataset": "VisA",
 339:                 "category": category,
 340:                 "image_path": item["image_path"],
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 488–501

```python
 488:     parser.add_argument("--image_size", type=int, default=256)
 489:     parser.add_argument("--center_crop_size", type=int, default=224)
 490:     parser.add_argument("--top_components", type=int, default=3)
 491:     parser.add_argument("--min_area", type=int, default=20)
 492:     parser.add_argument("--train_batch_size", type=int, default=32)
 493:     parser.add_argument("--eval_batch_size", type=int, default=32)
 494:     parser.add_argument("--num_workers", type=int, default=8)
 495:     parser.add_argument("--seed", type=int, default=42)
 496:     args = parser.parse_args()
 497:     args.run_start_time = time.time()
 498: 
 499:     torch.set_float32_matmul_precision("high")
 500: 
 501:     metric_rows = []
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 107–118

```python
 107:     return float(average_precision_score(y_true, y_score))
 108: 
 109: 
 110: def extract_score(batch, index, anomaly_map):
 111:     for field in ["pred_score", "pred_scores", "anomaly_score", "image_score", "score"]:
 112:         try:
 113:             values = get_field(batch, field)
 114:             value = take_item(values, index)
 115:             if hasattr(value, "detach"):
 116:                 value = value.detach().cpu().item()
 117:             elif hasattr(value, "item"):
 118:                 value = value.item()
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 145–158

```python
 145:         normal_dir="train/good",
 146:         abnormal_dir=["test/anomaly"],
 147:         normal_test_dir="test/good",
 148:         mask_dir=["ground_truth/anomaly"],
 149:         train_batch_size=args.train_batch_size,
 150:         eval_batch_size=args.eval_batch_size,
 151:         num_workers=args.num_workers,
 152:         seed=args.seed,
 153:         val_split_mode="same_as_test",
 154:     )
 155: 
 156:     return datamodule
 157: 
 158: 
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 213–226

```python
 213:         image_paths = get_field(batch, "image_path")
 214:         gt_masks = get_field(batch, "gt_mask")
 215:         anomaly_maps = get_field(batch, "anomaly_map")
 216: 
 217:         batch_size = len(image_paths) if isinstance(image_paths, (list, tuple)) else 1
 218: 
 219:         for i in range(batch_size):
 220:             image_path = str(take_item(image_paths, i))
 221:             y_image = infer_label_from_path(image_path)
 222: 
 223:             anomaly_map = normalize_map(take_item(anomaly_maps, i))
 224:             gt_mask = mask_to_2d(take_item(gt_masks, i))
 225: 
 226:             if anomaly_map is None:
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 266–277

```python
 266: 
 267:     pixel_auroc = safe_roc_auc(pixel_labels_np, pixel_scores_np)
 268:     pixel_ap = safe_ap(pixel_labels_np, pixel_scores_np)
 269: 
 270:     for item in per_image_cache:
 271:         pred_image_label = int(item["image_score"] >= image_thr)
 272: 
 273:         image_records.append(
 274:             {
 275:                 "dataset": "VisA",
 276:                 "category": category,
 277:                 "image_path": item["image_path"],
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 416–429

```python
 416:     parser.add_argument("--image_size", type=int, default=256)
 417:     parser.add_argument("--center_crop_size", type=int, default=224)
 418:     parser.add_argument("--top_components", type=int, default=3)
 419:     parser.add_argument("--min_area", type=int, default=20)
 420:     parser.add_argument("--train_batch_size", type=int, default=32)
 421:     parser.add_argument("--eval_batch_size", type=int, default=32)
 422:     parser.add_argument("--num_workers", type=int, default=8)
 423:     parser.add_argument("--seed", type=int, default=42)
 424:     args = parser.parse_args()
 425: 
 426:     torch.set_float32_matmul_precision("high")
 427: 
 428:     metric_rows = []
 429:     coverage_rows = []
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 143–159

```python
 143: 
 144:     df = df[pd.to_numeric(df["component_rank"], errors="coerce") > 0].copy()
 145:     boxes = {}
 146: 
 147:     for image_path, group in df.groupby("image_path"):
 148:         key = canonical_path(image_path)
 149:         group = group.sort_values("component_rank").head(top_k)
 150: 
 151:         box_list = []
 152:         for _, row in group.iterrows():
 153:             box_list.append(
 154:                 {
 155:                     "x1": int(row["x1"]),
 156:                     "y1": int(row["y1"]),
 157:                     "x2": int(row["x2"]),
 158:                     "y2": int(row["y2"]),
 159:                     "rank": int(row["component_rank"]),
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 238–249

```python
 238:     raise ValueError(f"Unknown eval mode: {eval_mode}")
 239: 
 240: 
 241: def encode_images(model, preprocess, images, device):
 242:     batch = torch.cat([preprocess(img).unsqueeze(0) for img in images], dim=0).to(device)
 243: 
 244:     with torch.no_grad():
 245:         features = model.encode_image(batch)
 246:         features = features / features.norm(dim=-1, keepdim=True)
 247: 
 248:     return features
 249: 
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 313–324

```python
 313:     detail_rows = []
 314:     fallback_count = 0
 315:     covered_count = 0
 316: 
 317:     for _, row in df.iterrows():
 318:         eval_images, used_mode, fallback = get_eval_images(row, boxes, eval_mode, args)
 319: 
 320:         if fallback:
 321:             fallback_count += 1
 322:         else:
 323:             covered_count += 1
 324: 
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 143–159

```python
 143: 
 144:     df = df[pd.to_numeric(df["component_rank"], errors="coerce") > 0].copy()
 145:     boxes = {}
 146: 
 147:     for image_path, group in df.groupby("image_path"):
 148:         key = canonical_path(image_path)
 149:         group = group.sort_values("component_rank").head(top_k)
 150: 
 151:         box_list = []
 152:         for _, row in group.iterrows():
 153:             box_list.append(
 154:                 {
 155:                     "x1": int(row["x1"]),
 156:                     "y1": int(row["y1"]),
 157:                     "x2": int(row["x2"]),
 158:                     "y2": int(row["y2"]),
 159:                     "rank": int(row["component_rank"]),
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 238–249

```python
 238:     raise ValueError(f"Unknown eval mode: {eval_mode}")
 239: 
 240: 
 241: def encode_images(model, preprocess, images, device):
 242:     batch = torch.cat([preprocess(img).unsqueeze(0) for img in images], dim=0).to(device)
 243: 
 244:     with torch.no_grad():
 245:         features = model.encode_image(batch)
 246:         features = features / features.norm(dim=-1, keepdim=True)
 247: 
 248:     return features
 249: 
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 313–324

```python
 313:     detail_rows = []
 314:     fallback_count = 0
 315:     covered_count = 0
 316: 
 317:     for _, row in df.iterrows():
 318:         eval_images, used_mode, fallback = get_eval_images(row, boxes, eval_mode, args)
 319: 
 320:         if fallback:
 321:             fallback_count += 1
 322:         else:
 323:             covered_count += 1
 324: 
```

### `experiments/stage9_qcr_u/run_stage9_a1_qcr_u_fusion.py` lines 523–546

```python
 523:     lines.append("")
 524:     lines.append("| Backbone | Strategy | Eval mode | Fusion | AUROC | AP | Best F1 | Delta AUROC vs VLM |")
 525:     lines.append("|---|---|---|---|---:|---:|---:|---:|")
 526: 
 527:     for _, r in best.iterrows():
 528:         lines.append(
 529:             f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['fusion_method']} | "
 530:             f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['delta_auroc_vs_vlm']:.4f} |"
 531:         )
 532: 
 533:     lines.append("")
 534:     lines.append("## 5. Best QCR-U Rows")
 535:     lines.append("")
 536:     lines.append("| Backbone | Strategy | Eval mode | Fusion | AUROC | AP | Best F1 | Delta AUROC vs VLM |")
 537:     lines.append("|---|---|---|---|---:|---:|---:|---:|")
 538: 
 539:     for _, r in qcr_best.iterrows():
 540:         lines.append(
 541:             f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['fusion_method']} | "
 542:             f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['delta_auroc_vs_vlm']:.4f} |"
 543:         )
 544: 
 545:     lines.append("")
 546:     lines.append("## 6. Interpretation Boundary")
```

### `experiments/stage9_qcr_u/run_stage9_a2_qcr_u_sanity_check.py` lines 382–393

```python
 382:     lines.append("")
 383:     lines.append("| Backbone | Strategy | Eval mode | Fusion | Macro AUROC | Macro AP | Macro F1 | Positive categories | Negative categories | Mean ΔAUROC vs VLM |")
 384:     lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
 385: 
 386:     for _, r in tables["best_macro_qcr"].iterrows():
 387:         lines.append(
 388:             f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['fusion_method']} | "
 389:             f"{r['macro_auroc']:.4f} | {r['macro_ap']:.4f} | {r['macro_best_f1']:.4f} | "
 390:             f"{int(r['num_categories_delta_positive'])} | {int(r['num_categories_delta_negative'])} | "
 391:             f"{r['mean_delta_auroc_vs_vlm']:.4f} |"
 392:         )
 393: 
```

### `experiments/stage9_qcr_u/run_stage9_a2_qcr_u_sanity_check.py` lines 396–407

```python
 396:     lines.append("")
 397:     lines.append("| Backbone | Strategy | Eval mode | Macro AUROC | Mean ΔAUROC vs VLM |")
 398:     lines.append("|---|---|---|---:|---:|")
 399: 
 400:     for _, r in tables["candidate_quality_macro"].iterrows():
 401:         lines.append(
 402:             f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | "
 403:             f"{r['macro_auroc']:.4f} | {r['mean_delta_auroc_vs_vlm']:.4f} |"
 404:         )
 405: 
 406:     lines.append("")
 407:     lines.append("## 6. QCR-U Negative-delta Categories")
```

### `experiments/stage9_qcr_u/run_stage9_a2_qcr_u_sanity_check.py` lines 412–435

```python
 412:     neg = tables["qcr_negative_categories"]
 413:     if neg.empty:
 414:         lines.append("| - | - | - | - | - | - | - | - |")
 415:     else:
 416:         for _, r in neg.iterrows():
 417:             lines.append(
 418:                 f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['category']} | {r['fusion_method']} | "
 419:                 f"{r['auroc']:.4f} | {r['vlm_only_auroc']:.4f} | {r['delta_auroc_vs_vlm']:.4f} |"
 420:             )
 421: 
 422:     lines.append("")
 423:     lines.append("## 7. Strongest Candidate-quality Separation")
 424:     lines.append("")
 425:     lines.append("| Backbone | Strategy | Eval mode | Category | Candidate rate normal | Candidate rate anomaly | Q normal | Q anomaly | corr(Q,y) |")
 426:     lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
 427: 
 428:     for _, r in tables["q_dominant_categories"].iterrows():
 429:         lines.append(
 430:             f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['category']} | "
 431:             f"{r['candidate_rate_normal']:.4f} | {r['candidate_rate_anomaly']:.4f} | "
 432:             f"{r['q_mean_normal']:.4f} | {r['q_mean_anomaly']:.4f} | {r['corr_q_y']:.4f} |"
 433:         )
 434: 
 435:     lines.append("")
```

### `experiments/stage9_qcr_u/run_stage9_a3_qcr_u_debias_check.py` lines 433–456

```python
 433:     lines.append("")
 434:     lines.append("| Backbone | Strategy | Eval mode | Method | AUROC | AP | Best F1 | ΔAUROC vs VLM |")
 435:     lines.append("|---|---|---|---|---:|---:|---:|---:|")
 436: 
 437:     for _, r in best_debiased.iterrows():
 438:         lines.append(
 439:             f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['debiased_method']} | "
 440:             f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['delta_auroc_vs_vlm']:.4f} |"
 441:         )
 442: 
 443:     lines.append("")
 444:     lines.append("## 5. Candidate Quality Original vs Neutral")
 445:     lines.append("")
 446:     lines.append("| Backbone | Strategy | Eval mode | Method | AUROC | AP | Best F1 | ΔAUROC vs VLM |")
 447:     lines.append("|---|---|---|---|---:|---:|---:|---:|")
 448: 
 449:     for _, r in best_candidate.iterrows():
 450:         lines.append(
 451:             f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['debiased_method']} | "
 452:             f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['delta_auroc_vs_vlm']:.4f} |"
 453:         )
 454: 
 455:     lines.append("")
 456:     lines.append("## 6. Stability Counts")
```

## timing

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py` lines 29–40

```python
  29: 
  30: def read_csv_robust(path: Path) -> pd.DataFrame:
  31:     df = pd.read_csv(path)
  32:     if len(df.columns) <= 1:
  33:         raise RuntimeError(f"{path} read as <=1 column. Fix CSV line breaks before running Stage 16-A2.")
  34:     return df
  35: 
  36: 
  37: def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
  38:     idx = ["backbone", "dataset", "strategy", "eval_mode"]
  39:     piv = df.pivot_table(index=idx, columns="variant_id", values=metric, aggfunc="first").reset_index()
  40:     piv.columns.name = None
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py` lines 46–57

```python
  46: 
  47:     required = ["V0", "V2", "V3", "V4", "V5"]
  48:     missing = [c for c in required if c not in piv.columns]
  49:     if missing:
  50:         raise RuntimeError(f"Missing variant columns in pivot: {missing}")
  51: 
  52:     out = piv.copy()
  53:     out["delta_v5_minus_v3_naive"] = out["V5"] - out["V3"]
  54:     out["delta_v5_minus_v4_quality"] = out["V5"] - out["V4"]
  55:     out["delta_v5_minus_v0_detector"] = out["V5"] - out["V0"]
  56:     out["delta_v5_minus_v2_crop"] = out["V5"] - out["V2"]
  57:     out["delta_v4_minus_v3_naive"] = out["V4"] - out["V3"]
```

### `experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py` lines 46–57

```python
  46: 
  47: def read_csv_strict(path: Path) -> pd.DataFrame:
  48:     df = pd.read_csv(path)
  49:     if len(df.columns) <= 1:
  50:         raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
  51:     return df
  52: 
  53: 
  54: def minmax_safe(x: pd.Series) -> pd.Series:
  55:     x = pd.to_numeric(x, errors="coerce").astype(float)
  56:     lo = x.min()
  57:     hi = x.max()
```

### `experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py` lines 165–176

```python
 165: 
 166: def build_base_table(df: pd.DataFrame) -> pd.DataFrame:
 167:     missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
 168:     if missing:
 169:         raise RuntimeError(f"Missing required columns: {missing}")
 170: 
 171:     base_cols = [
 172:         "backbone",
 173:         "dataset",
 174:         "category",
 175:         "strategy",
 176:         "eval_mode",
```

### `experiments/stage16_qcru_ablation/build_stage16_c_final_method_claims.py` lines 20–31

```python
  20:     if not path.exists():
  21:         raise FileNotFoundError(path)
  22:     df = pd.read_csv(path)
  23:     if len(df.columns) <= 1:
  24:         raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
  25:     return df
  26: 
  27: 
  28: def get_primary_delta(primary: pd.DataFrame, left: str, right: str) -> dict:
  29:     idx = ["backbone", "dataset", "strategy", "eval_mode"]
  30:     piv = primary.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
  31:     piv.columns.name = None
```

### `experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py` lines 41–52

```python
  41:         raw = known_header + "\n" + "\n".join(rows) + "\n"
  42: 
  43:     df = pd.read_csv(StringIO(raw))
  44:     if len(df.columns) <= 1:
  45:         raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
  46:     return df
  47: 
  48: 
  49: def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
  50:     for c in cols:
  51:         if c in df.columns:
  52:             df[c] = pd.to_numeric(df[c], errors="coerce")
```

### `experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py` lines 58–69

```python
  58:     df = to_num(df, ["image_auroc", "image_ap", "pixel_auroc", "pixel_f1"])
  59: 
  60:     mean = df[df["category"] == "MEAN"].copy()
  61:     if mean.empty:
  62:         raise RuntimeError("Stage 15-E has no MEAN rows.")
  63: 
  64:     rows = []
  65:     for _, r in mean.iterrows():
  66:         method = str(r["method"])
  67:         fairness_tag = str(r.get("fairness_tag", ""))
  68:         protocol = str(r.get("protocol", ""))
  69: 
```

### `experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py` lines 45–56

```python
  45:     if not path.exists():
  46:         raise FileNotFoundError(path)
  47:     df = pd.read_csv(path)
  48:     if len(df.columns) <= 1:
  49:         raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
  50:     return df
  51: 
  52: 
  53: def minmax_safe(x: pd.Series) -> pd.Series:
  54:     x = pd.to_numeric(x, errors="coerce").astype(float)
  55:     lo = x.min()
  56:     hi = x.max()
```

### `experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py` lines 129–140

```python
 129: 
 130: def build_base_table(df: pd.DataFrame) -> pd.DataFrame:
 131:     missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
 132:     if missing:
 133:         raise RuntimeError(f"Missing required columns: {missing}")
 134: 
 135:     optional_cols = [
 136:         "fallback",
 137:         "has_candidate",
 138:         "num_candidates",
 139:         "image_path",
 140:         "gt_label",
```

### `experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py` lines 26–37

```python
  26:     if not path.exists():
  27:         raise FileNotFoundError(path)
  28:     df = pd.read_csv(path)
  29:     if len(df.columns) <= 1:
  30:         raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
  31:     return df
  32: 
  33: 
  34: def safe_float(x, default=None):
  35:     try:
  36:         if pd.isna(x):
  37:             return default
```

### `experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py` lines 36–47

```python
  36: 
  37: def read_csv_strict(path: Path) -> pd.DataFrame:
  38:     df = pd.read_csv(path)
  39:     if len(df.columns) <= 1:
  40:         raise RuntimeError(
  41:             f"{path} was read as <=1 column. Local CSV formatting is broken; repair line breaks before Stage 16-A1."
  42:         )
  43:     return df
  44: 
  45: 
  46: def minmax_safe(x: pd.Series) -> pd.Series:
  47:     x = pd.to_numeric(x, errors="coerce").astype(float)
```

### `experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py` lines 175–186

```python
 175: 
 176: def build_base_table(df: pd.DataFrame) -> pd.DataFrame:
 177:     missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
 178:     if missing:
 179:         raise RuntimeError(f"Missing required columns: {missing}")
 180: 
 181:     # Stage 9 prediction table has many duplicated image rows because fusion weights/methods vary.
 182:     # Base signals are image-level and should be identical across those rows, so we deduplicate.
 183:     base_cols = [
 184:         "backbone",
 185:         "dataset",
 186:         "category",
```

### `experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py` lines 35–46

```python
  35: 
  36: def read_csv_strict(path: Path) -> pd.DataFrame:
  37:     df = pd.read_csv(path)
  38:     if len(df.columns) <= 1:
  39:         raise RuntimeError(f"{path} read as <=1 column. Fix CSV line breaks before running Stage 16-A3.")
  40:     return df
  41: 
  42: 
  43: def minmax_safe(x: pd.Series) -> pd.Series:
  44:     x = pd.to_numeric(x, errors="coerce").astype(float)
  45:     lo = x.min()
  46:     hi = x.max()
```

### `experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py` lines 154–165

```python
 154: 
 155: def build_base_table(df: pd.DataFrame) -> pd.DataFrame:
 156:     missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
 157:     if missing:
 158:         raise RuntimeError(f"Missing required columns: {missing}")
 159: 
 160:     base_cols = [
 161:         "backbone",
 162:         "dataset",
 163:         "category",
 164:         "strategy",
 165:         "eval_mode",
```

### `experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py` lines 279–290

```python
 279:     piv.columns.name = None
 280: 
 281:     for col in ["V3", "V4", "V5", "V6"]:
 282:         if col not in piv.columns:
 283:             raise RuntimeError(f"Missing {col} in per-config pivot.")
 284: 
 285:     piv["delta_v6_minus_v3_naive"] = piv["V6"] - piv["V3"]
 286:     piv["delta_v6_minus_v4_quality"] = piv["V6"] - piv["V4"]
 287:     piv["delta_v6_minus_v5_fixed_qc"] = piv["V6"] - piv["V5"]
 288:     piv["delta_v5_minus_v4_quality"] = piv["V5"] - piv["V4"]
 289: 
 290:     piv["v6_beats_naive"] = piv["delta_v6_minus_v3_naive"] > 0
```

### `experiments/stage7_generalization/progress_utils.py` lines 43–54

```python
  43:         self.category = category
  44:         self.category_index = max(1, int(category_index))
  45:         self.total_categories = max(1, int(total_categories))
  46:         self.refresh_interval = float(refresh_interval)
  47:         self.run_start_time = run_start_time or time.time()
  48:         self.fit_weight = float(fit_weight)
  49: 
  50:         self.stage = ""
  51:         self.total = 1
  52:         self.start_time = None
  53:         self.last_update = 0.0
  54:         self.epoch_index = 1
```

### `experiments/stage7_generalization/progress_utils.py` lines 74–109

```python
  74: 
  75:         return frac
  76: 
  77:     def _overall_eta(self, done, total):
  78:         now = time.time()
  79:         total_elapsed = now - self.run_start_time
  80: 
  81:         cat_frac = self._category_fraction(done, total)
  82:         overall_done = (self.category_index - 1 + cat_frac) / float(self.total_categories)
  83:         overall_done = max(1e-6, min(1.0, overall_done))
  84: 
  85:         total_estimated = total_elapsed / overall_done
  86:         return max(0.0, total_estimated - total_elapsed)
  87: 
  88:     def _print(self, done, total, force=False):
  89:         now = time.time()
  90:         if not force and (now - self.last_update) < self.refresh_interval:
  91:             return
  92: 
  93:         self.last_update = now
  94: 
  95:         if self.start_time is None:
  96:             self.start_time = now
  97: 
  98:         done = max(0, int(done))
  99:         total = _safe_total(total)
 100: 
 101:         elapsed = now - self.start_time
 102:         sec_per_batch = elapsed / max(done, 1)
 103:         stage_eta = max(total - done, 0) * sec_per_batch
 104:         total_eta = self._overall_eta(done, total)
 105:         percent = 100.0 * done / total
 106: 
 107:         if self.stage == "fit":
 108:             stage_name = f"fit epoch {self.epoch_index}/{self.max_epochs}"
 109:         else:
```

### `experiments/stage7_generalization/progress_utils.py` lines 124–135

```python
 124: 
 125:     def _start(self, stage, total):
 126:         self.stage = stage
 127:         self.total = _safe_total(total)
 128:         self.start_time = time.time()
 129:         self.last_update = 0.0
 130:         self._clear_line()
 131:         self._print(0, self.total, force=True)
 132: 
 133:     def _finish(self):
 134:         if self.total:
 135:             self._print(self.total, self.total, force=True)
```

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 493–504

```python
 493:     parser.add_argument("--eval_batch_size", type=int, default=32)
 494:     parser.add_argument("--num_workers", type=int, default=8)
 495:     parser.add_argument("--seed", type=int, default=42)
 496:     args = parser.parse_args()
 497:     args.run_start_time = time.time()
 498: 
 499:     torch.set_float32_matmul_precision("high")
 500: 
 501:     metric_rows = []
 502:     coverage_rows = []
 503: 
 504:     for category_index, category in enumerate(args.categories, start=1):
```

## device

### `experiments/stage7_generalization/run_visa_multibackbone_baseline_and_candidates.py` lines 240–252

```python
 240:     )
 241: 
 242:     engine = Engine(
 243:         default_root_dir=str(work_dir),
 244:         accelerator="gpu" if torch.cuda.is_available() else "cpu",
 245:         devices=1,
 246:         logger=False,
 247:         enable_progress_bar=False,
 248:         enable_model_summary=False,
 249:         callbacks=[progress_callback],
 250:         max_epochs=args.max_epochs,
 251:         limit_train_batches=args.limit_train_batches,
 252:         limit_predict_batches=args.limit_predict_batches,
```

### `experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py` lines 179–191

```python
 179:     progress_callback = OneLineProgressCallback(category=category)
 180: 
 181:     engine = Engine(
 182:         default_root_dir=str(work_dir),
 183:         accelerator="gpu" if torch.cuda.is_available() else "cpu",
 184:         devices=1,
 185:         logger=False,
 186:         enable_progress_bar=False,
 187:         enable_model_summary=False,
 188:         callbacks=[progress_callback],
 189:     )
 190: 
 191:     print(f"[INFO] Fitting PatchCore on VisA category: {category}")
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 97–124

```python
  97: 
  98:     raise ValueError(f"Unknown prompt strategy: {strategy}")
  99: 
 100: 
 101: def encode_prompt_set(model, tokenizer, prompts, device):
 102:     tokens = tokenizer(prompts).to(device)
 103: 
 104:     with torch.no_grad():
 105:         features = model.encode_text(tokens)
 106:         features = features / features.norm(dim=-1, keepdim=True)
 107: 
 108:     feature = features.mean(dim=0, keepdim=True)
 109:     feature = feature / feature.norm(dim=-1, keepdim=True)
 110:     return feature
 111: 
 112: 
 113: def build_text_features(model, tokenizer, category, strategy, device):
 114:     normal_prompts, anomaly_prompts = build_prompts(category, strategy)
 115: 
 116:     normal_feature = encode_prompt_set(model, tokenizer, normal_prompts, device)
 117:     anomaly_feature = encode_prompt_set(model, tokenizer, anomaly_prompts, device)
 118: 
 119:     text_features = torch.cat([normal_feature, anomaly_feature], dim=0)
 120: 
 121:     prompt_row = {
 122:         "category": category,
 123:         "strategy": strategy,
 124:         "normal_prompts": " || ".join(normal_prompts),
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 237–249

```python
 237: 
 238:     raise ValueError(f"Unknown eval mode: {eval_mode}")
 239: 
 240: 
 241: def encode_images(model, preprocess, images, device):
 242:     batch = torch.cat([preprocess(img).unsqueeze(0) for img in images], dim=0).to(device)
 243: 
 244:     with torch.no_grad():
 245:         features = model.encode_image(batch)
 246:         features = features / features.norm(dim=-1, keepdim=True)
 247: 
 248:     return features
 249: 
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 289–300

```python
 289: 
 290:     return best_thr, best_f1, best_acc
 291: 
 292: 
 293: def evaluate_category(args, model, preprocess, tokenizer, device, category, strategy, eval_mode):
 294:     pred_csv = Path(args.patchcore_root) / "VisA" / category / "patchcore_image_predictions.csv"
 295: 
 296:     if not pred_csv.exists():
 297:         raise FileNotFoundError(f"Missing PatchCore image prediction CSV: {pred_csv}")
 298: 
 299:     df = pd.read_csv(pred_csv)
 300:     df = df[df["label"].isin(["normal", "anomaly"])].copy().reset_index(drop=True)
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 304–315

```python
 304:         model=model,
 305:         tokenizer=tokenizer,
 306:         category=category,
 307:         strategy=strategy,
 308:         device=device,
 309:     )
 310: 
 311:     y_true = []
 312:     anomaly_scores = []
 313:     detail_rows = []
 314:     fallback_count = 0
 315:     covered_count = 0
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 321–332

```python
 321:             fallback_count += 1
 322:         else:
 323:             covered_count += 1
 324: 
 325:         image_features = encode_images(model, preprocess, eval_images, device)
 326:         sims_matrix = (image_features @ text_features.T).detach().cpu().numpy()
 327: 
 328:         # text index 0 = normal, 1 = anomaly.
 329:         # for top-k crops, use max anomaly margin over crops.
 330:         margins = sims_matrix[:, 1] - sims_matrix[:, 0]
 331:         anomaly_score = float(np.max(margins))
 332: 
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 386–413

```python
 386:     parser.add_argument("--strategies", nargs="+", default=["generic_binary", "category_binary", "inspection_binary"])
 387:     parser.add_argument("--eval_modes", nargs="+", default=["full_all", "crop_or_full", "crop_topk_ensemble"])
 388:     parser.add_argument("--clip_model", type=str, default="ViT-B-32")
 389:     parser.add_argument("--clip_pretrained", type=str, default="openai")
 390:     parser.add_argument("--device", type=str, default="")
 391:     parser.add_argument("--top_k", type=int, default=3)
 392:     parser.add_argument("--map_size", type=int, default=224)
 393:     parser.add_argument("--crop_padding", type=int, default=12)
 394:     parser.add_argument("--min_crop_size", type=int, default=48)
 395:     args = parser.parse_args()
 396: 
 397:     output_root = Path(args.output_root)
 398:     output_root.mkdir(parents=True, exist_ok=True)
 399: 
 400:     device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
 401: 
 402:     print(f"[INFO] Loading CLIP: {args.clip_model}, pretrained={args.clip_pretrained}, device={device}")
 403:     model, _, preprocess = open_clip.create_model_and_transforms(
 404:         args.clip_model,
 405:         pretrained=args.clip_pretrained,
 406:         device=device,
 407:     )
 408:     tokenizer = open_clip.get_tokenizer(args.clip_model)
 409:     model.eval()
 410: 
 411:     summary_rows = []
 412:     detail_rows = []
 413:     prompt_rows = []
```

### `experiments/stage7_generalization/visa_binary_prompt_reasoning.py` lines 421–432

```python
 421:                     args=args,
 422:                     model=model,
 423:                     preprocess=preprocess,
 424:                     tokenizer=tokenizer,
 425:                     device=device,
 426:                     category=category,
 427:                     strategy=strategy,
 428:                     eval_mode=eval_mode,
 429:                 )
 430: 
 431:                 summary_rows.append(summary)
 432:                 detail_rows.extend(details)
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 97–124

```python
  97: 
  98:     raise ValueError(f"Unknown prompt strategy: {strategy}")
  99: 
 100: 
 101: def encode_prompt_set(model, tokenizer, prompts, device):
 102:     tokens = tokenizer(prompts).to(device)
 103: 
 104:     with torch.no_grad():
 105:         features = model.encode_text(tokens)
 106:         features = features / features.norm(dim=-1, keepdim=True)
 107: 
 108:     feature = features.mean(dim=0, keepdim=True)
 109:     feature = feature / feature.norm(dim=-1, keepdim=True)
 110:     return feature
 111: 
 112: 
 113: def build_text_features(model, tokenizer, category, strategy, device):
 114:     normal_prompts, anomaly_prompts = build_prompts(category, strategy)
 115: 
 116:     normal_feature = encode_prompt_set(model, tokenizer, normal_prompts, device)
 117:     anomaly_feature = encode_prompt_set(model, tokenizer, anomaly_prompts, device)
 118: 
 119:     text_features = torch.cat([normal_feature, anomaly_feature], dim=0)
 120: 
 121:     prompt_row = {
 122:         "category": category,
 123:         "strategy": strategy,
 124:         "normal_prompts": " || ".join(normal_prompts),
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 237–249

```python
 237: 
 238:     raise ValueError(f"Unknown eval mode: {eval_mode}")
 239: 
 240: 
 241: def encode_images(model, preprocess, images, device):
 242:     batch = torch.cat([preprocess(img).unsqueeze(0) for img in images], dim=0).to(device)
 243: 
 244:     with torch.no_grad():
 245:         features = model.encode_image(batch)
 246:         features = features / features.norm(dim=-1, keepdim=True)
 247: 
 248:     return features
 249: 
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 289–300

```python
 289: 
 290:     return best_thr, best_f1, best_acc
 291: 
 292: 
 293: def evaluate_category(args, model, preprocess, tokenizer, device, category, strategy, eval_mode):
 294:     pred_csv = Path(args.candidate_root) / "VisA" / category / f"{args.backbone_model}_image_predictions.csv"
 295: 
 296:     if not pred_csv.exists():
 297:         raise FileNotFoundError(f"Missing PatchCore image prediction CSV: {pred_csv}")
 298: 
 299:     df = pd.read_csv(pred_csv)
 300:     df = df[df["label"].isin(["normal", "anomaly"])].copy().reset_index(drop=True)
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 304–315

```python
 304:         model=model,
 305:         tokenizer=tokenizer,
 306:         category=category,
 307:         strategy=strategy,
 308:         device=device,
 309:     )
 310: 
 311:     y_true = []
 312:     anomaly_scores = []
 313:     detail_rows = []
 314:     fallback_count = 0
 315:     covered_count = 0
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 321–332

```python
 321:             fallback_count += 1
 322:         else:
 323:             covered_count += 1
 324: 
 325:         image_features = encode_images(model, preprocess, eval_images, device)
 326:         sims_matrix = (image_features @ text_features.T).detach().cpu().numpy()
 327: 
 328:         # text index 0 = normal, 1 = anomaly.
 329:         # for top-k crops, use max anomaly margin over crops.
 330:         margins = sims_matrix[:, 1] - sims_matrix[:, 0]
 331:         anomaly_score = float(np.max(margins))
 332: 
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 395–422

```python
 395:     parser.add_argument("--strategies", nargs="+", default=["generic_binary", "category_binary", "inspection_binary"])
 396:     parser.add_argument("--eval_modes", nargs="+", default=["full_all", "crop_or_full", "crop_topk_ensemble"])
 397:     parser.add_argument("--clip_model", type=str, default="ViT-B-32")
 398:     parser.add_argument("--clip_pretrained", type=str, default="openai")
 399:     parser.add_argument("--device", type=str, default="")
 400:     parser.add_argument("--top_k", type=int, default=3)
 401:     parser.add_argument("--map_size", type=int, default=224)
 402:     parser.add_argument("--crop_padding", type=int, default=12)
 403:     parser.add_argument("--min_crop_size", type=int, default=48)
 404:     args = parser.parse_args()
 405: 
 406:     output_root = Path(args.output_root)
 407:     output_root.mkdir(parents=True, exist_ok=True)
 408: 
 409:     device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
 410: 
 411:     print(f"[INFO] Loading CLIP: {args.clip_model}, pretrained={args.clip_pretrained}, device={device}")
 412:     model, _, preprocess = open_clip.create_model_and_transforms(
 413:         args.clip_model,
 414:         pretrained=args.clip_pretrained,
 415:         device=device,
 416:     )
 417:     tokenizer = open_clip.get_tokenizer(args.clip_model)
 418:     model.eval()
 419: 
 420:     summary_rows = []
 421:     detail_rows = []
 422:     prompt_rows = []
```

### `experiments/stage7_generalization/visa_multibackbone_binary_prompt_reasoning.py` lines 430–441

```python
 430:                     args=args,
 431:                     model=model,
 432:                     preprocess=preprocess,
 433:                     tokenizer=tokenizer,
 434:                     device=device,
 435:                     category=category,
 436:                     strategy=strategy,
 437:                     eval_mode=eval_mode,
 438:                 )
 439: 
 440:                 summary_rows.append(summary)
 441:                 detail_rows.extend(details)
```
