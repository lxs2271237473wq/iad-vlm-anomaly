# Stage 22-A1c：旧 QCR 与 VisA 缓存精简追踪

该报告只提取后续冻结 Selective QCR 协议所需的信息。

- 扫描 Python 文件：`18`
- 未运行检测器或 VLM
- 未修改已有实验结果

## 1. 旧 QCR：Quality 计算代码

**函数 `main`：`experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 70–271 行**

```python
  70: def main() -> None:
  71:     OUT_DIR.mkdir(parents=True, exist_ok=True)
  72:     DOC_DIR.mkdir(parents=True, exist_ok=True)
  73: 
  74:     inventory_rows = [inspect_csv(ROOT / p) for p in SOURCE_PATHS]
  75:     inventory = pd.DataFrame(inventory_rows)
  76:     inventory.to_csv(OUT_INVENTORY, index=False, lineterminator="\n")
  77: 
  78:     plan_rows = [
  79:         {
  80:             "variant_id": "V0",
  81:             "variant": "detector_only",
  82:             "uses_detector_score": True,
  83:             "uses_full_image_vlm": False,
  84:             "uses_crop_vlm": False,
  85:             "uses_quality": False,
  86:             "uses_consistency": False,
  87:             "uses_unknown": False,
  88:             "purpose": "Anchor baseline; proves whether QCR-U beats the detector alone.",
  89:         },
  90:         {
  91:             "variant_id": "V1",
  92:             "variant": "full_image_vlm",
  93:             "uses_detector_score": False,
  94:             "uses_full_image_vlm": True,
  95:             "uses_crop_vlm": False,
  96:             "uses_quality": False,
  97:             "uses_consistency": False,
  98:             "uses_unknown": False,
  99:             "purpose": "Weak VLM sanity baseline; should not be the main comparison target.",
 100:         },
 101:         {
 102:             "variant_id": "V2",
 103:             "variant": "crop_topk_vlm",
 104:             "uses_detector_score": False,
 105:             "uses_full_image_vlm": False,
 106:             "uses_crop_vlm": True,
 107:             "uses_quality": False,
 108:             "uses_consistency": False,
 109:             "uses_unknown": False,
 110:             "purpose": "Tests whether localization-guided crops improve VLM scoring.",
 111:         },
 112:         {
 113:             "variant_id": "V3",
 114:             "variant": "naive_detector_crop_fusion",
 115:             "uses_detector_score": True,
 116:             "uses_full_image_vlm": False,
 117:             "uses_crop_vlm": True,
 118:             "uses_quality": False,
 119:             "uses_consistency": False,
 120:             "uses_unknown": False,
 121:             "purpose": "Naive fusion baseline; QCR-U must beat this or the method is not justified.",
 122:         },
 123:         {
 124:             "variant_id": "V4",
 125:             "variant": "quality_weighted_crop",
 126:             "uses_detector_score": True,
 127:             "uses_full_image_vlm": False,
 128:             "uses_crop_vlm": True,
 129:             "uses_quality": True,
 130:             "uses_consistency": False,
 131:             "uses_unknown": False,
 132:             "purpose": "Tests whether candidate quality contributes beyond crop scoring.",
 133:         },
 134:         {
 135:             "variant_id": "V5",
 136:             "variant": "quality_consistency_fusion",
 137:             "uses_detector_score": True,
 138:             "uses_full_image_vlm": False,
 139:             "uses_crop_vlm": True,
 140:             "uses_quality": True,
 141:             "uses_consistency": True,
 142:             "uses_unknown": False,
 143:             "purpose": "Core QCR-U binary anomaly recognition variant.",
 144:         },
 145:         {
 146:             "variant_id": "V6",
 147:             "variant": "qcr_u_full_optional_unknown",
 148:             "uses_detector_score": True,
 149:             "uses_full_image_vlm": False,
... 函数剩余 122 行已省略 ...
```

**函数 `make_summary`：`experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 75–131 行**

```python
  75: def make_summary(delta: pd.DataFrame) -> pd.DataFrame:
  76:     rows = []
  77: 
  78:     checks = [
  79:         ("V5 > V3 naive fusion", "v5_beats_naive", "delta_v5_minus_v3_naive"),
  80:         ("V5 > V4 quality-only", "v5_beats_quality_only", "delta_v5_minus_v4_quality"),
  81:         ("V5 > V0 detector-only", "v5_beats_detector", "delta_v5_minus_v0_detector"),
  82:         ("V5 > V2 crop-VLM-only", "v5_beats_crop", "delta_v5_minus_v2_crop"),
  83:         ("V4 > V3 naive fusion", "quality_beats_naive", "delta_v4_minus_v3_naive"),
  84:     ]
  85: 
  86:     for name, bool_col, delta_col in checks:
  87:         wins, total, rate = summarize_boolean(delta, bool_col)
  88:         rows.append(
  89:             {
  90:                 "check": name,
  91:                 "wins": wins,
  92:                 "total_protocols": total,
  93:                 "win_rate": rate,
  94:                 "mean_delta": delta[delta_col].mean(),
  95:                 "median_delta": delta[delta_col].median(),
  96:                 "min_delta": delta[delta_col].min(),
  97:                 "max_delta": delta[delta_col].max(),
  98:             }
  99:         )
 100: 
 101:     # By eval mode, because full_all may behave differently from crop protocols.
 102:     for eval_mode, g in delta.groupby("eval_mode"):
 103:         wins, total, rate = summarize_boolean(g, "v5_beats_naive")
 104:         rows.append(
 105:             {
 106:                 "check": f"V5 > V3 naive fusion by eval_mode={eval_mode}",
 107:                 "wins": wins,
 108:                 "total_protocols": total,
 109:                 "win_rate": rate,
 110:                 "mean_delta": g["delta_v5_minus_v3_naive"].mean(),
 111:                 "median_delta": g["delta_v5_minus_v3_naive"].median(),
 112:                 "min_delta": g["delta_v5_minus_v3_naive"].min(),
 113:                 "max_delta": g["delta_v5_minus_v3_naive"].max(),
 114:             }
 115:         )
 116: 
 117:         wins, total, rate = summarize_boolean(g, "v5_beats_quality_only")
 118:         rows.append(
 119:             {
 120:                 "check": f"V5 > V4 quality-only by eval_mode={eval_mode}",
 121:                 "wins": wins,
 122:                 "total_protocols": total,
 123:                 "win_rate": rate,
 124:                 "mean_delta": g["delta_v5_minus_v4_quality"].mean(),
 125:                 "median_delta": g["delta_v5_minus_v4_quality"].median(),
 126:                 "min_delta": g["delta_v5_minus_v4_quality"].min(),
 127:                 "max_delta": g["delta_v5_minus_v4_quality"].max(),
 128:             }
 129:         )
 130: 
 131:     return pd.DataFrame(rows)
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 164–250 行**

```python
 164: def write_report(delta: pd.DataFrame, summary: pd.DataFrame, failures: pd.DataFrame) -> None:
 165:     lines = []
 166:     lines += [
 167:         "# Stage 16-A2 QCR-U Robustness Check",
 168:         "",
 169:         "## 1. Purpose",
 170:         "",
 171:         "Stage 16-A1 showed that fixed quality-consistency fusion can improve the best protocol.",
 172:         "",
 173:         "Stage 16-A2 checks whether that gain is robust across all protocols, instead of only appearing in the best protocol.",
 174:         "",
 175:         "## 2. Overall Robustness Summary",
 176:         "",
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
 189:         "",
 190:         "## 3. Protocol-level Deltas",
 191:         "",
 192:         "| Backbone | Strategy | Eval Mode | V5 AUROC | V3 AUROC | V4 AUROC | V5-V3 | V5-V4 |",
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
 205:         "",
 206:         "## 4. Failure / Weakness Cases",
 207:         "",
 208:     ]
 209: 
 210:     if failures.empty:
 211:         lines.append("No failure case found under the current checks.")
 212:     else:
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
 225:             )
 226: 
 227:     lines += [
 228:         "",
 229:         "## 5. Decision Rule",
 230:         "",
 231:         "If V5 is consistently better than V3 naive fusion but often worse than V4 quality-only, the consistency term should not be claimed as universally beneficial.",
 232:         "",
 233:         "In that case, the next method should be revised from fixed Q+C fusion to adaptive QCR-U:",
 234:         "",
 235:         "```text",
 236:         "use quality-weighted crop as the stable core;",
 237:         "apply consistency only when detector and VLM evidence are both reliable;",
 238:         "avoid adding consistency under weak/full-image protocols where it hurts.",
 239:         "```",
 240:         "",
 241:         "## 6. Outputs",
 242:         "",
 243:         f"- `{OUT_DELTA.relative_to(ROOT)}`",
... 函数剩余 7 行已省略 ...
```

**函数 `build_base_table`：`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 166–241 行**

```python
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
 177:         "image_key",
 178:         "is_anomaly_final",
 179:         "fallback",
 180:         "has_candidate",
 181:         "num_candidates",
 182:         "vlm_score_norm",
 183:         "detector_score_norm",
 184:         "candidate_quality_norm",
 185:         "high_high_consistency",
 186:     ]
 187:     base_cols = [c for c in base_cols if c in df.columns]
 188: 
 189:     base = df[base_cols].copy()
 190:     base = base.drop_duplicates(
 191:         subset=["backbone", "dataset", "category", "strategy", "eval_mode", "image_key"]
 192:     ).reset_index(drop=True)
 193: 
 194:     for c in [
 195:         "is_anomaly_final",
 196:         "vlm_score_norm",
 197:         "detector_score_norm",
 198:         "candidate_quality_norm",
 199:         "high_high_consistency",
 200:         "num_candidates",
 201:     ]:
 202:         if c in base.columns:
 203:             base[c] = pd.to_numeric(base[c], errors="coerce")
 204: 
 205:     base["D"] = base["detector_score_norm"].fillna(0.0)
 206:     base["M"] = base["vlm_score_norm"].fillna(0.0)
 207:     base["Q"] = base["candidate_quality_norm"].fillna(0.0)
 208:     base["K"] = base["high_high_consistency"].fillna(0.0)
 209: 
 210:     base["score_detector_only"] = base["D"]
 211:     base["score_crop_vlm"] = base["M"]
 212:     base["score_naive"] = 0.5 * base["D"] + 0.5 * base["M"]
 213: 
 214:     base["score_quality_raw"] = (
 215:         0.5 * base["D"]
 216:         + 0.5 * (base["M"] * (0.5 + 0.5 * base["Q"]))
 217:     )
 218: 
 219:     base["score_fixed_qc_raw"] = (
 220:         0.40 * base["D"]
 221:         + 0.40 * base["M"]
 222:         + 0.10 * base["Q"]
 223:         + 0.10 * base["K"]
 224:     )
 225: 
 226:     agreement = (1.0 - (base["D"] - base["M"]).abs()).clip(lower=0.0, upper=1.0)
 227:     mutual_anomaly_evidence = np.minimum(base["D"], base["M"])
 228:     adaptive_gate = base["Q"] * base["K"] * agreement * mutual_anomaly_evidence
 229: 
 230:     base["adaptive_gate"] = adaptive_gate
 231:     base["score_adaptive_qcru_raw"] = base["score_quality_raw"] + 0.05 * adaptive_gate
 232: 
 233:     group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
 234:     for raw_col, out_col in [
 235:         ("score_quality_raw", "score_quality"),
 236:         ("score_fixed_qc_raw", "score_fixed_qc"),
 237:         ("score_adaptive_qcru_raw", "score_adaptive_qcru"),
 238:     ]:
 239:         base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)
 240: 
 241:     return base
```

**函数 `build_decision`：`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 304–370 行**

```python
 304: def build_decision(primary: pd.DataFrame, per_config: pd.DataFrame) -> pd.DataFrame:
 305:     rows = []
 306: 
 307:     def get_delta(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
 308:         idx = ["backbone", "dataset", "strategy", "eval_mode"]
 309:         piv = df.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
 310:         piv.columns.name = None
 311:         if left not in piv.columns or right not in piv.columns:
 312:             return pd.DataFrame()
 313:         piv[f"delta_{left}_minus_{right}"] = piv[left] - piv[right]
 314:         return piv
 315: 
 316:     for scope, df in [("primary_protocol", primary), ("all_protocols", per_config)]:
 317:         for left, right, label in [
 318:             ("V6", "V3", "adaptive_qcru_minus_naive"),
 319:             ("V6", "V4", "adaptive_qcru_minus_quality"),
 320:             ("V6", "V5", "adaptive_qcru_minus_fixed_qc"),
 321:             ("V4", "V3", "quality_minus_naive"),
 322:         ]:
 323:             d = get_delta(df, left, right)
 324:             if d.empty:
 325:                 continue
 326:             delta_col = f"delta_{left}_minus_{right}"
 327:             rows.append(
 328:                 {
 329:                     "scope": scope,
 330:                     "comparison": label,
 331:                     "num_protocols": len(d),
 332:                     "wins": int((d[delta_col] > 0).sum()),
 333:                     "win_rate": float((d[delta_col] > 0).mean()),
 334:                     "mean_delta": float(d[delta_col].mean()),
 335:                     "median_delta": float(d[delta_col].median()),
 336:                     "min_delta": float(d[delta_col].min()),
 337:                     "max_delta": float(d[delta_col].max()),
 338:                 }
 339:             )
 340: 
 341:     decision = pd.DataFrame(rows)
 342: 
 343:     # Conservative final recommendation.
 344:     primary_v6_v4 = decision[
 345:         (decision["scope"] == "primary_protocol")
 346:         & (decision["comparison"] == "adaptive_qcru_minus_quality")
 347:     ]
 348: 
 349:     if not primary_v6_v4.empty:
 350:         mean_delta = float(primary_v6_v4.iloc[0]["mean_delta"])
 351:         if mean_delta >= 0.005:
 352:             recommendation = "Adaptive QCR-U can be presented as the final candidate method."
 353:             method_name = "Adaptive QCR-U"
 354:         elif mean_delta > 0:
 355:             recommendation = "Use Quality-Calibrated QCR as the main method; describe adaptive consistency as a small gated refinement."
 356:             method_name = "Quality-Calibrated QCR with adaptive consistency refinement"
 357:         else:
 358:             recommendation = "Do not use Adaptive QCR-U as final method; use quality-weighted fusion."
 359:             method_name = "Quality-Calibrated Localization-Guided Fusion"
 360:     else:
 361:         recommendation = "Insufficient primary comparison."
 362:         method_name = "undecided"
 363: 
 364:     decision["final_recommendation"] = ""
 365:     decision["recommended_method_name"] = ""
 366:     if len(decision) > 0:
 367:         decision.loc[0, "final_recommendation"] = recommendation
 368:         decision.loc[0, "recommended_method_name"] = method_name
 369: 
 370:     return decision
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 373–462 行**

```python
 373: def write_report(
 374:     per_config: pd.DataFrame,
 375:     per_category: pd.DataFrame,
 376:     primary: pd.DataFrame,
 377:     decision: pd.DataFrame,
 378: ) -> None:
 379:     lines = []
 380:     lines += [
 381:         "# Stage 16-B Adaptive QCR-U Paper-facing Comparison",
 382:         "",
 383:         "## 1. Purpose",
 384:         "",
 385:         "This stage connects the Adaptive QCR-U candidate back to a paper-facing comparison table.",
 386:         "",
 387:         "It tests whether Adaptive QCR-U should be the final method name, or whether the method should be downgraded to quality-calibrated localization-guided fusion.",
 388:         "",
 389:         "## 2. Primary Protocol",
 390:         "",
 391:         "The primary protocol is:",
 392:         "",
 393:         "```text",
 394:         "dataset = VisA",
 395:         "strategy = inspection_binary",
 396:         "eval_mode = crop_topk_ensemble",
 397:         "```",
 398:         "",
 399:         "Reason: QCR-U is a candidate/crop reliability method. `full_all` is useful for diagnostics but is not the correct primary protocol for a crop-based reliability module.",
 400:         "",
 401:         "## 3. Primary Protocol Table",
 402:         "",
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
 415:         "## 4. Decision Summary",
 416:         "",
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
 429:         rec = decision.iloc[0]["final_recommendation"]
 430:         name = decision.iloc[0]["recommended_method_name"]
 431:     else:
 432:         rec = "insufficient evidence"
 433:         name = "undecided"
 434: 
 435:     lines += [
 436:         "",
 437:         "## 5. Final Trial Recommendation",
 438:         "",
 439:         f"- recommended method name: `{name}`",
 440:         f"- recommendation: {rec}",
 441:         "",
 442:         "## 6. Interpretation Rule",
 443:         "",
 444:         "If Adaptive QCR-U only improves over quality-only by a negligible margin, the paper should not overclaim adaptive consistency.",
 445:         "",
 446:         "In that case, the correct claim is:",
 447:         "",
 448:         "```text",
 449:         "Candidate quality provides the main reliability calibration gain, while adaptive consistency is a conservative refinement that avoids fixed-consistency degradation.",
 450:         "```",
 451:         "",
 452:         "## 7. Outputs",
... 函数剩余 10 行已省略 ...
```

**函数 `main`：`experiments/stage16_qcru_ablation/build_stage16_c_final_method_claims.py`，第 98–355 行**

```python
  98: def main() -> None:
  99:     OUT_DIR.mkdir(parents=True, exist_ok=True)
 100:     DOC_DIR.mkdir(parents=True, exist_ok=True)
 101: 
 102:     primary = read_csv_strict(IN_PRIMARY)
 103:     decision = read_csv_strict(IN_DECISION)
 104: 
 105:     # Primary-protocol deltas.
 106:     d_v4_v3 = get_primary_delta(primary, "V4", "V3")
 107:     d_v6_v3 = get_primary_delta(primary, "V6", "V3")
 108:     d_v6_v4 = get_primary_delta(primary, "V6", "V4")
 109:     d_v5_v4 = get_primary_delta(primary, "V5", "V4")
 110:     d_v6_v5 = get_primary_delta(primary, "V6", "V5")
 111: 
 112:     # All-protocol summaries from Stage 16-B decision file.
 113:     all_v4_v3 = lookup_decision(decision, "all_protocols", "quality_minus_naive")
 114:     all_v6_v3 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_naive")
 115:     all_v6_v4 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_quality")
 116:     all_v6_v5 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_fixed_qc")
 117: 
 118:     # Recommendation from Stage 16-B.
 119:     recommended_name = ""
 120:     final_recommendation = ""
 121:     if "recommended_method_name" in decision.columns:
 122:         vals = [v for v in decision["recommended_method_name"].dropna().astype(str).tolist() if v.strip()]
 123:         if vals:
 124:             recommended_name = vals[0]
 125:     if "final_recommendation" in decision.columns:
 126:         vals = [v for v in decision["final_recommendation"].dropna().astype(str).tolist() if v.strip()]
 127:         if vals:
 128:             final_recommendation = vals[0]
 129: 
 130:     if not recommended_name:
 131:         if d_v6_v4["mean_delta"] >= 0.005:
 132:             recommended_name = "Adaptive QCR-U"
 133:         elif d_v6_v4["mean_delta"] > 0:
 134:             recommended_name = "Quality-Calibrated QCR with adaptive consistency refinement"
 135:         else:
 136:             recommended_name = "Quality-Calibrated Localization-Guided Fusion"
 137: 
 138:     if not final_recommendation:
 139:         final_recommendation = (
 140:             "Use the quality-calibrated method as the main paper-facing method; "
 141:             "treat adaptive consistency as a conservative refinement."
 142:         )
 143: 
 144:     rows = [
 145:         {
 146:             "claim_id": "C1",
 147:             "claim_type": "final_method_name",
 148:             "claim": "Use Quality-Calibrated QCR as the main paper-facing method family.",
 149:             "evidence": (
 150:                 f"Stage 16-B recommends `{recommended_name}`. "
 151:                 f"Primary adaptive-minus-quality mean delta is {fmt(d_v6_v4['mean_delta'])} AUROC."
 152:             ),
 153:             "paper_status": "use",
 154:         },
 155:         {
 156:             "claim_id": "C2",
 157:             "claim_type": "main_effective_component",
 158:             "claim": "Candidate quality calibration is the main effective component.",
 159:             "evidence": (
 160:                 f"Primary quality-minus-naive mean delta is {fmt(d_v4_v3['mean_delta'])} AUROC; "
 161:                 f"all-protocol quality-minus-naive mean delta is {fmt(all_v4_v3['mean_delta'])} AUROC."
 162:             ),
 163:             "paper_status": "use",
 164:         },
 165:         {
 166:             "claim_id": "C3",
 167:             "claim_type": "auxiliary_component",
 168:             "claim": "Adaptive consistency is a conservative refinement, not the main source of improvement.",
 169:             "evidence": (
 170:                 f"Primary adaptive-minus-quality mean delta is only {fmt(d_v6_v4['mean_delta'])} AUROC; "
 171:                 f"all-protocol adaptive-minus-quality mean delta is {fmt(all_v6_v4['mean_delta'])} AUROC."
 172:             ),
 173:             "paper_status": "use_with_caution",
 174:         },
 175:         {
 176:             "claim_id": "C4",
 177:             "claim_type": "rejected_claim",
... 函数剩余 178 行已省略 ...
```

**函数 `rename_qcr_variant`：`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 115–128 行**

```python
 115: def rename_qcr_variant(variant_id: str, variant: str) -> tuple[str, str, bool]:
 116:     if variant_id == "V0":
 117:         return "Detector only", "anchor_baseline", True
 118:     if variant_id == "V2":
 119:         return "Crop VLM only", "vlm_crop_baseline", True
 120:     if variant_id == "V3":
 121:         return "Naive detector-crop fusion", "naive_fusion_baseline", True
 122:     if variant_id == "V4":
 123:         return "Quality-Calibrated QCR", "main_effective_method_core", True
 124:     if variant_id == "V5":
 125:         return "Fixed Q+C fusion", "diagnostic_not_final", False
 126:     if variant_id == "V6":
 127:         return "Quality-Calibrated QCR + adaptive consistency refinement", "final_refinement_variant", True
 128:     return variant, "other", True
```

**函数 `build_claim_deltas`：`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 168–261 行**

```python
 168: def build_claim_deltas(system_table: pd.DataFrame, qcr_table: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
 169:     rows = []
 170: 
 171:     def system_score(method: str) -> float | None:
 172:         r = system_table[system_table["method"] == method]
 173:         if r.empty:
 174:             return None
 175:         return float(r.iloc[0]["mean_image_auroc"])
 176: 
 177:     loco = system_score("PatchCore + context VLM, LOCO")
 178:     same = system_score("PatchCore + context VLM, same-set")
 179:     patch = system_score("PatchCore")
 180:     ead = system_score("EfficientAD-30 fixed-budget")
 181:     winclip = system_score("WinCLIP fixed protocol")
 182:     context = system_score("context-aware VLM")
 183:     full = system_score("full-image VLM")
 184: 
 185:     system_pairs = [
 186:         ("LOCO fusion vs PatchCore", loco, patch, "system_level_main_delta"),
 187:         ("LOCO fusion vs EfficientAD-30 fixed-budget", loco, ead, "system_level_main_delta"),
 188:         ("LOCO fusion vs WinCLIP fixed protocol", loco, winclip, "system_level_main_delta"),
 189:         ("LOCO fusion vs context-aware VLM", loco, context, "system_level_main_delta"),
 190:         ("context-aware VLM vs full-image VLM", context, full, "vlm_localization_delta"),
 191:         ("same-set upper bound vs LOCO fair result", same, loco, "upper_bound_gap"),
 192:     ]
 193: 
 194:     for name, left, right, delta_type in system_pairs:
 195:         if left is None or right is None:
 196:             continue
 197:         rows.append(
 198:             {
 199:                 "delta_type": delta_type,
 200:                 "scope": "system_panel",
 201:                 "comparison": name,
 202:                 "left_score": left,
 203:                 "right_score": right,
 204:                 "delta": left - right,
 205:                 "paper_interpretation": interpret_system_delta(name),
 206:             }
 207:         )
 208: 
 209:     # QCR primary protocol mean over backbones.
 210:     piv = qcr_table.pivot_table(
 211:         index=["dataset", "strategy", "eval_mode", "backbone"],
 212:         columns="variant_id",
 213:         values="image_auroc",
 214:         aggfunc="first",
 215:     ).reset_index()
 216:     piv.columns.name = None
 217: 
 218:     qcr_pairs = [
 219:         ("Quality-Calibrated QCR vs naive fusion", "V4", "V3", "qcr_core_delta"),
 220:         ("Adaptive refinement vs Quality-Calibrated QCR", "V6", "V4", "adaptive_refinement_delta"),
 221:         ("Adaptive refinement vs naive fusion", "V6", "V3", "qcr_final_delta"),
 222:         ("Fixed Q+C vs Quality-Calibrated QCR", "V5", "V4", "diagnostic_fixed_consistency_delta"),
 223:         ("Adaptive refinement vs fixed Q+C", "V6", "V5", "robustness_tradeoff_delta"),
 224:     ]
 225: 
 226:     for name, left_col, right_col, delta_type in qcr_pairs:
 227:         if left_col not in piv.columns or right_col not in piv.columns:
 228:             continue
 229:         d = piv[left_col] - piv[right_col]
 230:         rows.append(
 231:             {
 232:                 "delta_type": delta_type,
 233:                 "scope": "qcr_primary_protocol",
 234:                 "comparison": name,
 235:                 "left_score": float(piv[left_col].mean()),
 236:                 "right_score": float(piv[right_col].mean()),
 237:                 "delta": float(d.mean()),
 238:                 "paper_interpretation": interpret_qcr_delta(name, float(d.mean())),
 239:             }
 240:         )
 241: 
 242:     # Include Stage 16-B decision rows as evidence, but not as final table entries.
 243:     if not decision.empty:
 244:         for _, r in decision.iterrows():
 245:             rows.append(
 246:                 {
 247:                     "delta_type": "stage16b_decision_summary",
... 函数剩余 14 行已省略 ...
```

**函数 `interpret_qcr_delta`：`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 278–291 行**

```python
 278: def interpret_qcr_delta(name: str, delta: float) -> str:
 279:     if "Quality-Calibrated QCR vs naive" in name:
 280:         return "Candidate quality calibration is the main method gain."
 281:     if "Adaptive refinement vs Quality" in name:
 282:         if abs(delta) < 0.005:
 283:             return "Adaptive consistency is only a small refinement, not a main contribution."
 284:         return "Adaptive consistency provides a meaningful refinement."
 285:     if "Adaptive refinement vs naive" in name:
 286:         return "Final refinement variant improves over naive fusion."
 287:     if "Fixed Q+C" in name:
 288:         return "Fixed consistency is diagnostic only because robustness is not stable across protocols."
 289:     if "Adaptive refinement vs fixed" in name:
 290:         return "Adaptive refinement trades peak primary-protocol AUROC for robustness."
 291:     return "QCR delta."
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 294–444 行**

```python
 294: def write_report(
 295:     system_table: pd.DataFrame,
 296:     qcr_table: pd.DataFrame,
 297:     deltas: pd.DataFrame,
 298:     claims: pd.DataFrame,
 299: ) -> None:
 300:     lines = []
 301:     lines += [
 302:         "# Stage 16-D Paper-facing Final Comparison",
 303:         "",
 304:         "## 1. Purpose",
 305:         "",
 306:         "This stage creates the final paper-facing comparison tables after the method claim was locked in Stage 16-C.",
 307:         "",
 308:         "The final method family is:",
 309:         "",
 310:         "```text",
 311:         "Quality-Calibrated QCR",
 312:         "```",
 313:         "",
 314:         "The adaptive consistency term is treated only as a conservative refinement, not as the main performance source.",
 315:         "",
 316:         "## 2. Important Comparison Rule",
 317:         "",
 318:         "This report uses two panels because Stage 15 system baselines and Stage 16 QCR ablations are not the same protocol.",
 319:         "",
 320:         "- Panel A compares system-level baselines from Stage 15.",
 321:         "- Panel B compares QCR variants under the Stage 16-B QCR primary protocol.",
 322:         "",
 323:         "Do not merge the two panels into a single global ranking.",
 324:         "",
 325:         "## 3. Panel A: System-level Strong Baseline Comparison",
 326:         "",
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
 339:         "Paper use:",
 340:         "",
 341:         "- Use `PatchCore + context VLM, LOCO` as the fair system-level result.",
 342:         "- Use `same-set` only as an upper-bound diagnostic.",
 343:         "- Keep `EfficientAD-30` explicitly labeled as fixed-budget.",
 344:         "",
 345:         "## 4. Panel B: QCR Primary-protocol Ablation",
 346:         "",
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
 359:         "",
 360:         "Paper use:",
 361:         "",
 362:         "- Treat `Quality-Calibrated QCR` as the main effective method core.",
 363:         "- Treat `Quality-Calibrated QCR + adaptive consistency refinement` as the final conservative refinement.",
 364:         "- Treat `Fixed Q+C fusion` as diagnostic only, because it is not robust across protocols.",
 365:         "",
 366:         "## 5. Claim-ready Deltas",
 367:         "",
 368:         "| Scope | Comparison | Left Score | Right Score | Delta | Interpretation |",
 369:         "|---|---|---:|---:|---:|---|",
 370:     ]
 371: 
 372:     for _, r in deltas.iterrows():
 373:         left = r["left_score"]
... 函数剩余 71 行已省略 ...
```

**函数 `build_base_table`：`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 130–204 行**

```python
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
 141:         "defect_type",
 142:     ]
 143: 
 144:     base_cols = REQUIRED_COLUMNS + [c for c in optional_cols if c in df.columns]
 145:     base = df[base_cols].copy()
 146: 
 147:     base = base.drop_duplicates(
 148:         subset=["backbone", "dataset", "category", "strategy", "eval_mode", "image_key"]
 149:     ).reset_index(drop=True)
 150: 
 151:     for c in [
 152:         "is_anomaly_final",
 153:         "vlm_score_norm",
 154:         "detector_score_norm",
 155:         "candidate_quality_norm",
 156:         "high_high_consistency",
 157:         "num_candidates",
 158:     ]:
 159:         if c in base.columns:
 160:             base[c] = pd.to_numeric(base[c], errors="coerce")
 161: 
 162:     base["D"] = base["detector_score_norm"].fillna(0.0)
 163:     base["M"] = base["vlm_score_norm"].fillna(0.0)
 164:     base["Q"] = base["candidate_quality_norm"].fillna(0.0)
 165:     base["K"] = base["high_high_consistency"].fillna(0.0)
 166: 
 167:     base["score_naive"] = 0.5 * base["D"] + 0.5 * base["M"]
 168: 
 169:     base["score_quality_raw"] = (
 170:         0.5 * base["D"]
 171:         + 0.5 * (base["M"] * (0.5 + 0.5 * base["Q"]))
 172:     )
 173: 
 174:     base["score_fixed_qc_raw"] = (
 175:         0.40 * base["D"]
 176:         + 0.40 * base["M"]
 177:         + 0.10 * base["Q"]
 178:         + 0.10 * base["K"]
 179:     )
 180: 
 181:     agreement = (1.0 - (base["D"] - base["M"]).abs()).clip(lower=0.0, upper=1.0)
 182:     mutual_anomaly_evidence = np.minimum(base["D"], base["M"])
 183:     adaptive_gate = base["Q"] * base["K"] * agreement * mutual_anomaly_evidence
 184: 
 185:     base["agreement"] = agreement
 186:     base["mutual_anomaly_evidence"] = mutual_anomaly_evidence
 187:     base["adaptive_gate"] = adaptive_gate
 188:     base["score_adaptive_raw"] = base["score_quality_raw"] + 0.05 * adaptive_gate
 189: 
 190:     group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
 191:     for raw_col, out_col in [
 192:         ("score_quality_raw", "score_quality"),
 193:         ("score_fixed_qc_raw", "score_fixed_qc"),
 194:         ("score_adaptive_raw", "score_adaptive"),
 195:     ]:
 196:         base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)
 197: 
 198:     base["delta_quality_minus_naive"] = base["score_quality"] - base["score_naive"]
 199:     base["delta_fixed_minus_quality"] = base["score_fixed_qc"] - base["score_quality"]
 200:     base["delta_adaptive_minus_quality"] = base["score_adaptive"] - base["score_quality"]
 201:     base["delta_adaptive_minus_fixed"] = base["score_adaptive"] - base["score_fixed_qc"]
 202:     base["detector_vlm_disagreement"] = (base["D"] - base["M"]).abs()
 203: 
 204:     return base
```

**函数 `build_category_summary`：`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 337–396 行**

```python
 337: def build_category_summary(primary: pd.DataFrame) -> pd.DataFrame:
 338:     variants = [
 339:         ("V3", "naive_detector_crop_fusion", "score_naive"),
 340:         ("V4", "Quality-Calibrated QCR", "score_quality"),
 341:         ("V5", "Fixed Q+C fusion", "score_fixed_qc"),
 342:         ("V6", "Quality-Calibrated QCR + adaptive consistency refinement", "score_adaptive"),
 343:     ]
 344: 
 345:     rows = []
 346:     group_cols = ["backbone", "dataset", "strategy", "eval_mode", "category"]
 347: 
 348:     for keys, g in primary.groupby(group_cols, dropna=False):
 349:         base_row = dict(zip(group_cols, keys))
 350: 
 351:         for variant_id, method, score_col in variants:
 352:             m = eval_binary(g["is_anomaly_final"], g[score_col])
 353:             row = base_row.copy()
 354:             row.update(
 355:                 {
 356:                     "variant_id": variant_id,
 357:                     "method": method,
 358:                     "score_col": score_col,
 359:                     **m,
 360:                 }
 361:             )
 362:             rows.append(row)
 363: 
 364:     long = pd.DataFrame(rows)
 365: 
 366:     idx = group_cols
 367:     piv = long.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
 368:     piv.columns.name = None
 369: 
 370:     for col in ["V3", "V4", "V5", "V6"]:
 371:         if col not in piv.columns:
 372:             piv[col] = np.nan
 373: 
 374:     piv["delta_v4_quality_minus_v3_naive"] = piv["V4"] - piv["V3"]
 375:     piv["delta_v6_adaptive_minus_v4_quality"] = piv["V6"] - piv["V4"]
 376:     piv["delta_v5_fixed_minus_v4_quality"] = piv["V5"] - piv["V4"]
 377:     piv["delta_v6_adaptive_minus_v5_fixed"] = piv["V6"] - piv["V5"]
 378: 
 379:     def boundary_label(r):
 380:         labels = []
 381:         if pd.notna(r["delta_v4_quality_minus_v3_naive"]) and r["delta_v4_quality_minus_v3_naive"] <= 0:
 382:             labels.append("quality_not_helpful")
 383:         if pd.notna(r["delta_v6_adaptive_minus_v4_quality"]) and abs(r["delta_v6_adaptive_minus_v4_quality"]) < 0.001:
 384:             labels.append("adaptive_gain_negligible")
 385:         if pd.notna(r["delta_v5_fixed_minus_v4_quality"]) and r["delta_v5_fixed_minus_v4_quality"] > 0:
 386:             labels.append("fixed_consistency_can_peak_but_diagnostic")
 387:         if pd.notna(r["V6"]) and r["V6"] < 0.90:
 388:             labels.append("low_absolute_qcr_auc")
 389:         return ";".join(labels) if labels else "no_major_boundary"
 390: 
 391:     piv["boundary_label"] = piv.apply(boundary_label, axis=1)
 392: 
 393:     return piv.sort_values(
 394:         ["backbone", "delta_v4_quality_minus_v3_naive", "delta_v6_adaptive_minus_v4_quality"],
 395:         ascending=[True, True, True],
 396:     ).reset_index(drop=True)
```

**函数 `build_decision_summary`：`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 399–459 行**

```python
 399: def build_decision_summary(category_summary: pd.DataFrame, case_inventory: pd.DataFrame) -> pd.DataFrame:
 400:     rows = []
 401: 
 402:     def add(decision_id, topic, decision, evidence, paper_action):
 403:         rows.append(
 404:             {
 405:                 "decision_id": decision_id,
 406:                 "topic": topic,
 407:                 "decision": decision,
 408:                 "evidence": evidence,
 409:                 "paper_action": paper_action,
 410:             }
 411:         )
 412: 
 413:     q_delta = category_summary["delta_v4_quality_minus_v3_naive"].dropna()
 414:     a_delta = category_summary["delta_v6_adaptive_minus_v4_quality"].dropna()
 415:     f_delta = category_summary["delta_v5_fixed_minus_v4_quality"].dropna()
 416: 
 417:     add(
 418:         "E1",
 419:         "quality_calibration",
 420:         "Keep candidate quality calibration as the main method core.",
 421:         f"Per-category mean V4-V3 AUROC delta={q_delta.mean():+.4f}; wins={(q_delta > 0).sum()}/{len(q_delta)}.",
 422:         "Use as main contribution.",
 423:     )
 424: 
 425:     add(
 426:         "E2",
 427:         "adaptive_consistency",
 428:         "Keep adaptive consistency only as a refinement.",
 429:         f"Per-category mean V6-V4 AUROC delta={a_delta.mean():+.4f}; wins={(a_delta > 0).sum()}/{len(a_delta)}.",
 430:         "Use with caution; do not call it the main source of improvement.",
 431:     )
 432: 
 433:     add(
 434:         "E3",
 435:         "fixed_consistency",
 436:         "Do not use fixed Q+C as the final method even if it peaks on some categories.",
 437:         f"Per-category mean V5-V4 AUROC delta={f_delta.mean():+.4f}; positive cases={(f_delta > 0).sum()}/{len(f_delta)}.",
 438:         "Mention as diagnostic only.",
 439:     )
 440: 
 441:     if not case_inventory.empty:
 442:         counts = case_inventory["case_type"].value_counts().to_dict()
 443:         add(
 444:             "E4",
 445:             "case_inventory",
 446:             "Use selected cases for qualitative boundary analysis.",
 447:             "; ".join([f"{k}={v}" for k, v in counts.items()]),
 448:             "Inspect representative cases manually before paper figures.",
 449:         )
 450: 
 451:     add(
 452:         "E5",
 453:         "paper_boundary",
 454:         "The method should be claimed as reliability calibration, not full anomaly understanding.",
 455:         "The case taxonomy explicitly includes detector-VLM disagreement and candidate-quality boundary cases.",
 456:         "Use boundary-aware wording in paper.",
 457:     )
 458: 
 459:     return pd.DataFrame(rows)
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 462–580 行**

```python
 462: def write_report(category_summary: pd.DataFrame, case_inventory: pd.DataFrame, decision: pd.DataFrame) -> None:
 463:     lines = []
 464:     lines += [
 465:         "# Stage 16-E Failure Cases and Boundary Analysis",
 466:         "",
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
 479:         "strategy = inspection_binary",
 480:         "eval_mode = crop_topk_ensemble",
 481:         "```",
 482:         "",
 483:         "## 3. Category-level Boundary Summary",
 484:         "",
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
 497:         )
 498: 
 499:     lines += [
 500:         "",
 501:         "## 4. Case Types Extracted",
 502:         "",
 503:         "| Case Type | Meaning | Paper Use |",
 504:         "|---|---|---|",
 505:         "| quality_helps_anomaly_boost | anomaly images whose score is boosted by quality calibration | positive qualitative example |",
 506:         "| quality_helps_normal_suppression | normal images suppressed by quality calibration | false-positive reduction example |",
 507:         "| quality_boundary_anomaly_suppression | anomaly images suppressed by quality calibration | boundary / failure case |",
 508:         "| quality_boundary_normal_boost | normal images boosted by quality calibration | boundary / failure case |",
 509:         "| fixed_consistency_boundary_anomaly_suppression | anomaly images where fixed consistency hurts | explains why fixed Q+C is not final |",
 510:         "| fixed_consistency_boundary_normal_boost | normal images where fixed consistency increases risk | explains false-positive boundary |",
 511:         "| adaptive_refinement_high_gate | images with strongest adaptive gate | explains refinement behavior |",
 512:         "| detector_vlm_disagreement_boundary | images with high detector/VLM disagreement | explains detector-VLM conflict |",
 513:         "",
 514:     ]
 515: 
 516:     if case_inventory.empty:
 517:         lines.append("No case inventory generated.")
 518:     else:
 519:         counts = case_inventory["case_type"].value_counts().reset_index()
 520:         counts.columns = ["case_type", "count"]
 521:         lines += [
 522:             "Case counts:",
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
... 函数剩余 39 行已省略 ...
```

**函数 `make_claim_map`：`experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py`，第 91–265 行**

```python
  91: def make_claim_map(system: pd.DataFrame, deltas: pd.DataFrame, boundary: pd.DataFrame, category: pd.DataFrame) -> pd.DataFrame:
  92:     loco = score_of(system, "PatchCore + context VLM, LOCO")
  93:     same = score_of(system, "PatchCore + context VLM, same-set")
  94:     patchcore = score_of(system, "PatchCore")
  95:     ead30 = score_of(system, "EfficientAD-30 fixed-budget")
  96:     winclip = score_of(system, "WinCLIP fixed protocol")
  97:     full_vlm = score_of(system, "full-image VLM")
  98:     context_vlm = score_of(system, "context-aware VLM")
  99: 
 100:     d_loco_patch = delta_row(deltas, "LOCO fusion vs PatchCore")
 101:     d_loco_ead = delta_row(deltas, "LOCO fusion vs EfficientAD-30")
 102:     d_loco_winclip = delta_row(deltas, "LOCO fusion vs WinCLIP")
 103:     d_context_full = delta_row(deltas, "context-aware VLM vs full-image VLM")
 104:     d_quality_naive = delta_row(deltas, "Quality-Calibrated QCR vs naive fusion")
 105:     d_adaptive_quality = delta_row(deltas, "Adaptive refinement vs Quality-Calibrated QCR")
 106:     d_adaptive_naive = delta_row(deltas, "Adaptive refinement vs naive fusion")
 107:     d_fixed_quality = delta_row(deltas, "Fixed Q+C vs Quality-Calibrated QCR")
 108: 
 109:     e_quality = boundary_decision(boundary, "E1")
 110:     e_adaptive = boundary_decision(boundary, "E2")
 111:     e_fixed = boundary_decision(boundary, "E3")
 112:     e_cases = boundary_decision(boundary, "E4")
 113:     e_boundary = boundary_decision(boundary, "E5")
 114: 
 115:     cat_stats = compute_category_stats(category)
 116: 
 117:     rows = [
 118:         {
 119:             "claim_id": "P1",
 120:             "claim_category": "problem_framing",
 121:             "paper_claim": "Industrial anomaly VLM reasoning should be localization-guided rather than full-image only.",
 122:             "allowed_wording": "We study localization-guided VLM anomaly recognition, where detector localization evidence is converted into candidate-level visual-language evidence.",
 123:             "forbidden_wording": "We solve full industrial anomaly understanding with a general-purpose VLM.",
 124:             "evidence_files": "stage16_d_paper_facing_system_baseline_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
 125:             "evidence_summary": (
 126:                 f"context-aware VLM AUROC={context_vlm}; full-image VLM AUROC={full_vlm}; "
 127:                 f"context minus full-image delta={fmt(d_context_full.get('delta', None))}."
 128:             ),
 129:             "support_level": "moderate",
 130:             "paper_section": "Introduction; Method motivation; Experiments",
 131:             "caveat": "Do not claim semantic understanding or manufacturing-cause reasoning.",
 132:             "status": "use",
 133:         },
 134:         {
 135:             "claim_id": "P2",
 136:             "claim_category": "system_level_result",
 137:             "paper_claim": "Localization-guided VLM evidence is complementary to detector baselines.",
 138:             "allowed_wording": "The fair LOCO fusion improves over the detector-only PatchCore baseline and the fixed-budget EfficientAD baseline.",
 139:             "forbidden_wording": "The method fully beats all detector baselines under all budgets.",
 140:             "evidence_files": "stage16_d_paper_facing_system_baseline_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
 141:             "evidence_summary": (
 142:                 f"LOCO AUROC={loco}; PatchCore AUROC={patchcore}; EfficientAD-30 AUROC={ead30}; "
 143:                 f"LOCO-PatchCore={fmt(d_loco_patch.get('delta', None))}; "
 144:                 f"LOCO-EfficientAD30={fmt(d_loco_ead.get('delta', None))}."
 145:             ),
 146:             "support_level": "strong_but_protocol_limited",
 147:             "paper_section": "Main Results",
 148:             "caveat": "EfficientAD is fixed-budget; same-set fusion is upper-bound only.",
 149:             "status": "use",
 150:         },
 151:         {
 152:             "claim_id": "P3",
 153:             "claim_category": "external_baseline",
 154:             "paper_claim": "The proposed localization-guided route is stronger than the fixed WinCLIP protocol used in this study.",
 155:             "allowed_wording": "Under our fixed protocol, LOCO fusion outperforms WinCLIP.",
 156:             "forbidden_wording": "We comprehensively outperform all CLIP-based anomaly detection methods.",
 157:             "evidence_files": "stage16_d_paper_facing_system_baseline_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
 158:             "evidence_summary": (
 159:                 f"LOCO AUROC={loco}; WinCLIP AUROC={winclip}; "
 160:                 f"delta={fmt(d_loco_winclip.get('delta', None))}."
 161:             ),
 162:             "support_level": "moderate",
 163:             "paper_section": "Baselines",
 164:             "caveat": "AnomalyCLIP is not yet included; avoid broad CLIP-family claims.",
 165:             "status": "use_with_caution",
 166:         },
 167:         {
 168:             "claim_id": "P4",
 169:             "claim_category": "main_method_component",
 170:             "paper_claim": "Candidate quality calibration is the main effective method component.",
... 函数剩余 95 行已省略 ...
```

**函数 `make_status_table`：`experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py`，第 268–310 行**

```python
 268: def make_status_table(claim_map: pd.DataFrame) -> pd.DataFrame:
 269:     rows = []
 270: 
 271:     groups = [
 272:         ("main_claims_ready", claim_map[claim_map["status"].isin(["use", "use_with_caution"])]),
 273:         ("claims_to_reject_or_downgrade", claim_map[claim_map["status"].isin(["reject", "reject_as_final_method", "use_as_diagnostic_only"])]),
 274:     ]
 275: 
 276:     for group_name, g in groups:
 277:         rows.append(
 278:             {
 279:                 "status_group": group_name,
 280:                 "num_claims": len(g),
 281:                 "claim_ids": ";".join(g["claim_id"].astype(str).tolist()),
 282:                 "summary": "; ".join(g["paper_claim"].astype(str).tolist()),
 283:             }
 284:         )
 285: 
 286:     # Paper readiness flags.
 287:     rows.extend(
 288:         [
 289:             {
 290:                 "status_group": "paper_ready_method_name",
 291:                 "num_claims": 1,
 292:                 "claim_ids": "P4;P5;P6",
 293:                 "summary": "Use Quality-Calibrated QCR as the method family; adaptive consistency is refinement; fixed Q+C is diagnostic only.",
 294:             },
 295:             {
 296:                 "status_group": "remaining_experiment_risks",
 297:                 "num_claims": 3,
 298:                 "claim_ids": "R1;R2;R3",
 299:                 "summary": "EfficientAD remains fixed-budget; AnomalyCLIP is absent; representative failure figures still need manual visual inspection.",
 300:             },
 301:             {
 302:                 "status_group": "next_actions",
 303:                 "num_claims": 2,
 304:                 "claim_ids": "N1;N2",
 305:                 "summary": "Run defensive EfficientAD-100 fruit_jelly sensitivity later; start paper outline/table-to-text drafting after claim map.",
 306:             },
 307:         ]
 308:     )
 309: 
 310:     return pd.DataFrame(rows)
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py`，第 333–460 行**

```python
 333: def write_report(claim_map: pd.DataFrame, status: pd.DataFrame, rejected: pd.DataFrame) -> None:
 334:     lines = []
 335:     lines += [
 336:         "# Stage 16-F Final Claim-Evidence Map",
 337:         "",
 338:         "## 1. Purpose",
 339:         "",
 340:         "This stage maps every paper-facing claim to concrete experimental evidence and locks the forbidden claims.",
 341:         "",
 342:         "No new model is trained and no score is tuned in this stage.",
 343:         "",
 344:         "## 2. Final Method Naming",
 345:         "",
 346:         "Use this method family name:",
 347:         "",
 348:         "```text",
 349:         "Quality-Calibrated QCR",
 350:         "```",
 351:         "",
 352:         "Use this longer descriptive phrase when needed:",
 353:         "",
 354:         "```text",
 355:         "Quality-Calibrated Localization-Guided VLM Reasoning",
 356:         "```",
 357:         "",
 358:         "Use this only as the full variant name:",
 359:         "",
 360:         "```text",
 361:         "Quality-Calibrated QCR with adaptive consistency refinement",
 362:         "```",
 363:         "",
 364:         "Do not write the method as fixed Q+C QCR-U.",
 365:         "",
 366:         "## 3. Claim-Evidence Map",
 367:         "",
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
 380:         "## 4. Evidence Details",
 381:         "",
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
... 函数剩余 48 行已省略 ...
```

**函数 `build_base_table`：`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 176–249 行**

```python
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
 187:         "strategy",
 188:         "eval_mode",
 189:         "image_key",
 190:         "is_anomaly_final",
 191:         "fallback",
 192:         "has_candidate",
 193:         "num_candidates",
 194:         "vlm_score_norm",
 195:         "detector_score_norm",
 196:         "candidate_quality_norm",
 197:         "high_high_consistency",
 198:     ]
 199:     base_cols = [c for c in base_cols if c in df.columns]
 200: 
 201:     base = df[base_cols].copy()
 202:     base = base.drop_duplicates(
 203:         subset=["backbone", "dataset", "category", "strategy", "eval_mode", "image_key"]
 204:     ).reset_index(drop=True)
 205: 
 206:     for c in [
 207:         "is_anomaly_final",
 208:         "vlm_score_norm",
 209:         "detector_score_norm",
 210:         "candidate_quality_norm",
 211:         "high_high_consistency",
 212:         "num_candidates",
 213:     ]:
 214:         if c in base.columns:
 215:             base[c] = pd.to_numeric(base[c], errors="coerce")
 216: 
 217:     base["M_crop_vlm"] = base["vlm_score_norm"]
 218:     base["D_detector"] = base["detector_score_norm"]
 219:     base["Q_quality"] = base["candidate_quality_norm"].fillna(0.0)
 220:     base["K_consistency"] = base["high_high_consistency"].fillna(0.0)
 221: 
 222:     # Fixed, non-tuned ablation formulas.
 223:     base["score_detector_only"] = base["D_detector"]
 224:     base["score_crop_topk_vlm"] = base["M_crop_vlm"]
 225:     base["score_naive_detector_crop_fusion"] = 0.5 * base["D_detector"] + 0.5 * base["M_crop_vlm"]
 226: 
 227:     # Quality should modulate whether the crop VLM signal is trusted, not replace the detector score.
 228:     base["score_quality_weighted_crop_raw"] = (
 229:         0.5 * base["D_detector"]
 230:         + 0.5 * (base["M_crop_vlm"] * (0.5 + 0.5 * base["Q_quality"]))
 231:     )
 232: 
 233:     # Consistency gets a small fixed weight. This is not tuned on test labels.
 234:     base["score_quality_consistency_fusion_raw"] = (
 235:         0.40 * base["D_detector"]
 236:         + 0.40 * base["M_crop_vlm"]
 237:         + 0.10 * base["Q_quality"]
 238:         + 0.10 * base["K_consistency"]
 239:     )
 240: 
 241:     # Normalize raw variants within each protocol group so scores are comparable for threshold metrics.
 242:     group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
 243:     for raw_col, out_col in [
 244:         ("score_quality_weighted_crop_raw", "score_quality_weighted_crop"),
 245:         ("score_quality_consistency_fusion_raw", "score_quality_consistency_fusion"),
 246:     ]:
 247:         base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)
 248: 
 249:     return base
```

**函数 `make_variant_long`：`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 252–283 行**

```python
 252: def make_variant_long(base: pd.DataFrame) -> pd.DataFrame:
 253:     variants = [
 254:         ("V0", "detector_only", "score_detector_only", False, False),
 255:         ("V2", "crop_topk_vlm", "score_crop_topk_vlm", False, False),
 256:         ("V3", "naive_detector_crop_fusion", "score_naive_detector_crop_fusion", False, False),
 257:         ("V4", "quality_weighted_crop", "score_quality_weighted_crop", True, False),
 258:         ("V5", "quality_consistency_fusion", "score_quality_consistency_fusion", True, True),
 259:     ]
 260: 
 261:     rows = []
 262:     id_cols = [
 263:         "backbone",
 264:         "dataset",
 265:         "category",
 266:         "strategy",
 267:         "eval_mode",
 268:         "image_key",
 269:         "is_anomaly_final",
 270:     ]
 271:     optional_cols = ["has_candidate", "num_candidates"]
 272:     id_cols += [c for c in optional_cols if c in base.columns]
 273: 
 274:     for variant_id, variant, score_col, uses_q, uses_k in variants:
 275:         tmp = base[id_cols].copy()
 276:         tmp["variant_id"] = variant_id
 277:         tmp["variant"] = variant
 278:         tmp["score"] = base[score_col].astype(float)
 279:         tmp["uses_quality"] = uses_q
 280:         tmp["uses_consistency"] = uses_k
 281:         rows.append(tmp)
 282: 
 283:     return pd.concat(rows, ignore_index=True)
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 326–449 行**

```python
 326: def write_report(
 327:     base: pd.DataFrame,
 328:     per_config: pd.DataFrame,
 329:     per_category: pd.DataFrame,
 330:     best_protocol: pd.DataFrame,
 331: ) -> None:
 332:     lines = []
 333:     lines += [
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
 346:         "",
 347:         "The base table contains detector score, crop VLM score, candidate quality, and detector-VLM consistency.",
 348:         "",
 349:         "## 3. Fixed Ablation Variants",
 350:         "",
 351:         "| Variant | Formula | Meaning |",
 352:         "|---|---|---|",
 353:         "| detector_only | `D` | detector score only |",
 354:         "| crop_topk_vlm | `M` | crop VLM score only |",
 355:         "| naive_detector_crop_fusion | `0.5D + 0.5M` | naive fusion baseline |",
 356:         "| quality_weighted_crop | `0.5D + 0.5(M * (0.5 + 0.5Q))` | candidate quality modulates VLM evidence |",
 357:         "| quality_consistency_fusion | `0.4D + 0.4M + 0.1Q + 0.1K` | fixed Q+C fusion variant |",
 358:         "",
 359:         "Where `D` is detector score, `M` is crop VLM abnormal score, `Q` is candidate quality, and `K` is detector-VLM high-high consistency.",
 360:         "",
 361:         "## 4. Best Protocols by Q+C Fusion AUROC",
 362:         "",
 363:         "| Rank | Backbone | Dataset | Strategy | Eval Mode | V5 AUROC | V5 AP | V5 Best F1 |",
 364:         "|---:|---|---|---|---|---:|---:|---:|",
 365:     ]
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
 378:         "",
 379:         "## 5. Variant Comparison Within the Best Protocol",
 380:         "",
 381:     ]
 382: 
 383:     if not best_protocol.empty:
 384:         best = best_protocol.iloc[0]
 385:         mask = (
 386:             (per_config["backbone"] == best["backbone"])
 387:             & (per_config["dataset"] == best["dataset"])
 388:             & (per_config["strategy"] == best["strategy"])
 389:             & (per_config["eval_mode"] == best["eval_mode"])
 390:         )
 391:         comp = per_config[mask].sort_values("variant_id")
 392: 
 393:         lines += [
 394:             f"Best protocol by V5 AUROC: `{best['backbone']} / {best['dataset']} / {best['strategy']} / {best['eval_mode']}`.",
 395:             "",
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
... 函数剩余 44 行已省略 ...
```

**函数 `build_base_table`：`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 155–231 行**

```python
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
 166:         "image_key",
 167:         "is_anomaly_final",
 168:         "fallback",
 169:         "has_candidate",
 170:         "num_candidates",
 171:         "vlm_score_norm",
 172:         "detector_score_norm",
 173:         "candidate_quality_norm",
 174:         "high_high_consistency",
 175:     ]
 176:     base_cols = [c for c in base_cols if c in df.columns]
 177: 
 178:     base = df[base_cols].copy()
 179:     base = base.drop_duplicates(
 180:         subset=["backbone", "dataset", "category", "strategy", "eval_mode", "image_key"]
 181:     ).reset_index(drop=True)
 182: 
 183:     for c in [
 184:         "is_anomaly_final",
 185:         "vlm_score_norm",
 186:         "detector_score_norm",
 187:         "candidate_quality_norm",
 188:         "high_high_consistency",
 189:         "num_candidates",
 190:     ]:
 191:         if c in base.columns:
 192:             base[c] = pd.to_numeric(base[c], errors="coerce")
 193: 
 194:     base["D"] = base["detector_score_norm"].fillna(0.0)
 195:     base["M"] = base["vlm_score_norm"].fillna(0.0)
 196:     base["Q"] = base["candidate_quality_norm"].fillna(0.0)
 197:     base["K"] = base["high_high_consistency"].fillna(0.0)
 198: 
 199:     # Existing baselines.
 200:     base["score_naive"] = 0.5 * base["D"] + 0.5 * base["M"]
 201:     base["score_quality_raw"] = 0.5 * base["D"] + 0.5 * (base["M"] * (0.5 + 0.5 * base["Q"]))
 202:     base["score_fixed_qc_raw"] = 0.40 * base["D"] + 0.40 * base["M"] + 0.10 * base["Q"] + 0.10 * base["K"]
 203: 
 204:     # Adaptive QCR-U:
 205:     # Start from quality-weighted core.
 206:     # Add a conservative consistency bonus only when:
 207:     # - candidate quality is high,
 208:     # - detector and VLM agree,
 209:     # - both detector and VLM provide high anomaly evidence.
 210:     #
 211:     # This is label-free and intentionally conservative.
 212:     agreement = 1.0 - (base["D"] - base["M"]).abs()
 213:     agreement = agreement.clip(lower=0.0, upper=1.0)
 214: 
 215:     mutual_anomaly_evidence = np.minimum(base["D"], base["M"])
 216:     adaptive_gate = base["Q"] * base["K"] * agreement * mutual_anomaly_evidence
 217: 
 218:     base["adaptive_gate"] = adaptive_gate
 219: 
 220:     # Small fixed coefficient. This is not selected from test labels.
 221:     base["score_adaptive_qcru_raw"] = base["score_quality_raw"] + 0.05 * adaptive_gate
 222: 
 223:     group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
 224:     for raw_col, out_col in [
 225:         ("score_quality_raw", "score_quality"),
 226:         ("score_fixed_qc_raw", "score_fixed_qc"),
 227:         ("score_adaptive_qcru_raw", "score_adaptive_qcru"),
 228:     ]:
 229:         base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)
 230: 
 231:     return base
```

**函数 `make_long`：`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 234–260 行**

```python
 234: def make_long(base: pd.DataFrame) -> pd.DataFrame:
 235:     variants = [
 236:         ("V3", "naive_detector_crop_fusion", "score_naive"),
 237:         ("V4", "quality_weighted_crop", "score_quality"),
 238:         ("V5", "fixed_quality_consistency", "score_fixed_qc"),
 239:         ("V6", "adaptive_qcru", "score_adaptive_qcru"),
 240:     ]
 241: 
 242:     id_cols = [
 243:         "backbone",
 244:         "dataset",
 245:         "category",
 246:         "strategy",
 247:         "eval_mode",
 248:         "image_key",
 249:         "is_anomaly_final",
 250:     ]
 251: 
 252:     rows = []
 253:     for vid, variant, col in variants:
 254:         tmp = base[id_cols].copy()
 255:         tmp["variant_id"] = vid
 256:         tmp["variant"] = variant
 257:         tmp["score"] = base[col].astype(float)
 258:         rows.append(tmp)
 259: 
 260:     return pd.concat(rows, ignore_index=True)
```

**函数 `summarize_delta`：`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 298–348 行**

```python
 298: def summarize_delta(delta: pd.DataFrame) -> pd.DataFrame:
 299:     checks = [
 300:         ("V6 > V3 naive", "v6_beats_naive", "delta_v6_minus_v3_naive"),
 301:         ("V6 > V4 quality", "v6_beats_quality", "delta_v6_minus_v4_quality"),
 302:         ("V6 > V5 fixed Q+C", "v6_beats_fixed_qc", "delta_v6_minus_v5_fixed_qc"),
 303:         ("V5 > V4 quality", "v5_beats_quality", "delta_v5_minus_v4_quality"),
 304:     ]
 305: 
 306:     rows = []
 307:     for name, win_col, delta_col in checks:
 308:         rows.append(
 309:             {
 310:                 "check": name,
 311:                 "wins": int(delta[win_col].sum()),
 312:                 "total_protocols": int(len(delta)),
 313:                 "win_rate": float(delta[win_col].mean()),
 314:                 "mean_delta": float(delta[delta_col].mean()),
 315:                 "median_delta": float(delta[delta_col].median()),
 316:                 "min_delta": float(delta[delta_col].min()),
 317:                 "max_delta": float(delta[delta_col].max()),
 318:             }
 319:         )
 320: 
 321:     for eval_mode, g in delta.groupby("eval_mode"):
 322:         rows.append(
 323:             {
 324:                 "check": f"V6 > V4 quality by eval_mode={eval_mode}",
 325:                 "wins": int(g["v6_beats_quality"].sum()),
 326:                 "total_protocols": int(len(g)),
 327:                 "win_rate": float(g["v6_beats_quality"].mean()),
 328:                 "mean_delta": float(g["delta_v6_minus_v4_quality"].mean()),
 329:                 "median_delta": float(g["delta_v6_minus_v4_quality"].median()),
 330:                 "min_delta": float(g["delta_v6_minus_v4_quality"].min()),
 331:                 "max_delta": float(g["delta_v6_minus_v4_quality"].max()),
 332:             }
 333:         )
 334: 
 335:         rows.append(
 336:             {
 337:                 "check": f"V6 > V5 fixed Q+C by eval_mode={eval_mode}",
 338:                 "wins": int(g["v6_beats_fixed_qc"].sum()),
 339:                 "total_protocols": int(len(g)),
 340:                 "win_rate": float(g["v6_beats_fixed_qc"].mean()),
 341:                 "mean_delta": float(g["delta_v6_minus_v5_fixed_qc"].mean()),
 342:                 "median_delta": float(g["delta_v6_minus_v5_fixed_qc"].median()),
 343:                 "min_delta": float(g["delta_v6_minus_v5_fixed_qc"].min()),
 344:                 "max_delta": float(g["delta_v6_minus_v5_fixed_qc"].max()),
 345:             }
 346:         )
 347: 
 348:     return pd.DataFrame(rows)
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 351–444 行**

```python
 351: def write_report(per_config: pd.DataFrame, delta: pd.DataFrame, summary: pd.DataFrame) -> None:
 352:     best = per_config[per_config["variant_id"] == "V6"].sort_values("auroc", ascending=False).reset_index(drop=True)
 353:     best["rank_by_v6_auroc"] = range(1, len(best) + 1)
 354: 
 355:     lines = []
 356:     lines += [
 357:         "# Stage 16-A3 Adaptive QCR-U",
 358:         "",
 359:         "## 1. Purpose",
 360:         "",
 361:         "Stage 16-A2 showed that candidate quality is stable, while fixed consistency is not universally beneficial.",
 362:         "",
 363:         "This stage tests an adaptive QCR-U score that uses quality-weighted crop scoring as the stable core and applies consistency only as a conservative reliability-gated bonus.",
 364:         "",
 365:         "## 2. Formula",
 366:         "",
 367:         "```text",
 368:         "D = detector anomaly score",
 369:         "M = crop VLM anomaly score",
 370:         "Q = candidate quality",
 371:         "K = high-high detector/VLM consistency",
 372:         "",
 373:         "S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)",
 374:         "agreement = 1 - |D - M|",
 375:         "mutual_anomaly_evidence = min(D, M)",
 376:         "gate = Q * K * agreement * mutual_anomaly_evidence",
 377:         "S_adaptive = S_quality + 0.05 * gate",
 378:         "```",
 379:         "",
 380:         "The coefficient `0.05` is fixed and intentionally conservative. It is not selected by test-set tuning.",
 381:         "",
 382:         "## 3. Robustness Summary",
 383:         "",
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
 396:         "",
 397:         "## 4. Adaptive QCR-U Protocol Ranking",
 398:         "",
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
 411:         "## 5. Protocol-level Delta Table",
 412:         "",
 413:         "| Backbone | Strategy | Eval Mode | V3 | V4 | V5 | V6 | V6-V3 | V6-V4 | V6-V5 |",
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
 426: 
 427:     lines += [
 428:         "",
 429:         "## 6. Decision Rule",
 430:         "",
... 函数剩余 14 行已省略 ...
```

**函数 `main`：`experiments/stage18_ad2_qcr_ablation/build_stage18_a0_ad2_qcr_inventory.py`，第 35–172 行**

```python
  35: def main() -> None:
  36:     OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
  37:     OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
  38: 
  39:     rows = []
  40: 
  41:     for path in CANDIDATE_FILES:
  42:         df = read_csv_safe(path)
  43: 
  44:         if df is None:
  45:             rows.append(
  46:                 {
  47:                     "file": str(path.relative_to(ROOT)),
  48:                     "exists_and_readable": False,
  49:                     "num_rows": None,
  50:                     "num_cols": None,
  51:                     "has_dataset_col": False,
  52:                     "datasets": "",
  53:                     "has_category_col": False,
  54:                     "categories_found": "",
  55:                     "ad2_categories_found": "",
  56:                     "ad2_coverage_count": 0,
  57:                     "can_directly_run_ad2_qcr_ablation": False,
  58:                     "notes": "missing or unreadable",
  59:                 }
  60:             )
  61:             continue
  62: 
  63:         cols = set(df.columns)
  64:         has_dataset = "dataset" in cols
  65:         has_category = "category" in cols
  66: 
  67:         datasets = ""
  68:         if has_dataset:
  69:             datasets = ";".join(sorted(map(str, df["dataset"].dropna().unique())))
  70: 
  71:         categories_found = ""
  72:         ad2_found = []
  73: 
  74:         if has_category:
  75:             cats = sorted(map(str, df["category"].dropna().unique()))
  76:             categories_found = ";".join(cats)
  77:             ad2_found = [c for c in AD2_CATEGORIES if c in cats]
  78: 
  79:         required_score_cols = [
  80:             "vlm_score_norm",
  81:             "detector_score_norm",
  82:             "candidate_quality_norm",
  83:         ]
  84: 
  85:         has_required_score_cols = all(c in cols for c in required_score_cols)
  86: 
  87:         can_direct = (
  88:             has_category
  89:             and len(ad2_found) == len(AD2_CATEGORIES)
  90:             and has_required_score_cols
  91:         )
  92: 
  93:         rows.append(
  94:             {
  95:                 "file": str(path.relative_to(ROOT)),
  96:                 "exists_and_readable": True,
  97:                 "num_rows": len(df),
  98:                 "num_cols": len(df.columns),
  99:                 "has_dataset_col": has_dataset,
 100:                 "datasets": datasets,
 101:                 "has_category_col": has_category,
 102:                 "categories_found": categories_found,
 103:                 "ad2_categories_found": ";".join(ad2_found),
 104:                 "ad2_coverage_count": len(ad2_found),
 105:                 "can_directly_run_ad2_qcr_ablation": can_direct,
 106:                 "notes": (
 107:                     "contains AD2 QCR-ready predictions"
 108:                     if can_direct
 109:                     else "does not contain full AD2 QCR-ready prediction columns/categories"
 110:                 ),
 111:             }
 112:         )
 113: 
 114:     out = pd.DataFrame(rows)
... 函数剩余 58 行已省略 ...
```

**函数 `scan_file`：`experiments/stage18_ad2_qcr_ablation/build_stage18_b0_ad2_qcr_source_inventory.py`，第 170–270 行**

```python
 170: def scan_file(path: Path) -> dict:
 171:     df = read_table_sample(path)
 172: 
 173:     base = {
 174:         "file": str(path.relative_to(ROOT)),
 175:         "suffix": path.suffix.lower(),
 176:         "readable": False,
 177:         "num_sample_rows": None,
 178:         "num_cols": None,
 179:         "category_col": "",
 180:         "ad2_categories_found": "",
 181:         "ad2_coverage_count": 0,
 182:         "image_id_col": "",
 183:         "label_col": "",
 184:         "detector_score_col": "",
 185:         "vlm_score_col": "",
 186:         "quality_col": "",
 187:         "consistency_col": "",
 188:         "has_image_id": False,
 189:         "has_label": False,
 190:         "has_detector_score": False,
 191:         "has_vlm_score": False,
 192:         "has_quality": False,
 193:         "has_consistency": False,
 194:         "source_role": "unreadable_or_irrelevant",
 195:         "qcr_assembly_value": "none",
 196:         "notes": "",
 197:     }
 198: 
 199:     if df is None or df.empty or len(df.columns) <= 1:
 200:         return base
 201: 
 202:     cols = set(df.columns)
 203: 
 204:     category_col, ad2_found = detect_categories(df, path)
 205: 
 206:     image_id_col = find_first_existing(cols, IMAGE_ID_COLS)
 207:     label_col = find_first_existing(cols, LABEL_COLS)
 208:     detector_col = find_first_existing(cols, DETECTOR_SCORE_COLS)
 209:     vlm_col = find_first_existing(cols, VLM_SCORE_COLS)
 210:     quality_col = find_first_existing(cols, QUALITY_COLS)
 211:     consistency_col = find_first_existing(cols, CONSISTENCY_COLS)
 212: 
 213:     has_image = bool(image_id_col)
 214:     has_label = bool(label_col)
 215:     has_detector = bool(detector_col)
 216:     has_vlm = bool(vlm_col)
 217:     has_quality = bool(quality_col)
 218:     has_consistency = bool(consistency_col)
 219: 
 220:     role_parts = []
 221:     if has_detector:
 222:         role_parts.append("detector")
 223:     if has_vlm:
 224:         role_parts.append("vlm")
 225:     if has_quality:
 226:         role_parts.append("quality")
 227:     if has_consistency:
 228:         role_parts.append("consistency")
 229:     if has_label:
 230:         role_parts.append("label")
 231: 
 232:     if len(ad2_found) == 0:
 233:         source_role = "non_ad2_or_summary"
 234:         qcr_value = "none"
 235:     elif has_image and has_label and has_detector and has_vlm and has_quality:
 236:         source_role = "ad2_qcr_ready_or_near_ready"
 237:         qcr_value = "high"
 238:     elif has_image and has_label and (has_detector or has_vlm or has_quality):
 239:         source_role = "ad2_partial_per_image_source"
 240:         qcr_value = "medium"
 241:     elif len(ad2_found) > 0:
 242:         source_role = "ad2_summary_or_category_level_source"
 243:         qcr_value = "low"
 244:     else:
 245:         source_role = "unclassified"
 246: 
 247:     return {
 248:         **base,
 249:         "readable": True,
... 函数剩余 21 行已省略 ...
```

**函数 `main`：`experiments/stage18_ad2_qcr_ablation/build_stage18_b0_ad2_qcr_source_inventory.py`，第 299–400 行**

```python
 299: def main() -> None:
 300:     OUT_DIR.mkdir(parents=True, exist_ok=True)
 301:     DOC_DIR.mkdir(parents=True, exist_ok=True)
 302: 
 303:     files = iter_candidate_files()
 304: 
 305:     rows = []
 306:     for i, p in enumerate(files, start=1):
 307:         if i % 100 == 0:
 308:             print(f"[SCAN] {i}/{len(files)} {p}")
 309:         rows.append(scan_file(p))
 310: 
 311:     inv = pd.DataFrame(rows)
 312:     inv.to_csv(OUT_INVENTORY, index=False, lineterminator="\n")
 313: 
 314:     if inv.empty:
 315:         candidates = inv
 316:     else:
 317:         candidates = inv[
 318:             (inv["ad2_coverage_count"] > 0)
 319:             & (inv["qcr_assembly_value"].isin(["high", "medium", "low"]))
 320:         ].copy()
 321: 
 322:         order = {"high": 0, "medium": 1, "low": 2, "none": 3}
 323:         candidates["_order"] = candidates["qcr_assembly_value"].map(order).fillna(9)
 324:         candidates = candidates.sort_values(
 325:             ["_order", "ad2_coverage_count", "file"],
 326:             ascending=[True, False, True],
 327:         ).drop(columns=["_order"])
 328: 
 329:     candidates.to_csv(OUT_CANDIDATES, index=False, lineterminator="\n")
 330: 
 331:     high = int((candidates["qcr_assembly_value"] == "high").sum()) if not candidates.empty else 0
 332:     medium = int((candidates["qcr_assembly_value"] == "medium").sum()) if not candidates.empty else 0
 333:     low = int((candidates["qcr_assembly_value"] == "low").sum()) if not candidates.empty else 0
 334: 
 335:     lines = [
 336:         "# Stage 18-B0 AD2 QCR Source Inventory",
 337:         "",
 338:         "## Purpose",
 339:         "",
 340:         "Scan existing result/run files to determine whether AD2 four-category QCR predictions can be assembled from existing per-image sources.",
 341:         "",
 342:         "## Summary",
 343:         "",
 344:         f"- scanned files: `{len(files)}`",
 345:         f"- AD2 high-value QCR-ready/near-ready files: `{high}`",
 346:         f"- AD2 medium-value partial per-image files: `{medium}`",
 347:         f"- AD2 low-value summary/category-level files: `{low}`",
 348:         "",
 349:         "## Candidate files",
 350:         "",
 351:         "| File | Coverage | Role | Value | Image ID | Label | Detector | VLM | Quality | Notes |",
 352:         "|---|---:|---|---|---|---|---|---|---|---|",
 353:     ]
 354: 
 355:     if candidates.empty:
 356:         lines.append("| none | 0/4 | none | none |  |  |  |  |  | No AD2 source candidates found. |")
 357:     else:
 358:         for _, r in candidates.head(80).iterrows():
 359:             lines.append(
 360:                 f"| `{r['file']}` | {r['ad2_coverage_count']}/4 | "
 361:                 f"{r['source_role']} | {r['qcr_assembly_value']} | "
 362:                 f"{r['image_id_col']} | {r['label_col']} | "
 363:                 f"{r['detector_score_col']} | {r['vlm_score_col']} | "
 364:                 f"{r['quality_col']} | {r['notes']} |"
 365:             )
 366: 
 367:     lines += [
 368:         "",
 369:         "## Decision rule",
 370:         "",
 371:         "- If high-value files exist, proceed to Stage 18-B1: assemble AD2 QCR predictions from existing sources.",
 372:         "- If only medium-value files exist, inspect whether detector/VLM/quality sources can be joined by image key.",
 373:         "- If no high/medium-value files exist, proceed to Stage 18-C: generate AD2 QCR predictions from scratch.",
 374:         "",
 375:     ]
 376: 
 377:     if high > 0:
 378:         decision = "proceed_to_stage18_b1_assemble_existing_sources"
... 函数剩余 22 行已省略 ...
```

**函数 `main`：`experiments/stage18_ad2_qcr_ablation/build_stage18_b1_ad2_qcr_schema_profile.py`，第 350–460 行**

```python
 350: def main() -> None:
 351:     OUT_DIR.mkdir(parents=True, exist_ok=True)
 352:     DOC_DIR.mkdir(parents=True, exist_ok=True)
 353: 
 354:     profile_rows = []
 355:     column_rows = []
 356: 
 357:     for path in SOURCE_FILES:
 358:         profile, cols = profile_file(path)
 359:         profile_rows.append(profile)
 360:         column_rows.extend(cols)
 361: 
 362:     profile = pd.DataFrame(profile_rows)
 363:     columns = pd.DataFrame(column_rows)
 364:     joins = join_feasibility(SOURCE_FILES)
 365: 
 366:     profile.to_csv(OUT_PROFILE, index=False, lineterminator="\n")
 367:     columns.to_csv(OUT_COLUMNS, index=False, lineterminator="\n")
 368:     joins.to_csv(OUT_JOIN, index=False, lineterminator="\n")
 369: 
 370:     qcr_ready = profile[profile["qcr_readiness"] == "qcr_ready"]
 371:     partial = profile[profile["qcr_readiness"] == "partial_join_source"]
 372: 
 373:     lines = [
 374:         "# Stage 18-B1 AD2 QCR Source Schema Profile",
 375:         "",
 376:         "## Purpose",
 377:         "",
 378:         "Inspect Stage11/Stage13 AD2 source files to decide whether AD2 four-category QCR predictions can be assembled from existing files.",
 379:         "",
 380:         "## Summary",
 381:         "",
 382:         f"- qcr_ready files: `{len(qcr_ready)}`",
 383:         f"- partial_join_source files: `{len(partial)}`",
 384:         "",
 385:         "## File profile",
 386:         "",
 387:         "| File | Rows | AD2 coverage | Readiness | Key cols | Label cols | Detector-like | VLM-like | Quality-like |",
 388:         "|---|---:|---:|---|---|---|---|---|---|",
 389:     ]
 390: 
 391:     for _, r in profile.iterrows():
 392:         lines.append(
 393:             f"| `{r['file']}` | {r['num_rows']} | {r['ad2_coverage_count']}/4 | "
 394:             f"{r['qcr_readiness']} | {r['key_cols']} | {r['label_cols']} | "
 395:             f"{r['detector_like_cols']} | {r['vlm_like_cols']} | {r['quality_like_cols']} |"
 396:         )
 397: 
 398:     lines += [
 399:         "",
 400:         "## Strong join candidates",
 401:         "",
 402:         "| File A | File B | Key A | Key B | Overlap | Ratio A | Ratio B | Notes |",
 403:         "|---|---|---|---|---:|---:|---:|---|",
 404:     ]
 405: 
 406:     if joins.empty:
 407:         lines.append("| none | none |  |  | 0 | 0 | 0 | no files loaded |")
 408:     else:
 409:         top = joins.sort_values(["overlap_count", "overlap_ratio_a"], ascending=[False, False]).head(20)
 410:         for _, r in top.iterrows():
 411:             lines.append(
 412:                 f"| `{r['file_a']}` | `{r['file_b']}` | {r['best_key_a']} | {r['best_key_b']} | "
 413:                 f"{r['overlap_count']} | {float(r['overlap_ratio_a']):.3f} | "
 414:                 f"{float(r['overlap_ratio_b']):.3f} | {r['notes']} |"
 415:             )
 416: 
 417:     lines += [
 418:         "",
 419:         "## Decision rule",
 420:         "",
 421:         "- If a qcr_ready file exists, proceed to Stage 18-B2 directly.",
 422:         "- If partial files have strong joins and contain D/M/Q/label across files, assemble in Stage 18-B2.",
 423:         "- If VLM or quality is missing, proceed to Stage 18-C to generate missing AD2 QCR predictions.",
 424:         "",
 425:     ]
 426: 
 427:     if len(qcr_ready) > 0:
 428:         decision = "proceed_to_stage18_b2_direct_qcr_assembly"
 429:     elif len(partial) > 0:
... 函数剩余 31 行已省略 ...
```

**函数 `main`：`experiments/stage18_ad2_qcr_ablation/build_stage18_b4_ad2_qcr_claim_safe_decision.py`，第 79–308 行**

```python
  79: def main() -> None:
  80:     OUT_DIR.mkdir(parents=True, exist_ok=True)
  81:     DOC_DIR.mkdir(parents=True, exist_ok=True)
  82: 
  83:     b2_summary = read_csv(IN_B2_SUMMARY)
  84:     b2_deltas = read_csv(IN_B2_DELTAS)
  85:     b3 = read_csv(IN_B3_RANKED)
  86: 
  87:     b3["quality_family"] = b3["q_source"].map(quality_family)
  88:     b3["invalid_as_candidate_quality"] = b3["q_source"].map(is_invalid_q_source)
  89:     b3["passes_mean_positive"] = b3["mean_delta_V4_minus_V3"] > 0
  90:     b3["passes_3_of_4_wins"] = b3["wins_V4_over_V3"] >= 3
  91:     b3["claim_safe_candidate"] = (
  92:         (~b3["invalid_as_candidate_quality"])
  93:         & (b3["quality_family"] != "unsupported_or_vlm_evidence")
  94:         & b3["passes_mean_positive"]
  95:         & b3["passes_3_of_4_wins"]
  96:     )
  97: 
  98:     valid = b3[b3["claim_safe_candidate"]].copy()
  99: 
 100:     valid_perf = valid.sort_values(
 101:         ["mean_delta_V4_minus_V3", "wins_V4_over_V3", "mean_auroc_V4_quality"],
 102:         ascending=[False, False, False],
 103:     ).reset_index(drop=True)
 104: 
 105:     valid_stable = valid.sort_values(
 106:         ["worst_category_delta_V4_minus_V3", "wins_V4_over_V3", "mean_delta_V4_minus_V3"],
 107:         ascending=[False, False, False],
 108:     ).reset_index(drop=True)
 109: 
 110:     all_valid_ranked = valid.sort_values(
 111:         [
 112:             "wins_V4_over_V3",
 113:             "mean_delta_V4_minus_V3",
 114:             "worst_category_delta_V4_minus_V3",
 115:             "mean_auroc_V4_quality",
 116:         ],
 117:         ascending=[False, False, False, False],
 118:     ).reset_index(drop=True)
 119: 
 120:     all_valid_ranked.to_csv(OUT_VALID, index=False, lineterminator="\n")
 121: 
 122:     b2_v3 = b2_summary[b2_summary["variant_id"] == "V3"].iloc[0]
 123:     b2_v4 = b2_summary[b2_summary["variant_id"] == "V4"].iloc[0]
 124:     b2_v6 = b2_summary[b2_summary["variant_id"] == "V6"].iloc[0]
 125: 
 126:     b2_v4_delta = b2_deltas[
 127:         b2_deltas["comparison"] == "Quality-Calibrated QCR vs naive fusion"
 128:     ].iloc[0]["delta_a_minus_b"]
 129: 
 130:     b2_v6_delta = b2_deltas[
 131:         b2_deltas["comparison"] == "Adaptive refinement vs naive fusion"
 132:     ].iloc[0]["delta_a_minus_b"]
 133: 
 134:     decision_rows = []
 135: 
 136:     # 1. Raw B2 default result.
 137:     decision_rows.append(
 138:         {
 139:             "case_id": "B2_default_q_source",
 140:             "paper_status": "boundary_result_not_main_claim",
 141:             "q_source": "candidate_score_mean_max",
 142:             "q_direction": "direct",
 143:             "quality_family": "candidate_region_score",
 144:             "mean_auroc_V3_naive": float(b2_v3["mean_image_auroc"]),
 145:             "mean_auroc_V4_quality": float(b2_v4["mean_image_auroc"]),
 146:             "mean_auroc_V6_adaptive": float(b2_v6["mean_image_auroc"]),
 147:             "delta_V4_minus_V3": float(b2_v4_delta),
 148:             "delta_V6_minus_V3": float(b2_v6_delta),
 149:             "wins_V4_over_V3": np.nan,
 150:             "worst_category": "",
 151:             "worst_category_delta_V4_minus_V3": np.nan,
 152:             "decision": "Do not use as main AD2 QCR evidence because V4 is below V3.",
 153:         }
 154:     )
 155: 
 156:     # 2. Invalid best source.
 157:     invalid_best = b3.iloc[0]
 158:     decision_rows.append(
... 函数剩余 150 行已省略 ...
```

**函数 `main`：`experiments/stage18_ad2_qcr_ablation/build_stage18_b7_qcr_final_claim_update.py`，第 45–245 行**

```python
  45: def main() -> None:
  46:     OUT_DIR.mkdir(parents=True, exist_ok=True)
  47:     DOC_DIR.mkdir(parents=True, exist_ok=True)
  48: 
  49:     summary = read_csv(IN_B6_SUMMARY)
  50:     folds = read_csv(IN_B6_FOLDS)
  51: 
  52:     s = summary[summary["selector"] == LOCKED_SELECTOR]
  53:     if s.empty:
  54:         raise RuntimeError(f"Missing locked selector in B6 summary: {LOCKED_SELECTOR}")
  55:     s = s.iloc[0]
  56: 
  57:     f = folds[folds["selector"] == LOCKED_SELECTOR].copy()
  58:     if f.empty:
  59:         raise RuntimeError(f"Missing locked selector folds: {LOCKED_SELECTOR}")
  60: 
  61:     # Decide final AD2 score variant.
  62:     quality_mean = float(s["mean_test_quality_qcr"])
  63:     adaptive_mean = float(s["mean_test_adaptive_qcr"])
  64: 
  65:     if quality_mean >= adaptive_mean:
  66:         final_variant = "Quality-Calibrated QCR"
  67:         final_score = quality_mean
  68:         final_delta = float(s["mean_delta_quality_minus_V3"])
  69:         final_wins = int(s["wins_quality_over_V3"])
  70:         final_worst_delta = float(s["worst_quality_delta"])
  71:         final_note = "Quality-only calibration is selected because it is slightly stronger than adaptive refinement on AD2."
  72:     else:
  73:         final_variant = "Quality-Calibrated QCR + adaptive refinement"
  74:         final_score = adaptive_mean
  75:         final_delta = float(s["mean_delta_adaptive_minus_V3"])
  76:         final_wins = int(s["wins_adaptive_over_V3"])
  77:         final_worst_delta = float(s["worst_adaptive_delta"])
  78:         final_note = "Adaptive refinement is selected because it is stronger than quality-only calibration on AD2."
  79: 
  80:     final_table = pd.DataFrame(
  81:         [
  82:             {
  83:                 "setting": "AD2 four-category LOCO policy",
  84:                 "method": "Naive detector-crop fusion",
  85:                 "mean_image_auroc": float(s["mean_test_V3"]),
  86:                 "delta_vs_naive": 0.0,
  87:                 "wins_vs_naive": "",
  88:                 "paper_role": "baseline",
  89:             },
  90:             {
  91:                 "setting": "AD2 four-category LOCO policy",
  92:                 "method": "Quality-Calibrated QCR",
  93:                 "mean_image_auroc": quality_mean,
  94:                 "delta_vs_naive": float(s["mean_delta_quality_minus_V3"]),
  95:                 "wins_vs_naive": f"{int(s['wins_quality_over_V3'])}/4",
  96:                 "paper_role": "main_qcr_support",
  97:             },
  98:             {
  99:                 "setting": "AD2 four-category LOCO policy",
 100:                 "method": "Quality-Calibrated QCR + adaptive refinement",
 101:                 "mean_image_auroc": adaptive_mean,
 102:                 "delta_vs_naive": float(s["mean_delta_adaptive_minus_V3"]),
 103:                 "wins_vs_naive": f"{int(s['wins_adaptive_over_V3'])}/4",
 104:                 "paper_role": "auxiliary_refinement",
 105:             },
 106:         ]
 107:     )
 108: 
 109:     final_table.to_csv(OUT_FINAL_TABLE, index=False, lineterminator="\n")
 110: 
 111:     fold_table = f[
 112:         [
 113:             "heldout_category",
 114:             "selected_q_source",
 115:             "selected_q_direction",
 116:             "selected_eta",
 117:             "selected_gamma",
 118:             "test_V3",
 119:             "test_quality_qcr",
 120:             "test_adaptive_qcr",
 121:             "test_delta_quality_minus_V3",
 122:             "test_delta_adaptive_minus_V3",
 123:         ]
 124:     ].copy()
... 函数剩余 121 行已省略 ...
```

**函数 `build_predictions`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 98–239 行**

```python
  98: def build_predictions() -> pd.DataFrame:
  99:     img = read_csv_strict(IN_IMAGE)
 100:     cand = read_csv_strict(IN_CAND)
 101: 
 102:     img = img[img["category"].isin(AD2_CATEGORIES)].copy()
 103:     cand = cand[cand["category"].isin(AD2_CATEGORIES)].copy()
 104: 
 105:     required_img = ["category", "image_path", "gt_binary", "patchcore_score"]
 106:     for c in required_img:
 107:         if c not in img.columns:
 108:             raise RuntimeError(f"Missing required image-level column: {c}")
 109: 
 110:     # Aggregate candidate quality per image. Do not use GT coverage columns as quality.
 111:     q_cols = [
 112:         c for c in [
 113:             "candidate_score_max",
 114:             "candidate_score_mean",
 115:             "tight_candidate_mask_density",
 116:             "context_candidate_mask_density",
 117:             "map_area",
 118:         ]
 119:         if c in cand.columns
 120:     ]
 121: 
 122:     if not q_cols:
 123:         raise RuntimeError("No candidate quality/source columns found in candidate score file.")
 124: 
 125:     agg_dict = {}
 126:     if "candidate_score_mean" in cand.columns:
 127:         agg_dict["candidate_score_mean_max"] = ("candidate_score_mean", "max")
 128:         agg_dict["candidate_score_mean_mean"] = ("candidate_score_mean", "mean")
 129:     if "candidate_score_max" in cand.columns:
 130:         agg_dict["candidate_score_max_max"] = ("candidate_score_max", "max")
 131:     if "tight_candidate_mask_density" in cand.columns:
 132:         agg_dict["tight_candidate_mask_density_max"] = ("tight_candidate_mask_density", "max")
 133:     if "context_candidate_mask_density" in cand.columns:
 134:         agg_dict["context_candidate_mask_density_max"] = ("context_candidate_mask_density", "max")
 135:     if "candidate_rank" in cand.columns:
 136:         agg_dict["num_candidates"] = ("candidate_rank", "count")
 137: 
 138:     q = cand.groupby(["category", "image_path"], as_index=False).agg(**agg_dict)
 139: 
 140:     df = img.merge(q, on=["category", "image_path"], how="left", validate="one_to_one")
 141: 
 142:     # Raw evidence.
 143:     df["D_raw_patchcore"] = pd.to_numeric(df["patchcore_score"], errors="coerce")
 144: 
 145:     if "context_topk_mean_score" in df.columns:
 146:         df["M_raw_crop_topk"] = pd.to_numeric(df["context_topk_mean_score"], errors="coerce")
 147:         m_source = "context_topk_mean_score"
 148:     elif "context_topk_max_score" in df.columns:
 149:         df["M_raw_crop_topk"] = pd.to_numeric(df["context_topk_max_score"], errors="coerce")
 150:         m_source = "context_topk_max_score"
 151:     elif "context_top1_score" in df.columns:
 152:         df["M_raw_crop_topk"] = pd.to_numeric(df["context_top1_score"], errors="coerce")
 153:         m_source = "context_top1_score"
 154:     else:
 155:         raise RuntimeError("No context crop VLM score column found in image-level predictions.")
 156: 
 157:     if "full_image_score" in df.columns:
 158:         df["F_raw_full_image_vlm"] = pd.to_numeric(df["full_image_score"], errors="coerce")
 159:         full_source = "full_image_score"
 160:     elif "full_image_anomaly_score" in df.columns:
 161:         df["F_raw_full_image_vlm"] = pd.to_numeric(df["full_image_anomaly_score"], errors="coerce")
 162:         full_source = "full_image_anomaly_score"
 163:     else:
 164:         df["F_raw_full_image_vlm"] = np.nan
 165:         full_source = "missing"
 166: 
 167:     if "candidate_score_mean_max" in df.columns:
 168:         df["Q_raw_candidate_quality"] = pd.to_numeric(df["candidate_score_mean_max"], errors="coerce")
 169:         q_source = "max(candidate_score_mean)"
 170:     elif "candidate_score_max_max" in df.columns:
 171:         df["Q_raw_candidate_quality"] = pd.to_numeric(df["candidate_score_max_max"], errors="coerce")
 172:         q_source = "max(candidate_score_max)"
 173:     else:
 174:         raise RuntimeError("No usable non-GT candidate quality column found after aggregation.")
 175: 
 176:     # Per-category normalization to avoid cross-category scale leakage.
 177:     df = norm_by_category(df, "D_raw_patchcore", "D")
... 函数剩余 62 行已省略 ...
```

**函数 `evaluate`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 242–347 行**

```python
 242: def evaluate(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
 243:     variants = [
 244:         ("V0", "Detector only", "V0_detector_only", "baseline_detector"),
 245:         ("V1", "Full-image VLM", "V1_full_image_vlm", "baseline_vlm"),
 246:         ("V2", "Crop top-k VLM", "V2_crop_topk_vlm", "baseline_crop_vlm"),
 247:         ("V3", "Naive detector-crop fusion", "V3_naive_detector_crop_fusion", "fusion_baseline"),
 248:         ("V4", "Quality-Calibrated QCR", "V4_quality_calibrated_qcr", "main_method_core"),
 249:         ("V5", "Fixed Q+C fusion", "V5_fixed_qc_diagnostic", "diagnostic_not_final"),
 250:         ("V6", "Quality-Calibrated QCR + adaptive refinement", "V6_adaptive_qcr_refinement", "final_refinement"),
 251:     ]
 252: 
 253:     rows = []
 254: 
 255:     for cat, sub in pred.groupby("category"):
 256:         y = pd.to_numeric(sub["gt_binary"], errors="coerce").astype(int)
 257: 
 258:         for vid, method, col, role in variants:
 259:             if col not in sub.columns:
 260:                 continue
 261: 
 262:             score = pd.to_numeric(sub[col], errors="coerce")
 263:             if score.notna().sum() == 0:
 264:                 continue
 265: 
 266:             auroc = safe_auroc(y, score)
 267:             ap = safe_ap(y, score)
 268: 
 269:             ok = y.notna() & score.notna()
 270:             if ok.sum() > 0 and y[ok].nunique() >= 2:
 271:                 best_f1, best_acc, best_thr = best_f1_and_acc(y[ok].values, score[ok].values)
 272:             else:
 273:                 best_f1, best_acc, best_thr = np.nan, np.nan, np.nan
 274: 
 275:             rows.append(
 276:                 {
 277:                     "category": cat,
 278:                     "variant_id": vid,
 279:                     "method": method,
 280:                     "score_col": col,
 281:                     "paper_role": role,
 282:                     "num_images": int(ok.sum()),
 283:                     "num_normal": int((y[ok] == 0).sum()),
 284:                     "num_anomaly": int((y[ok] == 1).sum()),
 285:                     "image_auroc": auroc,
 286:                     "average_precision": ap,
 287:                     "best_f1": best_f1,
 288:                     "best_accuracy": best_acc,
 289:                     "best_threshold": best_thr,
 290:                 }
 291:             )
 292: 
 293:     per_cat = pd.DataFrame(rows)
 294: 
 295:     summary_rows = []
 296:     for vid, method, col, role in variants:
 297:         sub = per_cat[per_cat["variant_id"] == vid].copy()
 298:         if sub.empty:
 299:             continue
 300: 
 301:         summary_rows.append(
 302:             {
 303:                 "variant_id": vid,
 304:                 "method": method,
 305:                 "score_col": col,
 306:                 "paper_role": role,
 307:                 "num_categories": int(sub["category"].nunique()),
 308:                 "mean_image_auroc": float(sub["image_auroc"].mean()),
 309:                 "std_image_auroc": float(sub["image_auroc"].std(ddof=0)),
 310:                 "mean_average_precision": float(sub["average_precision"].mean()),
 311:                 "mean_best_f1": float(sub["best_f1"].mean()),
 312:                 "mean_best_accuracy": float(sub["best_accuracy"].mean()),
 313:             }
 314:         )
 315: 
 316:     summary = pd.DataFrame(summary_rows)
 317: 
 318:     delta_pairs = [
 319:         ("V4", "V3", "Quality-Calibrated QCR vs naive fusion"),
 320:         ("V6", "V4", "Adaptive refinement vs Quality-Calibrated QCR"),
 321:         ("V6", "V3", "Adaptive refinement vs naive fusion"),
... 函数剩余 26 行已省略 ...
```

**函数 `write_report`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 362–441 行**

```python
 362: def write_report(pred: pd.DataFrame, per_cat: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
 363:     lines = [
 364:         "# Stage 18-B2 AD2 Four-category QCR Ablation",
 365:         "",
 366:         "## Purpose",
 367:         "",
 368:         "Assemble AD2 four-category QCR ablation from existing Stage11 image-level VLM predictions and candidate-level quality evidence.",
 369:         "",
 370:         "This aligns the QCR ablation with the AD2 four-category system-level baseline setting.",
 371:         "",
 372:         "## Data",
 373:         "",
 374:         f"- input image-level predictions: `{IN_IMAGE.relative_to(ROOT)}`",
 375:         f"- input candidate scores: `{IN_CAND.relative_to(ROOT)}`",
 376:         f"- assembled images: `{len(pred)}`",
 377:         f"- categories: `{'; '.join(sorted(pred['category'].unique()))}`",
 378:         "- detector evidence `D`: normalized `patchcore_score`",
 379:         "- crop VLM evidence `M`: normalized context top-k VLM score",
 380:         "- candidate quality `Q`: normalized non-GT candidate score evidence",
 381:         "- consistency `K`: soft high-high consistency `D*M`",
 382:         "",
 383:         "## Summary table",
 384:         "",
 385:         "| Variant | Method | Role | Mean AUROC | Mean F1 |",
 386:         "|---|---|---|---:|---:|",
 387:     ]
 388: 
 389:     for _, r in summary.iterrows():
 390:         lines.append(
 391:             f"| {r['variant_id']} | {r['method']} | {r['paper_role']} | "
 392:             f"{fmt(r['mean_image_auroc'])} | {fmt(r['mean_best_f1'])} |"
 393:         )
 394: 
 395:     lines += [
 396:         "",
 397:         "## Claim-ready deltas",
 398:         "",
 399:         "| Comparison | Delta AUROC | A | B |",
 400:         "|---|---:|---:|---:|",
 401:     ]
 402: 
 403:     for _, r in deltas.iterrows():
 404:         lines.append(
 405:             f"| {r['comparison']} | {signed(r['delta_a_minus_b'])} | "
 406:             f"{fmt(r['mean_image_auroc_a'])} | {fmt(r['mean_image_auroc_b'])} |"
 407:         )
 408: 
 409:     lines += [
 410:         "",
 411:         "## Per-category AUROC",
 412:         "",
 413:         "| Category | Variant | Method | AUROC | F1 |",
 414:         "|---|---|---|---:|---:|",
 415:     ]
 416: 
 417:     for _, r in per_cat.iterrows():
 418:         lines.append(
 419:             f"| {r['category']} | {r['variant_id']} | {r['method']} | "
 420:             f"{fmt(r['image_auroc'])} | {fmt(r['best_f1'])} |"
 421:         )
 422: 
 423:     lines += [
 424:         "",
 425:         "## Interpretation rules",
 426:         "",
 427:         "- If V4 improves over V3, AD2 supports candidate quality calibration.",
 428:         "- If V6 only slightly improves over V4, keep adaptive consistency as refinement.",
 429:         "- If V5 is strong but unstable or not selected, keep fixed Q+C as diagnostic.",
 430:         "- Do not use this table to claim pixel-level segmentation SOTA.",
 431:         "",
 432:         "## Outputs",
 433:         "",
 434:         f"- `{OUT_PRED.relative_to(ROOT)}`",
 435:         f"- `{OUT_PER_CATEGORY.relative_to(ROOT)}`",
 436:         f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
 437:         f"- `{OUT_DELTAS.relative_to(ROOT)}`",
 438:         "",
 439:     ]
 440: 
 441:     OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
```

**函数 `write_report`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b3_ad2_q_source_sweep.py`，第 235–291 行**

```python
 235: def write_report(per_cat: pd.DataFrame, summary: pd.DataFrame, ranked: pd.DataFrame) -> None:
 236:     best = ranked.iloc[0]
 237: 
 238:     lines = [
 239:         "# Stage 18-B3 AD2 Q Source Sweep",
 240:         "",
 241:         "## Purpose",
 242:         "",
 243:         "Diagnose whether the Stage 18-B2 AD2 QCR drop against naive fusion is caused by the selected candidate quality source.",
 244:         "",
 245:         "The sweep keeps the same QCR formulas as the current paper method and only changes the non-GT candidate quality source.",
 246:         "",
 247:         "## Best ranked source",
 248:         "",
 249:         f"- q_source: `{best['q_source']}`",
 250:         f"- q_direction: `{best['q_direction']}`",
 251:         f"- mean V3 AUROC: `{best['mean_auroc_V3_naive']:.4f}`",
 252:         f"- mean V4 AUROC: `{best['mean_auroc_V4_quality']:.4f}`",
 253:         f"- mean V6 AUROC: `{best['mean_auroc_V6_adaptive']:.4f}`",
 254:         f"- V4 minus V3: `{best['mean_delta_V4_minus_V3']:+.4f}`",
 255:         f"- V6 minus V3: `{best['mean_delta_V6_minus_V3']:+.4f}`",
 256:         f"- V4 wins over V3: `{int(best['wins_V4_over_V3'])}/4`",
 257:         f"- worst category: `{best['worst_category']}`",
 258:         f"- worst category delta V4-V3: `{best['worst_category_delta_V4_minus_V3']:+.4f}`",
 259:         "",
 260:         "## Top 10 sources",
 261:         "",
 262:         "| Rank | Q source | Direction | V3 | V4 | V6 | V4-V3 | V6-V3 | Wins V4/V3 | Worst category |",
 263:         "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
 264:     ]
 265: 
 266:     for i, (_, r) in enumerate(ranked.head(10).iterrows(), start=1):
 267:         lines.append(
 268:             f"| {i} | {r['q_source']} | {r['q_direction']} | "
 269:             f"{r['mean_auroc_V3_naive']:.4f} | {r['mean_auroc_V4_quality']:.4f} | "
 270:             f"{r['mean_auroc_V6_adaptive']:.4f} | {r['mean_delta_V4_minus_V3']:+.4f} | "
 271:             f"{r['mean_delta_V6_minus_V3']:+.4f} | {int(r['wins_V4_over_V3'])}/4 | "
 272:             f"{r['worst_category']} |"
 273:         )
 274: 
 275:     lines += [
 276:         "",
 277:         "## Decision rule",
 278:         "",
 279:         "- If a non-GT Q source gives V4 > V3 on mean AUROC and wins at least 3/4 categories, AD2 QCR can be promoted to a stronger supporting ablation.",
 280:         "- If no Q source passes that threshold, AD2 QCR should be reported as a boundary/diagnostic result rather than a main claim.",
 281:         "- Do not select a Q source using ground-truth overlap, ground-truth mask quality, or label-derived information.",
 282:         "",
 283:         "## Outputs",
 284:         "",
 285:         f"- `{OUT_PER_CATEGORY.relative_to(ROOT)}`",
 286:         f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
 287:         f"- `{OUT_RANKED.relative_to(ROOT)}`",
 288:         "",
 289:     ]
 290: 
 291:     OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
```

**函数 `aggregate_candidate_quality_sources`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b5_ad2_loco_qcr_policy_optimization.py`，第 65–93 行**

```python
  65: def aggregate_candidate_quality_sources(cand: pd.DataFrame) -> pd.DataFrame:
  66:     group_cols = ["category", "image_path"]
  67: 
  68:     allowed_base_cols = [
  69:         "candidate_score_mean",
  70:         "candidate_score_max",
  71:         "tight_candidate_mask_density",
  72:         "context_candidate_mask_density",
  73:         "map_area",
  74:     ]
  75: 
  76:     agg_spec = {}
  77: 
  78:     for col in allowed_base_cols:
  79:         if col not in cand.columns:
  80:             continue
  81: 
  82:         # Non-GT candidate/source statistics only.
  83:         agg_spec[f"{col}_max"] = (col, "max")
  84:         agg_spec[f"{col}_mean"] = (col, "mean")
  85:         agg_spec[f"{col}_min"] = (col, "min")
  86: 
  87:     if "candidate_rank" in cand.columns:
  88:         agg_spec["num_candidates"] = ("candidate_rank", "count")
  89: 
  90:     if not agg_spec:
  91:         raise RuntimeError("No valid candidate-quality source columns found.")
  92: 
  93:     return cand.groupby(group_cols, as_index=False).agg(**agg_spec)
```

**函数 `build_base`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b5_ad2_loco_qcr_policy_optimization.py`，第 96–158 行**

```python
  96: def build_base() -> tuple[pd.DataFrame, list[str]]:
  97:     img = read_csv(IN_IMAGE)
  98:     cand = read_csv(IN_CAND)
  99: 
 100:     img = img[img["category"].isin(AD2_CATEGORIES)].copy()
 101:     cand = cand[cand["category"].isin(AD2_CATEGORIES)].copy()
 102: 
 103:     for c in ["category", "image_path", "gt_binary", "patchcore_score"]:
 104:         if c not in img.columns:
 105:             raise RuntimeError(f"Missing required image-level column: {c}")
 106: 
 107:     if "context_topk_mean_score" in img.columns:
 108:         m_col = "context_topk_mean_score"
 109:     elif "context_topk_max_score" in img.columns:
 110:         m_col = "context_topk_max_score"
 111:     elif "context_top1_score" in img.columns:
 112:         m_col = "context_top1_score"
 113:     else:
 114:         raise RuntimeError("No crop/context VLM score column found.")
 115: 
 116:     q = aggregate_candidate_quality_sources(cand)
 117:     df = img.merge(q, on=["category", "image_path"], how="left", validate="one_to_one")
 118: 
 119:     df["D_raw"] = pd.to_numeric(df["patchcore_score"], errors="coerce")
 120:     df["M_raw"] = pd.to_numeric(df[m_col], errors="coerce")
 121: 
 122:     df = norm_by_category(df, "D_raw", "D")
 123:     df = norm_by_category(df, "M_raw", "M")
 124: 
 125:     df["V3_naive"] = 0.5 * df["D"] + 0.5 * df["M"]
 126: 
 127:     protected = {
 128:         "category",
 129:         "image_path",
 130:         "gt_binary",
 131:         "patchcore_score",
 132:         m_col,
 133:         "D_raw",
 134:         "M_raw",
 135:         "D",
 136:         "M",
 137:         "V3_naive",
 138:     }
 139: 
 140:     q_cols = []
 141:     for c in df.columns:
 142:         if c in protected:
 143:             continue
 144: 
 145:         lc = c.lower()
 146: 
 147:         # Hard leakage / invalid evidence filter.
 148:         if any(t in lc for t in ["gt", "label", "target", "full_image", "context_top", "vlm", "clip"]):
 149:             continue
 150: 
 151:         s = pd.to_numeric(df[c], errors="coerce")
 152:         if s.notna().sum() >= 10 and s.nunique(dropna=True) > 1:
 153:             q_cols.append(c)
 154: 
 155:     if not q_cols:
 156:         raise RuntimeError("No safe Q candidate columns after filtering.")
 157: 
 158:     return df, q_cols
```

**函数 `write_report`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b5_ad2_loco_qcr_policy_optimization.py`，第 331–401 行**

```python
 331: def write_report(q_cols: list[str], selected: pd.DataFrame, summary: pd.DataFrame) -> None:
 332:     s = summary.iloc[0]
 333: 
 334:     if s["mean_delta_adaptive_minus_V3"] > 0 and s["wins_adaptive_over_V3"] >= 3:
 335:         final_status = "promote_qcr_as_cross_category_calibrated_ad2_support"
 336:     elif s["mean_delta_quality_minus_V3"] > 0 and s["wins_quality_over_V3"] >= 3:
 337:         final_status = "promote_quality_qcr_without_adaptive_as_ad2_support"
 338:     elif s["mean_delta_adaptive_minus_V3"] > 0:
 339:         final_status = "weak_positive_mean_but_not_category_stable"
 340:     else:
 341:         final_status = "do_not_promote_ad2_qcr_main_claim"
 342: 
 343:     lines = [
 344:         "# Stage 18-B5 AD2 LOCO QCR Policy Optimization",
 345:         "",
 346:         "## Purpose",
 347:         "",
 348:         "Optimize QCR policy without using the held-out AD2 category labels for selection.",
 349:         "",
 350:         "Each fold selects Q source, Q direction, eta, and gamma on three AD2 categories, then evaluates on the held-out category.",
 351:         "",
 352:         "## Safe Q source candidates",
 353:         "",
 354:         "```text",
 355:         *q_cols,
 356:         "```",
 357:         "",
 358:         "## Summary",
 359:         "",
 360:         f"- final_status: `{final_status}`",
 361:         f"- mean test V3 naive: `{fmt(s['mean_test_V3'])}`",
 362:         f"- mean test quality QCR: `{fmt(s['mean_test_quality_qcr'])}`",
 363:         f"- mean test adaptive QCR: `{fmt(s['mean_test_adaptive_qcr'])}`",
 364:         f"- quality QCR minus V3: `{signed(s['mean_delta_quality_minus_V3'])}`",
 365:         f"- adaptive QCR minus V3: `{signed(s['mean_delta_adaptive_minus_V3'])}`",
 366:         f"- quality QCR wins over V3: `{int(s['wins_quality_over_V3'])}/4`",
 367:         f"- adaptive QCR wins over V3: `{int(s['wins_adaptive_over_V3'])}/4`",
 368:         f"- worst adaptive category: `{s['worst_adaptive_category']}`",
 369:         f"- worst adaptive delta: `{signed(s['worst_adaptive_delta'])}`",
 370:         "",
 371:         "## Selected folds",
 372:         "",
 373:         "| Held-out | Selected Q | Direction | eta | gamma | Test V3 | Test Quality | Test Adaptive | Adaptive-V3 |",
 374:         "|---|---|---|---:|---:|---:|---:|---:|---:|",
 375:     ]
 376: 
 377:     for _, r in selected.iterrows():
 378:         lines.append(
 379:             f"| {r['heldout_category']} | {r['selected_q_source']} | {r['selected_q_direction']} | "
 380:             f"{r['selected_eta']:.2f} | {r['selected_gamma']:.2f} | "
 381:             f"{fmt(r['test_V3'])} | {fmt(r['test_quality_qcr'])} | "
 382:             f"{fmt(r['test_adaptive_qcr'])} | {signed(r['test_delta_adaptive_minus_V3'])} |"
 383:         )
 384: 
 385:     lines += [
 386:         "",
 387:         "## Decision rule",
 388:         "",
 389:         "- If adaptive QCR has positive mean delta and wins at least 3/4 held-out categories, QCR can be promoted as cross-category calibrated AD2 support.",
 390:         "- If only mean delta is positive but wins fewer than 3/4, report AD2 as weak/boundary support.",
 391:         "- If mean delta is negative, keep AD2 QCR as source-sensitivity diagnostic and retain VisA as the main ablation.",
 392:         "",
 393:         "## Outputs",
 394:         "",
 395:         f"- `{OUT_ALL_CONFIGS.relative_to(ROOT)}`",
 396:         f"- `{OUT_SELECTED.relative_to(ROOT)}`",
 397:         f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
 398:         "",
 399:     ]
 400: 
 401:     OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
```

**函数 `is_valid_candidate_quality_source`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b6_ad2_loco_robust_qcr_selector_sweep.py`，第 34–61 行**

```python
  34: def is_valid_candidate_quality_source(q_source: str) -> bool:
  35:     q = str(q_source).lower()
  36: 
  37:     # Hard invalid: these are VLM/evidence scores or label-like terms, not candidate-quality proxies.
  38:     invalid_tokens = [
  39:         "full_image",
  40:         "context_top",
  41:         "tight_top",
  42:         "vlm",
  43:         "clip",
  44:         "gt",
  45:         "label",
  46:         "target",
  47:         "anomaly_binary",
  48:     ]
  49: 
  50:     if any(t in q for t in invalid_tokens):
  51:         return False
  52: 
  53:     valid_prefixes = [
  54:         "candidate_score_",
  55:         "tight_candidate_mask_density",
  56:         "context_candidate_mask_density",
  57:         "map_area",
  58:         "num_candidates",
  59:     ]
  60: 
  61:     return any(q.startswith(p) for p in valid_prefixes)
```

**函数 `train_summary`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b6_ad2_loco_robust_qcr_selector_sweep.py`，第 64–97 行**

```python
  64: def train_summary(train: pd.DataFrame) -> pd.DataFrame:
  65:     g = (
  66:         train.groupby(GROUP_COLS, as_index=False)
  67:         .agg(
  68:             train_mean_V3=("auroc_V3_naive", "mean"),
  69:             train_mean_quality=("auroc_quality_qcr", "mean"),
  70:             train_mean_adaptive=("auroc_adaptive_qcr", "mean"),
  71:             train_delta_quality=("delta_quality_minus_V3", "mean"),
  72:             train_delta_adaptive=("delta_adaptive_minus_V3", "mean"),
  73:             train_wins_quality=("delta_quality_minus_V3", lambda x: int((x > 0).sum())),
  74:             train_wins_adaptive=("delta_adaptive_minus_V3", lambda x: int((x > 0).sum())),
  75:             train_worst_delta_quality=("delta_quality_minus_V3", "min"),
  76:             train_worst_delta_adaptive=("delta_adaptive_minus_V3", "min"),
  77:             train_std_delta_adaptive=("delta_adaptive_minus_V3", "std"),
  78:         )
  79:         .fillna({"train_std_delta_adaptive": 0.0})
  80:     )
  81: 
  82:     # A conservative robust objective: prefer positive average delta, many wins, and avoid catastrophic categories.
  83:     g["robust_delta_score"] = (
  84:         g["train_delta_adaptive"]
  85:         + 0.50 * g["train_worst_delta_adaptive"]
  86:         + 0.01 * g["train_wins_adaptive"]
  87:         - 0.05 * g["train_std_delta_adaptive"]
  88:     )
  89: 
  90:     # Another conservative objective focused on quality-only, because gamma often does not help.
  91:     g["robust_quality_score"] = (
  92:         g["train_delta_quality"]
  93:         + 0.50 * g["train_worst_delta_quality"]
  94:         + 0.01 * g["train_wins_quality"]
  95:     )
  96: 
  97:     return g
```

**函数 `select_config`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b6_ad2_loco_robust_qcr_selector_sweep.py`，第 100–163 行**

```python
 100: def select_config(ts: pd.DataFrame, selector: str) -> pd.Series:
 101:     work = ts.copy()
 102: 
 103:     if selector == "B5_baseline_max_train_adaptive_auroc":
 104:         return work.sort_values(
 105:             ["train_mean_adaptive", "train_delta_adaptive", "train_wins_adaptive", "train_worst_delta_adaptive"],
 106:             ascending=[False, False, False, False],
 107:         ).iloc[0]
 108: 
 109:     if selector == "max_train_delta_adaptive":
 110:         return work.sort_values(
 111:             ["train_delta_adaptive", "train_wins_adaptive", "train_worst_delta_adaptive", "train_mean_adaptive"],
 112:             ascending=[False, False, False, False],
 113:         ).iloc[0]
 114: 
 115:     if selector == "wins_then_delta_adaptive":
 116:         return work.sort_values(
 117:             ["train_wins_adaptive", "train_delta_adaptive", "train_worst_delta_adaptive", "train_mean_adaptive"],
 118:             ascending=[False, False, False, False],
 119:         ).iloc[0]
 120: 
 121:     if selector == "worst_delta_then_mean_delta_adaptive":
 122:         return work.sort_values(
 123:             ["train_worst_delta_adaptive", "train_delta_adaptive", "train_wins_adaptive", "train_mean_adaptive"],
 124:             ascending=[False, False, False, False],
 125:         ).iloc[0]
 126: 
 127:     if selector == "robust_delta_score":
 128:         return work.sort_values(
 129:             ["robust_delta_score", "train_wins_adaptive", "train_delta_adaptive", "train_worst_delta_adaptive"],
 130:             ascending=[False, False, False, False],
 131:         ).iloc[0]
 132: 
 133:     if selector == "robust_quality_score":
 134:         return work.sort_values(
 135:             ["robust_quality_score", "train_wins_quality", "train_delta_quality", "train_worst_delta_quality"],
 136:             ascending=[False, False, False, False],
 137:         ).iloc[0]
 138: 
 139:     if selector == "semantic_candidate_score_max_min_inverted":
 140:         sub = work[
 141:             (work["q_source"] == "candidate_score_max_min")
 142:             & (work["q_direction"] == "inverted")
 143:         ].copy()
 144:         if sub.empty:
 145:             sub = work.copy()
 146:         return sub.sort_values(
 147:             ["robust_delta_score", "train_wins_adaptive", "train_delta_adaptive", "train_worst_delta_adaptive"],
 148:             ascending=[False, False, False, False],
 149:         ).iloc[0]
 150: 
 151:     if selector == "semantic_candidate_score_max_mean_inverted":
 152:         sub = work[
 153:             (work["q_source"] == "candidate_score_max_mean")
 154:             & (work["q_direction"] == "inverted")
 155:         ].copy()
 156:         if sub.empty:
 157:             sub = work.copy()
 158:         return sub.sort_values(
 159:             ["robust_delta_score", "train_wins_adaptive", "train_delta_adaptive", "train_worst_delta_adaptive"],
 160:             ascending=[False, False, False, False],
 161:         ).iloc[0]
 162: 
 163:     raise ValueError(f"Unknown selector: {selector}")
```

**函数 `write_report`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b6_ad2_loco_robust_qcr_selector_sweep.py`，第 274–335 行**

```python
 274: def write_report(valid_sources: list[str], folds: pd.DataFrame, summary: pd.DataFrame) -> None:
 275:     best = summary.sort_values(
 276:         ["mean_delta_adaptive_minus_V3", "wins_adaptive_over_V3", "worst_adaptive_delta"],
 277:         ascending=[False, False, False],
 278:     ).iloc[0]
 279: 
 280:     lines = [
 281:         "# Stage 18-B6 AD2 LOCO Robust QCR Selector Sweep",
 282:         "",
 283:         "## Purpose",
 284:         "",
 285:         "Test whether AD2 QCR can be rescued by a more robust train-category selector rather than the B5 selector that maximizes train adaptive AUROC.",
 286:         "",
 287:         "No held-out category labels are used for selecting Q source, direction, eta, or gamma.",
 288:         "",
 289:         "## Valid candidate-quality sources",
 290:         "",
 291:         "```text",
 292:         *valid_sources,
 293:         "```",
 294:         "",
 295:         "## Best selector",
 296:         "",
 297:         f"- selector: `{best['selector']}`",
 298:         f"- claim_status: `{best['claim_status']}`",
 299:         f"- mean test V3: `{fmt(best['mean_test_V3'])}`",
 300:         f"- mean test adaptive QCR: `{fmt(best['mean_test_adaptive_qcr'])}`",
 301:         f"- adaptive QCR minus V3: `{signed(best['mean_delta_adaptive_minus_V3'])}`",
 302:         f"- adaptive wins over V3: `{int(best['wins_adaptive_over_V3'])}/4`",
 303:         f"- worst adaptive category: `{best['worst_adaptive_category']}`",
 304:         f"- worst adaptive delta: `{signed(best['worst_adaptive_delta'])}`",
 305:         "",
 306:         "## Selector summary",
 307:         "",
 308:         "| Selector | Status | V3 | Adaptive | Delta | Wins | Worst category | Worst delta |",
 309:         "|---|---|---:|---:|---:|---:|---|---:|",
 310:     ]
 311: 
 312:     for _, r in summary.iterrows():
 313:         lines.append(
 314:             f"| {r['selector']} | {r['claim_status']} | "
 315:             f"{fmt(r['mean_test_V3'])} | {fmt(r['mean_test_adaptive_qcr'])} | "
 316:             f"{signed(r['mean_delta_adaptive_minus_V3'])} | "
 317:             f"{int(r['wins_adaptive_over_V3'])}/4 | "
 318:             f"{r['worst_adaptive_category']} | {signed(r['worst_adaptive_delta'])} |"
 319:         )
 320: 
 321:     lines += [
 322:         "",
 323:         "## Decision rule",
 324:         "",
 325:         "- If at least one selector has positive mean adaptive delta and wins at least 3/4 held-out categories, AD2 QCR can be used as supporting cross-category evidence.",
 326:         "- If all selectors have negative mean delta, stop optimizing AD2 QCR and report AD2 as boundary/sensitivity evidence.",
 327:         "",
 328:         "## Outputs",
 329:         "",
 330:         f"- `{OUT_FOLDS.relative_to(ROOT)}`",
 331:         f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
 332:         "",
 333:     ]
 334: 
 335:     OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
```

**函数 `main`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b6_ad2_loco_robust_qcr_selector_sweep.py`，第 338–377 行**

```python
 338: def main() -> None:
 339:     OUT_DIR.mkdir(parents=True, exist_ok=True)
 340:     DOC_DIR.mkdir(parents=True, exist_ok=True)
 341: 
 342:     all_configs = read_csv(IN_ALL_CONFIGS)
 343: 
 344:     all_configs["valid_candidate_quality_source"] = all_configs["q_source"].map(is_valid_candidate_quality_source)
 345:     all_configs = all_configs[all_configs["valid_candidate_quality_source"]].copy()
 346: 
 347:     valid_sources = sorted(all_configs["q_source"].unique().tolist())
 348: 
 349:     selectors = [
 350:         "B5_baseline_max_train_adaptive_auroc",
 351:         "max_train_delta_adaptive",
 352:         "wins_then_delta_adaptive",
 353:         "worst_delta_then_mean_delta_adaptive",
 354:         "robust_delta_score",
 355:         "robust_quality_score",
 356:         "semantic_candidate_score_max_min_inverted",
 357:         "semantic_candidate_score_max_mean_inverted",
 358:     ]
 359: 
 360:     fold_frames = []
 361:     for selector in selectors:
 362:         fold_frames.append(run_selector(all_configs, selector))
 363: 
 364:     folds = pd.concat(fold_frames, ignore_index=True)
 365:     summary = summarize(folds)
 366: 
 367:     folds.to_csv(OUT_FOLDS, index=False, lineterminator="\n")
 368:     summary.to_csv(OUT_SUMMARY, index=False, lineterminator="\n")
 369: 
 370:     write_report(valid_sources, folds, summary)
 371: 
 372:     print("[DONE]", OUT_FOLDS)
 373:     print("[DONE]", OUT_SUMMARY)
 374:     print("[DONE]", OUT_REPORT)
 375:     print()
 376:     print("===== selector summary =====")
 377:     print(summary.to_string(index=False))
```

## 2. 旧 QCR：Consistency 计算代码

**函数 `main`：`experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 70–271 行**

```python
  70: def main() -> None:
  71:     OUT_DIR.mkdir(parents=True, exist_ok=True)
  72:     DOC_DIR.mkdir(parents=True, exist_ok=True)
  73: 
  74:     inventory_rows = [inspect_csv(ROOT / p) for p in SOURCE_PATHS]
  75:     inventory = pd.DataFrame(inventory_rows)
  76:     inventory.to_csv(OUT_INVENTORY, index=False, lineterminator="\n")
  77: 
  78:     plan_rows = [
  79:         {
  80:             "variant_id": "V0",
  81:             "variant": "detector_only",
  82:             "uses_detector_score": True,
  83:             "uses_full_image_vlm": False,
  84:             "uses_crop_vlm": False,
  85:             "uses_quality": False,
  86:             "uses_consistency": False,
  87:             "uses_unknown": False,
  88:             "purpose": "Anchor baseline; proves whether QCR-U beats the detector alone.",
  89:         },
  90:         {
  91:             "variant_id": "V1",
  92:             "variant": "full_image_vlm",
  93:             "uses_detector_score": False,
  94:             "uses_full_image_vlm": True,
  95:             "uses_crop_vlm": False,
  96:             "uses_quality": False,
  97:             "uses_consistency": False,
  98:             "uses_unknown": False,
  99:             "purpose": "Weak VLM sanity baseline; should not be the main comparison target.",
 100:         },
 101:         {
 102:             "variant_id": "V2",
 103:             "variant": "crop_topk_vlm",
 104:             "uses_detector_score": False,
 105:             "uses_full_image_vlm": False,
 106:             "uses_crop_vlm": True,
 107:             "uses_quality": False,
 108:             "uses_consistency": False,
 109:             "uses_unknown": False,
 110:             "purpose": "Tests whether localization-guided crops improve VLM scoring.",
 111:         },
 112:         {
 113:             "variant_id": "V3",
 114:             "variant": "naive_detector_crop_fusion",
 115:             "uses_detector_score": True,
 116:             "uses_full_image_vlm": False,
 117:             "uses_crop_vlm": True,
 118:             "uses_quality": False,
 119:             "uses_consistency": False,
 120:             "uses_unknown": False,
 121:             "purpose": "Naive fusion baseline; QCR-U must beat this or the method is not justified.",
 122:         },
 123:         {
 124:             "variant_id": "V4",
 125:             "variant": "quality_weighted_crop",
 126:             "uses_detector_score": True,
 127:             "uses_full_image_vlm": False,
 128:             "uses_crop_vlm": True,
 129:             "uses_quality": True,
 130:             "uses_consistency": False,
 131:             "uses_unknown": False,
 132:             "purpose": "Tests whether candidate quality contributes beyond crop scoring.",
 133:         },
 134:         {
 135:             "variant_id": "V5",
 136:             "variant": "quality_consistency_fusion",
 137:             "uses_detector_score": True,
 138:             "uses_full_image_vlm": False,
 139:             "uses_crop_vlm": True,
 140:             "uses_quality": True,
 141:             "uses_consistency": True,
 142:             "uses_unknown": False,
 143:             "purpose": "Core QCR-U binary anomaly recognition variant.",
 144:         },
 145:         {
 146:             "variant_id": "V6",
 147:             "variant": "qcr_u_full_optional_unknown",
 148:             "uses_detector_score": True,
 149:             "uses_full_image_vlm": False,
... 函数剩余 122 行已省略 ...
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 164–250 行**

```python
 164: def write_report(delta: pd.DataFrame, summary: pd.DataFrame, failures: pd.DataFrame) -> None:
 165:     lines = []
 166:     lines += [
 167:         "# Stage 16-A2 QCR-U Robustness Check",
 168:         "",
 169:         "## 1. Purpose",
 170:         "",
 171:         "Stage 16-A1 showed that fixed quality-consistency fusion can improve the best protocol.",
 172:         "",
 173:         "Stage 16-A2 checks whether that gain is robust across all protocols, instead of only appearing in the best protocol.",
 174:         "",
 175:         "## 2. Overall Robustness Summary",
 176:         "",
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
 189:         "",
 190:         "## 3. Protocol-level Deltas",
 191:         "",
 192:         "| Backbone | Strategy | Eval Mode | V5 AUROC | V3 AUROC | V4 AUROC | V5-V3 | V5-V4 |",
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
 205:         "",
 206:         "## 4. Failure / Weakness Cases",
 207:         "",
 208:     ]
 209: 
 210:     if failures.empty:
 211:         lines.append("No failure case found under the current checks.")
 212:     else:
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
 225:             )
 226: 
 227:     lines += [
 228:         "",
 229:         "## 5. Decision Rule",
 230:         "",
 231:         "If V5 is consistently better than V3 naive fusion but often worse than V4 quality-only, the consistency term should not be claimed as universally beneficial.",
 232:         "",
 233:         "In that case, the next method should be revised from fixed Q+C fusion to adaptive QCR-U:",
 234:         "",
 235:         "```text",
 236:         "use quality-weighted crop as the stable core;",
 237:         "apply consistency only when detector and VLM evidence are both reliable;",
 238:         "avoid adding consistency under weak/full-image protocols where it hurts.",
 239:         "```",
 240:         "",
 241:         "## 6. Outputs",
 242:         "",
 243:         f"- `{OUT_DELTA.relative_to(ROOT)}`",
... 函数剩余 7 行已省略 ...
```

**函数 `build_base_table`：`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 166–241 行**

```python
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
 177:         "image_key",
 178:         "is_anomaly_final",
 179:         "fallback",
 180:         "has_candidate",
 181:         "num_candidates",
 182:         "vlm_score_norm",
 183:         "detector_score_norm",
 184:         "candidate_quality_norm",
 185:         "high_high_consistency",
 186:     ]
 187:     base_cols = [c for c in base_cols if c in df.columns]
 188: 
 189:     base = df[base_cols].copy()
 190:     base = base.drop_duplicates(
 191:         subset=["backbone", "dataset", "category", "strategy", "eval_mode", "image_key"]
 192:     ).reset_index(drop=True)
 193: 
 194:     for c in [
 195:         "is_anomaly_final",
 196:         "vlm_score_norm",
 197:         "detector_score_norm",
 198:         "candidate_quality_norm",
 199:         "high_high_consistency",
 200:         "num_candidates",
 201:     ]:
 202:         if c in base.columns:
 203:             base[c] = pd.to_numeric(base[c], errors="coerce")
 204: 
 205:     base["D"] = base["detector_score_norm"].fillna(0.0)
 206:     base["M"] = base["vlm_score_norm"].fillna(0.0)
 207:     base["Q"] = base["candidate_quality_norm"].fillna(0.0)
 208:     base["K"] = base["high_high_consistency"].fillna(0.0)
 209: 
 210:     base["score_detector_only"] = base["D"]
 211:     base["score_crop_vlm"] = base["M"]
 212:     base["score_naive"] = 0.5 * base["D"] + 0.5 * base["M"]
 213: 
 214:     base["score_quality_raw"] = (
 215:         0.5 * base["D"]
 216:         + 0.5 * (base["M"] * (0.5 + 0.5 * base["Q"]))
 217:     )
 218: 
 219:     base["score_fixed_qc_raw"] = (
 220:         0.40 * base["D"]
 221:         + 0.40 * base["M"]
 222:         + 0.10 * base["Q"]
 223:         + 0.10 * base["K"]
 224:     )
 225: 
 226:     agreement = (1.0 - (base["D"] - base["M"]).abs()).clip(lower=0.0, upper=1.0)
 227:     mutual_anomaly_evidence = np.minimum(base["D"], base["M"])
 228:     adaptive_gate = base["Q"] * base["K"] * agreement * mutual_anomaly_evidence
 229: 
 230:     base["adaptive_gate"] = adaptive_gate
 231:     base["score_adaptive_qcru_raw"] = base["score_quality_raw"] + 0.05 * adaptive_gate
 232: 
 233:     group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
 234:     for raw_col, out_col in [
 235:         ("score_quality_raw", "score_quality"),
 236:         ("score_fixed_qc_raw", "score_fixed_qc"),
 237:         ("score_adaptive_qcru_raw", "score_adaptive_qcru"),
 238:     ]:
 239:         base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)
 240: 
 241:     return base
```

**函数 `build_decision`：`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 304–370 行**

```python
 304: def build_decision(primary: pd.DataFrame, per_config: pd.DataFrame) -> pd.DataFrame:
 305:     rows = []
 306: 
 307:     def get_delta(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
 308:         idx = ["backbone", "dataset", "strategy", "eval_mode"]
 309:         piv = df.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
 310:         piv.columns.name = None
 311:         if left not in piv.columns or right not in piv.columns:
 312:             return pd.DataFrame()
 313:         piv[f"delta_{left}_minus_{right}"] = piv[left] - piv[right]
 314:         return piv
 315: 
 316:     for scope, df in [("primary_protocol", primary), ("all_protocols", per_config)]:
 317:         for left, right, label in [
 318:             ("V6", "V3", "adaptive_qcru_minus_naive"),
 319:             ("V6", "V4", "adaptive_qcru_minus_quality"),
 320:             ("V6", "V5", "adaptive_qcru_minus_fixed_qc"),
 321:             ("V4", "V3", "quality_minus_naive"),
 322:         ]:
 323:             d = get_delta(df, left, right)
 324:             if d.empty:
 325:                 continue
 326:             delta_col = f"delta_{left}_minus_{right}"
 327:             rows.append(
 328:                 {
 329:                     "scope": scope,
 330:                     "comparison": label,
 331:                     "num_protocols": len(d),
 332:                     "wins": int((d[delta_col] > 0).sum()),
 333:                     "win_rate": float((d[delta_col] > 0).mean()),
 334:                     "mean_delta": float(d[delta_col].mean()),
 335:                     "median_delta": float(d[delta_col].median()),
 336:                     "min_delta": float(d[delta_col].min()),
 337:                     "max_delta": float(d[delta_col].max()),
 338:                 }
 339:             )
 340: 
 341:     decision = pd.DataFrame(rows)
 342: 
 343:     # Conservative final recommendation.
 344:     primary_v6_v4 = decision[
 345:         (decision["scope"] == "primary_protocol")
 346:         & (decision["comparison"] == "adaptive_qcru_minus_quality")
 347:     ]
 348: 
 349:     if not primary_v6_v4.empty:
 350:         mean_delta = float(primary_v6_v4.iloc[0]["mean_delta"])
 351:         if mean_delta >= 0.005:
 352:             recommendation = "Adaptive QCR-U can be presented as the final candidate method."
 353:             method_name = "Adaptive QCR-U"
 354:         elif mean_delta > 0:
 355:             recommendation = "Use Quality-Calibrated QCR as the main method; describe adaptive consistency as a small gated refinement."
 356:             method_name = "Quality-Calibrated QCR with adaptive consistency refinement"
 357:         else:
 358:             recommendation = "Do not use Adaptive QCR-U as final method; use quality-weighted fusion."
 359:             method_name = "Quality-Calibrated Localization-Guided Fusion"
 360:     else:
 361:         recommendation = "Insufficient primary comparison."
 362:         method_name = "undecided"
 363: 
 364:     decision["final_recommendation"] = ""
 365:     decision["recommended_method_name"] = ""
 366:     if len(decision) > 0:
 367:         decision.loc[0, "final_recommendation"] = recommendation
 368:         decision.loc[0, "recommended_method_name"] = method_name
 369: 
 370:     return decision
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 373–462 行**

```python
 373: def write_report(
 374:     per_config: pd.DataFrame,
 375:     per_category: pd.DataFrame,
 376:     primary: pd.DataFrame,
 377:     decision: pd.DataFrame,
 378: ) -> None:
 379:     lines = []
 380:     lines += [
 381:         "# Stage 16-B Adaptive QCR-U Paper-facing Comparison",
 382:         "",
 383:         "## 1. Purpose",
 384:         "",
 385:         "This stage connects the Adaptive QCR-U candidate back to a paper-facing comparison table.",
 386:         "",
 387:         "It tests whether Adaptive QCR-U should be the final method name, or whether the method should be downgraded to quality-calibrated localization-guided fusion.",
 388:         "",
 389:         "## 2. Primary Protocol",
 390:         "",
 391:         "The primary protocol is:",
 392:         "",
 393:         "```text",
 394:         "dataset = VisA",
 395:         "strategy = inspection_binary",
 396:         "eval_mode = crop_topk_ensemble",
 397:         "```",
 398:         "",
 399:         "Reason: QCR-U is a candidate/crop reliability method. `full_all` is useful for diagnostics but is not the correct primary protocol for a crop-based reliability module.",
 400:         "",
 401:         "## 3. Primary Protocol Table",
 402:         "",
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
 415:         "## 4. Decision Summary",
 416:         "",
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
 429:         rec = decision.iloc[0]["final_recommendation"]
 430:         name = decision.iloc[0]["recommended_method_name"]
 431:     else:
 432:         rec = "insufficient evidence"
 433:         name = "undecided"
 434: 
 435:     lines += [
 436:         "",
 437:         "## 5. Final Trial Recommendation",
 438:         "",
 439:         f"- recommended method name: `{name}`",
 440:         f"- recommendation: {rec}",
 441:         "",
 442:         "## 6. Interpretation Rule",
 443:         "",
 444:         "If Adaptive QCR-U only improves over quality-only by a negligible margin, the paper should not overclaim adaptive consistency.",
 445:         "",
 446:         "In that case, the correct claim is:",
 447:         "",
 448:         "```text",
 449:         "Candidate quality provides the main reliability calibration gain, while adaptive consistency is a conservative refinement that avoids fixed-consistency degradation.",
 450:         "```",
 451:         "",
 452:         "## 7. Outputs",
... 函数剩余 10 行已省略 ...
```

**函数 `main`：`experiments/stage16_qcru_ablation/build_stage16_c_final_method_claims.py`，第 98–355 行**

```python
  98: def main() -> None:
  99:     OUT_DIR.mkdir(parents=True, exist_ok=True)
 100:     DOC_DIR.mkdir(parents=True, exist_ok=True)
 101: 
 102:     primary = read_csv_strict(IN_PRIMARY)
 103:     decision = read_csv_strict(IN_DECISION)
 104: 
 105:     # Primary-protocol deltas.
 106:     d_v4_v3 = get_primary_delta(primary, "V4", "V3")
 107:     d_v6_v3 = get_primary_delta(primary, "V6", "V3")
 108:     d_v6_v4 = get_primary_delta(primary, "V6", "V4")
 109:     d_v5_v4 = get_primary_delta(primary, "V5", "V4")
 110:     d_v6_v5 = get_primary_delta(primary, "V6", "V5")
 111: 
 112:     # All-protocol summaries from Stage 16-B decision file.
 113:     all_v4_v3 = lookup_decision(decision, "all_protocols", "quality_minus_naive")
 114:     all_v6_v3 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_naive")
 115:     all_v6_v4 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_quality")
 116:     all_v6_v5 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_fixed_qc")
 117: 
 118:     # Recommendation from Stage 16-B.
 119:     recommended_name = ""
 120:     final_recommendation = ""
 121:     if "recommended_method_name" in decision.columns:
 122:         vals = [v for v in decision["recommended_method_name"].dropna().astype(str).tolist() if v.strip()]
 123:         if vals:
 124:             recommended_name = vals[0]
 125:     if "final_recommendation" in decision.columns:
 126:         vals = [v for v in decision["final_recommendation"].dropna().astype(str).tolist() if v.strip()]
 127:         if vals:
 128:             final_recommendation = vals[0]
 129: 
 130:     if not recommended_name:
 131:         if d_v6_v4["mean_delta"] >= 0.005:
 132:             recommended_name = "Adaptive QCR-U"
 133:         elif d_v6_v4["mean_delta"] > 0:
 134:             recommended_name = "Quality-Calibrated QCR with adaptive consistency refinement"
 135:         else:
 136:             recommended_name = "Quality-Calibrated Localization-Guided Fusion"
 137: 
 138:     if not final_recommendation:
 139:         final_recommendation = (
 140:             "Use the quality-calibrated method as the main paper-facing method; "
 141:             "treat adaptive consistency as a conservative refinement."
 142:         )
 143: 
 144:     rows = [
 145:         {
 146:             "claim_id": "C1",
 147:             "claim_type": "final_method_name",
 148:             "claim": "Use Quality-Calibrated QCR as the main paper-facing method family.",
 149:             "evidence": (
 150:                 f"Stage 16-B recommends `{recommended_name}`. "
 151:                 f"Primary adaptive-minus-quality mean delta is {fmt(d_v6_v4['mean_delta'])} AUROC."
 152:             ),
 153:             "paper_status": "use",
 154:         },
 155:         {
 156:             "claim_id": "C2",
 157:             "claim_type": "main_effective_component",
 158:             "claim": "Candidate quality calibration is the main effective component.",
 159:             "evidence": (
 160:                 f"Primary quality-minus-naive mean delta is {fmt(d_v4_v3['mean_delta'])} AUROC; "
 161:                 f"all-protocol quality-minus-naive mean delta is {fmt(all_v4_v3['mean_delta'])} AUROC."
 162:             ),
 163:             "paper_status": "use",
 164:         },
 165:         {
 166:             "claim_id": "C3",
 167:             "claim_type": "auxiliary_component",
 168:             "claim": "Adaptive consistency is a conservative refinement, not the main source of improvement.",
 169:             "evidence": (
 170:                 f"Primary adaptive-minus-quality mean delta is only {fmt(d_v6_v4['mean_delta'])} AUROC; "
 171:                 f"all-protocol adaptive-minus-quality mean delta is {fmt(all_v6_v4['mean_delta'])} AUROC."
 172:             ),
 173:             "paper_status": "use_with_caution",
 174:         },
 175:         {
 176:             "claim_id": "C4",
 177:             "claim_type": "rejected_claim",
... 函数剩余 178 行已省略 ...
```

**函数 `rename_qcr_variant`：`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 115–128 行**

```python
 115: def rename_qcr_variant(variant_id: str, variant: str) -> tuple[str, str, bool]:
 116:     if variant_id == "V0":
 117:         return "Detector only", "anchor_baseline", True
 118:     if variant_id == "V2":
 119:         return "Crop VLM only", "vlm_crop_baseline", True
 120:     if variant_id == "V3":
 121:         return "Naive detector-crop fusion", "naive_fusion_baseline", True
 122:     if variant_id == "V4":
 123:         return "Quality-Calibrated QCR", "main_effective_method_core", True
 124:     if variant_id == "V5":
 125:         return "Fixed Q+C fusion", "diagnostic_not_final", False
 126:     if variant_id == "V6":
 127:         return "Quality-Calibrated QCR + adaptive consistency refinement", "final_refinement_variant", True
 128:     return variant, "other", True
```

**函数 `interpret_qcr_delta`：`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 278–291 行**

```python
 278: def interpret_qcr_delta(name: str, delta: float) -> str:
 279:     if "Quality-Calibrated QCR vs naive" in name:
 280:         return "Candidate quality calibration is the main method gain."
 281:     if "Adaptive refinement vs Quality" in name:
 282:         if abs(delta) < 0.005:
 283:             return "Adaptive consistency is only a small refinement, not a main contribution."
 284:         return "Adaptive consistency provides a meaningful refinement."
 285:     if "Adaptive refinement vs naive" in name:
 286:         return "Final refinement variant improves over naive fusion."
 287:     if "Fixed Q+C" in name:
 288:         return "Fixed consistency is diagnostic only because robustness is not stable across protocols."
 289:     if "Adaptive refinement vs fixed" in name:
 290:         return "Adaptive refinement trades peak primary-protocol AUROC for robustness."
 291:     return "QCR delta."
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 294–444 行**

```python
 294: def write_report(
 295:     system_table: pd.DataFrame,
 296:     qcr_table: pd.DataFrame,
 297:     deltas: pd.DataFrame,
 298:     claims: pd.DataFrame,
 299: ) -> None:
 300:     lines = []
 301:     lines += [
 302:         "# Stage 16-D Paper-facing Final Comparison",
 303:         "",
 304:         "## 1. Purpose",
 305:         "",
 306:         "This stage creates the final paper-facing comparison tables after the method claim was locked in Stage 16-C.",
 307:         "",
 308:         "The final method family is:",
 309:         "",
 310:         "```text",
 311:         "Quality-Calibrated QCR",
 312:         "```",
 313:         "",
 314:         "The adaptive consistency term is treated only as a conservative refinement, not as the main performance source.",
 315:         "",
 316:         "## 2. Important Comparison Rule",
 317:         "",
 318:         "This report uses two panels because Stage 15 system baselines and Stage 16 QCR ablations are not the same protocol.",
 319:         "",
 320:         "- Panel A compares system-level baselines from Stage 15.",
 321:         "- Panel B compares QCR variants under the Stage 16-B QCR primary protocol.",
 322:         "",
 323:         "Do not merge the two panels into a single global ranking.",
 324:         "",
 325:         "## 3. Panel A: System-level Strong Baseline Comparison",
 326:         "",
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
 339:         "Paper use:",
 340:         "",
 341:         "- Use `PatchCore + context VLM, LOCO` as the fair system-level result.",
 342:         "- Use `same-set` only as an upper-bound diagnostic.",
 343:         "- Keep `EfficientAD-30` explicitly labeled as fixed-budget.",
 344:         "",
 345:         "## 4. Panel B: QCR Primary-protocol Ablation",
 346:         "",
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
 359:         "",
 360:         "Paper use:",
 361:         "",
 362:         "- Treat `Quality-Calibrated QCR` as the main effective method core.",
 363:         "- Treat `Quality-Calibrated QCR + adaptive consistency refinement` as the final conservative refinement.",
 364:         "- Treat `Fixed Q+C fusion` as diagnostic only, because it is not robust across protocols.",
 365:         "",
 366:         "## 5. Claim-ready Deltas",
 367:         "",
 368:         "| Scope | Comparison | Left Score | Right Score | Delta | Interpretation |",
 369:         "|---|---|---:|---:|---:|---|",
 370:     ]
 371: 
 372:     for _, r in deltas.iterrows():
 373:         left = r["left_score"]
... 函数剩余 71 行已省略 ...
```

**函数 `build_base_table`：`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 130–204 行**

```python
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
 141:         "defect_type",
 142:     ]
 143: 
 144:     base_cols = REQUIRED_COLUMNS + [c for c in optional_cols if c in df.columns]
 145:     base = df[base_cols].copy()
 146: 
 147:     base = base.drop_duplicates(
 148:         subset=["backbone", "dataset", "category", "strategy", "eval_mode", "image_key"]
 149:     ).reset_index(drop=True)
 150: 
 151:     for c in [
 152:         "is_anomaly_final",
 153:         "vlm_score_norm",
 154:         "detector_score_norm",
 155:         "candidate_quality_norm",
 156:         "high_high_consistency",
 157:         "num_candidates",
 158:     ]:
 159:         if c in base.columns:
 160:             base[c] = pd.to_numeric(base[c], errors="coerce")
 161: 
 162:     base["D"] = base["detector_score_norm"].fillna(0.0)
 163:     base["M"] = base["vlm_score_norm"].fillna(0.0)
 164:     base["Q"] = base["candidate_quality_norm"].fillna(0.0)
 165:     base["K"] = base["high_high_consistency"].fillna(0.0)
 166: 
 167:     base["score_naive"] = 0.5 * base["D"] + 0.5 * base["M"]
 168: 
 169:     base["score_quality_raw"] = (
 170:         0.5 * base["D"]
 171:         + 0.5 * (base["M"] * (0.5 + 0.5 * base["Q"]))
 172:     )
 173: 
 174:     base["score_fixed_qc_raw"] = (
 175:         0.40 * base["D"]
 176:         + 0.40 * base["M"]
 177:         + 0.10 * base["Q"]
 178:         + 0.10 * base["K"]
 179:     )
 180: 
 181:     agreement = (1.0 - (base["D"] - base["M"]).abs()).clip(lower=0.0, upper=1.0)
 182:     mutual_anomaly_evidence = np.minimum(base["D"], base["M"])
 183:     adaptive_gate = base["Q"] * base["K"] * agreement * mutual_anomaly_evidence
 184: 
 185:     base["agreement"] = agreement
 186:     base["mutual_anomaly_evidence"] = mutual_anomaly_evidence
 187:     base["adaptive_gate"] = adaptive_gate
 188:     base["score_adaptive_raw"] = base["score_quality_raw"] + 0.05 * adaptive_gate
 189: 
 190:     group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
 191:     for raw_col, out_col in [
 192:         ("score_quality_raw", "score_quality"),
 193:         ("score_fixed_qc_raw", "score_fixed_qc"),
 194:         ("score_adaptive_raw", "score_adaptive"),
 195:     ]:
 196:         base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)
 197: 
 198:     base["delta_quality_minus_naive"] = base["score_quality"] - base["score_naive"]
 199:     base["delta_fixed_minus_quality"] = base["score_fixed_qc"] - base["score_quality"]
 200:     base["delta_adaptive_minus_quality"] = base["score_adaptive"] - base["score_quality"]
 201:     base["delta_adaptive_minus_fixed"] = base["score_adaptive"] - base["score_fixed_qc"]
 202:     base["detector_vlm_disagreement"] = (base["D"] - base["M"]).abs()
 203: 
 204:     return base
```

**函数 `select_top_cases`：`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 218–251 行**

```python
 218: def select_top_cases(g: pd.DataFrame, case_type: str, sort_col: str, ascending: bool, n: int = 5) -> pd.DataFrame:
 219:     cols = [
 220:         "backbone",
 221:         "dataset",
 222:         "strategy",
 223:         "eval_mode",
 224:         "category",
 225:         "image_key",
 226:         "is_anomaly_final",
 227:         "D",
 228:         "M",
 229:         "Q",
 230:         "K",
 231:         "agreement",
 232:         "mutual_anomaly_evidence",
 233:         "adaptive_gate",
 234:         "score_naive",
 235:         "score_quality",
 236:         "score_fixed_qc",
 237:         "score_adaptive",
 238:         "delta_quality_minus_naive",
 239:         "delta_fixed_minus_quality",
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

**函数 `build_case_inventory`：`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 254–334 行**

```python
 254: def build_case_inventory(primary: pd.DataFrame) -> pd.DataFrame:
 255:     rows = []
 256: 
 257:     for keys, g in primary.groupby(["backbone", "dataset", "strategy", "eval_mode"], dropna=False):
 258:         anomaly = g[g["is_anomaly_final"] == 1].copy()
 259:         normal = g[g["is_anomaly_final"] == 0].copy()
 260: 
 261:         if not anomaly.empty:
 262:             rows.append(
 263:                 select_top_cases(
 264:                     anomaly,
 265:                     "quality_helps_anomaly_boost",
 266:                     "delta_quality_minus_naive",
 267:                     ascending=False,
 268:                 )
 269:             )
 270:             rows.append(
 271:                 select_top_cases(
 272:                     anomaly,
 273:                     "quality_boundary_anomaly_suppression",
 274:                     "delta_quality_minus_naive",
 275:                     ascending=True,
 276:                 )
 277:             )
 278:             rows.append(
 279:                 select_top_cases(
 280:                     anomaly,
 281:                     "fixed_consistency_boundary_anomaly_suppression",
 282:                     "delta_fixed_minus_quality",
 283:                     ascending=True,
 284:                 )
 285:             )
 286: 
 287:         if not normal.empty:
 288:             rows.append(
 289:                 select_top_cases(
 290:                     normal,
 291:                     "quality_helps_normal_suppression",
 292:                     "delta_quality_minus_naive",
 293:                     ascending=True,
 294:                 )
 295:             )
 296:             rows.append(
 297:                 select_top_cases(
 298:                     normal,
 299:                     "quality_boundary_normal_boost",
 300:                     "delta_quality_minus_naive",
 301:                     ascending=False,
 302:                 )
 303:             )
 304:             rows.append(
 305:                 select_top_cases(
 306:                     normal,
 307:                     "fixed_consistency_boundary_normal_boost",
 308:                     "delta_fixed_minus_quality",
 309:                     ascending=False,
 310:                 )
 311:             )
 312: 
 313:         rows.append(
 314:             select_top_cases(
 315:                 g,
 316:                 "adaptive_refinement_high_gate",
 317:                 "adaptive_gate",
 318:                 ascending=False,
 319:             )
 320:         )
 321:         rows.append(
 322:             select_top_cases(
 323:                 g,
 324:                 "detector_vlm_disagreement_boundary",
 325:                 "detector_vlm_disagreement",
 326:                 ascending=False,
 327:             )
 328:         )
 329: 
 330:     if not rows:
 331:         return pd.DataFrame()
 332: 
 333:     out = pd.concat(rows, ignore_index=True)
... 函数剩余 1 行已省略 ...
```

**函数 `build_category_summary`：`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 337–396 行**

```python
 337: def build_category_summary(primary: pd.DataFrame) -> pd.DataFrame:
 338:     variants = [
 339:         ("V3", "naive_detector_crop_fusion", "score_naive"),
 340:         ("V4", "Quality-Calibrated QCR", "score_quality"),
 341:         ("V5", "Fixed Q+C fusion", "score_fixed_qc"),
 342:         ("V6", "Quality-Calibrated QCR + adaptive consistency refinement", "score_adaptive"),
 343:     ]
 344: 
 345:     rows = []
 346:     group_cols = ["backbone", "dataset", "strategy", "eval_mode", "category"]
 347: 
 348:     for keys, g in primary.groupby(group_cols, dropna=False):
 349:         base_row = dict(zip(group_cols, keys))
 350: 
 351:         for variant_id, method, score_col in variants:
 352:             m = eval_binary(g["is_anomaly_final"], g[score_col])
 353:             row = base_row.copy()
 354:             row.update(
 355:                 {
 356:                     "variant_id": variant_id,
 357:                     "method": method,
 358:                     "score_col": score_col,
 359:                     **m,
 360:                 }
 361:             )
 362:             rows.append(row)
 363: 
 364:     long = pd.DataFrame(rows)
 365: 
 366:     idx = group_cols
 367:     piv = long.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
 368:     piv.columns.name = None
 369: 
 370:     for col in ["V3", "V4", "V5", "V6"]:
 371:         if col not in piv.columns:
 372:             piv[col] = np.nan
 373: 
 374:     piv["delta_v4_quality_minus_v3_naive"] = piv["V4"] - piv["V3"]
 375:     piv["delta_v6_adaptive_minus_v4_quality"] = piv["V6"] - piv["V4"]
 376:     piv["delta_v5_fixed_minus_v4_quality"] = piv["V5"] - piv["V4"]
 377:     piv["delta_v6_adaptive_minus_v5_fixed"] = piv["V6"] - piv["V5"]
 378: 
 379:     def boundary_label(r):
 380:         labels = []
 381:         if pd.notna(r["delta_v4_quality_minus_v3_naive"]) and r["delta_v4_quality_minus_v3_naive"] <= 0:
 382:             labels.append("quality_not_helpful")
 383:         if pd.notna(r["delta_v6_adaptive_minus_v4_quality"]) and abs(r["delta_v6_adaptive_minus_v4_quality"]) < 0.001:
 384:             labels.append("adaptive_gain_negligible")
 385:         if pd.notna(r["delta_v5_fixed_minus_v4_quality"]) and r["delta_v5_fixed_minus_v4_quality"] > 0:
 386:             labels.append("fixed_consistency_can_peak_but_diagnostic")
 387:         if pd.notna(r["V6"]) and r["V6"] < 0.90:
 388:             labels.append("low_absolute_qcr_auc")
 389:         return ";".join(labels) if labels else "no_major_boundary"
 390: 
 391:     piv["boundary_label"] = piv.apply(boundary_label, axis=1)
 392: 
 393:     return piv.sort_values(
 394:         ["backbone", "delta_v4_quality_minus_v3_naive", "delta_v6_adaptive_minus_v4_quality"],
 395:         ascending=[True, True, True],
 396:     ).reset_index(drop=True)
```

**函数 `build_decision_summary`：`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 399–459 行**

```python
 399: def build_decision_summary(category_summary: pd.DataFrame, case_inventory: pd.DataFrame) -> pd.DataFrame:
 400:     rows = []
 401: 
 402:     def add(decision_id, topic, decision, evidence, paper_action):
 403:         rows.append(
 404:             {
 405:                 "decision_id": decision_id,
 406:                 "topic": topic,
 407:                 "decision": decision,
 408:                 "evidence": evidence,
 409:                 "paper_action": paper_action,
 410:             }
 411:         )
 412: 
 413:     q_delta = category_summary["delta_v4_quality_minus_v3_naive"].dropna()
 414:     a_delta = category_summary["delta_v6_adaptive_minus_v4_quality"].dropna()
 415:     f_delta = category_summary["delta_v5_fixed_minus_v4_quality"].dropna()
 416: 
 417:     add(
 418:         "E1",
 419:         "quality_calibration",
 420:         "Keep candidate quality calibration as the main method core.",
 421:         f"Per-category mean V4-V3 AUROC delta={q_delta.mean():+.4f}; wins={(q_delta > 0).sum()}/{len(q_delta)}.",
 422:         "Use as main contribution.",
 423:     )
 424: 
 425:     add(
 426:         "E2",
 427:         "adaptive_consistency",
 428:         "Keep adaptive consistency only as a refinement.",
 429:         f"Per-category mean V6-V4 AUROC delta={a_delta.mean():+.4f}; wins={(a_delta > 0).sum()}/{len(a_delta)}.",
 430:         "Use with caution; do not call it the main source of improvement.",
 431:     )
 432: 
 433:     add(
 434:         "E3",
 435:         "fixed_consistency",
 436:         "Do not use fixed Q+C as the final method even if it peaks on some categories.",
 437:         f"Per-category mean V5-V4 AUROC delta={f_delta.mean():+.4f}; positive cases={(f_delta > 0).sum()}/{len(f_delta)}.",
 438:         "Mention as diagnostic only.",
 439:     )
 440: 
 441:     if not case_inventory.empty:
 442:         counts = case_inventory["case_type"].value_counts().to_dict()
 443:         add(
 444:             "E4",
 445:             "case_inventory",
 446:             "Use selected cases for qualitative boundary analysis.",
 447:             "; ".join([f"{k}={v}" for k, v in counts.items()]),
 448:             "Inspect representative cases manually before paper figures.",
 449:         )
 450: 
 451:     add(
 452:         "E5",
 453:         "paper_boundary",
 454:         "The method should be claimed as reliability calibration, not full anomaly understanding.",
 455:         "The case taxonomy explicitly includes detector-VLM disagreement and candidate-quality boundary cases.",
 456:         "Use boundary-aware wording in paper.",
 457:     )
 458: 
 459:     return pd.DataFrame(rows)
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 462–580 行**

```python
 462: def write_report(category_summary: pd.DataFrame, case_inventory: pd.DataFrame, decision: pd.DataFrame) -> None:
 463:     lines = []
 464:     lines += [
 465:         "# Stage 16-E Failure Cases and Boundary Analysis",
 466:         "",
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
 479:         "strategy = inspection_binary",
 480:         "eval_mode = crop_topk_ensemble",
 481:         "```",
 482:         "",
 483:         "## 3. Category-level Boundary Summary",
 484:         "",
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
 497:         )
 498: 
 499:     lines += [
 500:         "",
 501:         "## 4. Case Types Extracted",
 502:         "",
 503:         "| Case Type | Meaning | Paper Use |",
 504:         "|---|---|---|",
 505:         "| quality_helps_anomaly_boost | anomaly images whose score is boosted by quality calibration | positive qualitative example |",
 506:         "| quality_helps_normal_suppression | normal images suppressed by quality calibration | false-positive reduction example |",
 507:         "| quality_boundary_anomaly_suppression | anomaly images suppressed by quality calibration | boundary / failure case |",
 508:         "| quality_boundary_normal_boost | normal images boosted by quality calibration | boundary / failure case |",
 509:         "| fixed_consistency_boundary_anomaly_suppression | anomaly images where fixed consistency hurts | explains why fixed Q+C is not final |",
 510:         "| fixed_consistency_boundary_normal_boost | normal images where fixed consistency increases risk | explains false-positive boundary |",
 511:         "| adaptive_refinement_high_gate | images with strongest adaptive gate | explains refinement behavior |",
 512:         "| detector_vlm_disagreement_boundary | images with high detector/VLM disagreement | explains detector-VLM conflict |",
 513:         "",
 514:     ]
 515: 
 516:     if case_inventory.empty:
 517:         lines.append("No case inventory generated.")
 518:     else:
 519:         counts = case_inventory["case_type"].value_counts().reset_index()
 520:         counts.columns = ["case_type", "count"]
 521:         lines += [
 522:             "Case counts:",
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
... 函数剩余 39 行已省略 ...
```

**函数 `make_claim_map`：`experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py`，第 91–265 行**

```python
  91: def make_claim_map(system: pd.DataFrame, deltas: pd.DataFrame, boundary: pd.DataFrame, category: pd.DataFrame) -> pd.DataFrame:
  92:     loco = score_of(system, "PatchCore + context VLM, LOCO")
  93:     same = score_of(system, "PatchCore + context VLM, same-set")
  94:     patchcore = score_of(system, "PatchCore")
  95:     ead30 = score_of(system, "EfficientAD-30 fixed-budget")
  96:     winclip = score_of(system, "WinCLIP fixed protocol")
  97:     full_vlm = score_of(system, "full-image VLM")
  98:     context_vlm = score_of(system, "context-aware VLM")
  99: 
 100:     d_loco_patch = delta_row(deltas, "LOCO fusion vs PatchCore")
 101:     d_loco_ead = delta_row(deltas, "LOCO fusion vs EfficientAD-30")
 102:     d_loco_winclip = delta_row(deltas, "LOCO fusion vs WinCLIP")
 103:     d_context_full = delta_row(deltas, "context-aware VLM vs full-image VLM")
 104:     d_quality_naive = delta_row(deltas, "Quality-Calibrated QCR vs naive fusion")
 105:     d_adaptive_quality = delta_row(deltas, "Adaptive refinement vs Quality-Calibrated QCR")
 106:     d_adaptive_naive = delta_row(deltas, "Adaptive refinement vs naive fusion")
 107:     d_fixed_quality = delta_row(deltas, "Fixed Q+C vs Quality-Calibrated QCR")
 108: 
 109:     e_quality = boundary_decision(boundary, "E1")
 110:     e_adaptive = boundary_decision(boundary, "E2")
 111:     e_fixed = boundary_decision(boundary, "E3")
 112:     e_cases = boundary_decision(boundary, "E4")
 113:     e_boundary = boundary_decision(boundary, "E5")
 114: 
 115:     cat_stats = compute_category_stats(category)
 116: 
 117:     rows = [
 118:         {
 119:             "claim_id": "P1",
 120:             "claim_category": "problem_framing",
 121:             "paper_claim": "Industrial anomaly VLM reasoning should be localization-guided rather than full-image only.",
 122:             "allowed_wording": "We study localization-guided VLM anomaly recognition, where detector localization evidence is converted into candidate-level visual-language evidence.",
 123:             "forbidden_wording": "We solve full industrial anomaly understanding with a general-purpose VLM.",
 124:             "evidence_files": "stage16_d_paper_facing_system_baseline_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
 125:             "evidence_summary": (
 126:                 f"context-aware VLM AUROC={context_vlm}; full-image VLM AUROC={full_vlm}; "
 127:                 f"context minus full-image delta={fmt(d_context_full.get('delta', None))}."
 128:             ),
 129:             "support_level": "moderate",
 130:             "paper_section": "Introduction; Method motivation; Experiments",
 131:             "caveat": "Do not claim semantic understanding or manufacturing-cause reasoning.",
 132:             "status": "use",
 133:         },
 134:         {
 135:             "claim_id": "P2",
 136:             "claim_category": "system_level_result",
 137:             "paper_claim": "Localization-guided VLM evidence is complementary to detector baselines.",
 138:             "allowed_wording": "The fair LOCO fusion improves over the detector-only PatchCore baseline and the fixed-budget EfficientAD baseline.",
 139:             "forbidden_wording": "The method fully beats all detector baselines under all budgets.",
 140:             "evidence_files": "stage16_d_paper_facing_system_baseline_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
 141:             "evidence_summary": (
 142:                 f"LOCO AUROC={loco}; PatchCore AUROC={patchcore}; EfficientAD-30 AUROC={ead30}; "
 143:                 f"LOCO-PatchCore={fmt(d_loco_patch.get('delta', None))}; "
 144:                 f"LOCO-EfficientAD30={fmt(d_loco_ead.get('delta', None))}."
 145:             ),
 146:             "support_level": "strong_but_protocol_limited",
 147:             "paper_section": "Main Results",
 148:             "caveat": "EfficientAD is fixed-budget; same-set fusion is upper-bound only.",
 149:             "status": "use",
 150:         },
 151:         {
 152:             "claim_id": "P3",
 153:             "claim_category": "external_baseline",
 154:             "paper_claim": "The proposed localization-guided route is stronger than the fixed WinCLIP protocol used in this study.",
 155:             "allowed_wording": "Under our fixed protocol, LOCO fusion outperforms WinCLIP.",
 156:             "forbidden_wording": "We comprehensively outperform all CLIP-based anomaly detection methods.",
 157:             "evidence_files": "stage16_d_paper_facing_system_baseline_table.csv; stage16_d_paper_facing_claim_ready_deltas.csv",
 158:             "evidence_summary": (
 159:                 f"LOCO AUROC={loco}; WinCLIP AUROC={winclip}; "
 160:                 f"delta={fmt(d_loco_winclip.get('delta', None))}."
 161:             ),
 162:             "support_level": "moderate",
 163:             "paper_section": "Baselines",
 164:             "caveat": "AnomalyCLIP is not yet included; avoid broad CLIP-family claims.",
 165:             "status": "use_with_caution",
 166:         },
 167:         {
 168:             "claim_id": "P4",
 169:             "claim_category": "main_method_component",
 170:             "paper_claim": "Candidate quality calibration is the main effective method component.",
... 函数剩余 95 行已省略 ...
```

**函数 `make_status_table`：`experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py`，第 268–310 行**

```python
 268: def make_status_table(claim_map: pd.DataFrame) -> pd.DataFrame:
 269:     rows = []
 270: 
 271:     groups = [
 272:         ("main_claims_ready", claim_map[claim_map["status"].isin(["use", "use_with_caution"])]),
 273:         ("claims_to_reject_or_downgrade", claim_map[claim_map["status"].isin(["reject", "reject_as_final_method", "use_as_diagnostic_only"])]),
 274:     ]
 275: 
 276:     for group_name, g in groups:
 277:         rows.append(
 278:             {
 279:                 "status_group": group_name,
 280:                 "num_claims": len(g),
 281:                 "claim_ids": ";".join(g["claim_id"].astype(str).tolist()),
 282:                 "summary": "; ".join(g["paper_claim"].astype(str).tolist()),
 283:             }
 284:         )
 285: 
 286:     # Paper readiness flags.
 287:     rows.extend(
 288:         [
 289:             {
 290:                 "status_group": "paper_ready_method_name",
 291:                 "num_claims": 1,
 292:                 "claim_ids": "P4;P5;P6",
 293:                 "summary": "Use Quality-Calibrated QCR as the method family; adaptive consistency is refinement; fixed Q+C is diagnostic only.",
 294:             },
 295:             {
 296:                 "status_group": "remaining_experiment_risks",
 297:                 "num_claims": 3,
 298:                 "claim_ids": "R1;R2;R3",
 299:                 "summary": "EfficientAD remains fixed-budget; AnomalyCLIP is absent; representative failure figures still need manual visual inspection.",
 300:             },
 301:             {
 302:                 "status_group": "next_actions",
 303:                 "num_claims": 2,
 304:                 "claim_ids": "N1;N2",
 305:                 "summary": "Run defensive EfficientAD-100 fruit_jelly sensitivity later; start paper outline/table-to-text drafting after claim map.",
 306:             },
 307:         ]
 308:     )
 309: 
 310:     return pd.DataFrame(rows)
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py`，第 333–460 行**

```python
 333: def write_report(claim_map: pd.DataFrame, status: pd.DataFrame, rejected: pd.DataFrame) -> None:
 334:     lines = []
 335:     lines += [
 336:         "# Stage 16-F Final Claim-Evidence Map",
 337:         "",
 338:         "## 1. Purpose",
 339:         "",
 340:         "This stage maps every paper-facing claim to concrete experimental evidence and locks the forbidden claims.",
 341:         "",
 342:         "No new model is trained and no score is tuned in this stage.",
 343:         "",
 344:         "## 2. Final Method Naming",
 345:         "",
 346:         "Use this method family name:",
 347:         "",
 348:         "```text",
 349:         "Quality-Calibrated QCR",
 350:         "```",
 351:         "",
 352:         "Use this longer descriptive phrase when needed:",
 353:         "",
 354:         "```text",
 355:         "Quality-Calibrated Localization-Guided VLM Reasoning",
 356:         "```",
 357:         "",
 358:         "Use this only as the full variant name:",
 359:         "",
 360:         "```text",
 361:         "Quality-Calibrated QCR with adaptive consistency refinement",
 362:         "```",
 363:         "",
 364:         "Do not write the method as fixed Q+C QCR-U.",
 365:         "",
 366:         "## 3. Claim-Evidence Map",
 367:         "",
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
 380:         "## 4. Evidence Details",
 381:         "",
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
... 函数剩余 48 行已省略 ...
```

**函数 `build_base_table`：`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 176–249 行**

```python
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
 187:         "strategy",
 188:         "eval_mode",
 189:         "image_key",
 190:         "is_anomaly_final",
 191:         "fallback",
 192:         "has_candidate",
 193:         "num_candidates",
 194:         "vlm_score_norm",
 195:         "detector_score_norm",
 196:         "candidate_quality_norm",
 197:         "high_high_consistency",
 198:     ]
 199:     base_cols = [c for c in base_cols if c in df.columns]
 200: 
 201:     base = df[base_cols].copy()
 202:     base = base.drop_duplicates(
 203:         subset=["backbone", "dataset", "category", "strategy", "eval_mode", "image_key"]
 204:     ).reset_index(drop=True)
 205: 
 206:     for c in [
 207:         "is_anomaly_final",
 208:         "vlm_score_norm",
 209:         "detector_score_norm",
 210:         "candidate_quality_norm",
 211:         "high_high_consistency",
 212:         "num_candidates",
 213:     ]:
 214:         if c in base.columns:
 215:             base[c] = pd.to_numeric(base[c], errors="coerce")
 216: 
 217:     base["M_crop_vlm"] = base["vlm_score_norm"]
 218:     base["D_detector"] = base["detector_score_norm"]
 219:     base["Q_quality"] = base["candidate_quality_norm"].fillna(0.0)
 220:     base["K_consistency"] = base["high_high_consistency"].fillna(0.0)
 221: 
 222:     # Fixed, non-tuned ablation formulas.
 223:     base["score_detector_only"] = base["D_detector"]
 224:     base["score_crop_topk_vlm"] = base["M_crop_vlm"]
 225:     base["score_naive_detector_crop_fusion"] = 0.5 * base["D_detector"] + 0.5 * base["M_crop_vlm"]
 226: 
 227:     # Quality should modulate whether the crop VLM signal is trusted, not replace the detector score.
 228:     base["score_quality_weighted_crop_raw"] = (
 229:         0.5 * base["D_detector"]
 230:         + 0.5 * (base["M_crop_vlm"] * (0.5 + 0.5 * base["Q_quality"]))
 231:     )
 232: 
 233:     # Consistency gets a small fixed weight. This is not tuned on test labels.
 234:     base["score_quality_consistency_fusion_raw"] = (
 235:         0.40 * base["D_detector"]
 236:         + 0.40 * base["M_crop_vlm"]
 237:         + 0.10 * base["Q_quality"]
 238:         + 0.10 * base["K_consistency"]
 239:     )
 240: 
 241:     # Normalize raw variants within each protocol group so scores are comparable for threshold metrics.
 242:     group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
 243:     for raw_col, out_col in [
 244:         ("score_quality_weighted_crop_raw", "score_quality_weighted_crop"),
 245:         ("score_quality_consistency_fusion_raw", "score_quality_consistency_fusion"),
 246:     ]:
 247:         base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)
 248: 
 249:     return base
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 326–449 行**

```python
 326: def write_report(
 327:     base: pd.DataFrame,
 328:     per_config: pd.DataFrame,
 329:     per_category: pd.DataFrame,
 330:     best_protocol: pd.DataFrame,
 331: ) -> None:
 332:     lines = []
 333:     lines += [
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
 346:         "",
 347:         "The base table contains detector score, crop VLM score, candidate quality, and detector-VLM consistency.",
 348:         "",
 349:         "## 3. Fixed Ablation Variants",
 350:         "",
 351:         "| Variant | Formula | Meaning |",
 352:         "|---|---|---|",
 353:         "| detector_only | `D` | detector score only |",
 354:         "| crop_topk_vlm | `M` | crop VLM score only |",
 355:         "| naive_detector_crop_fusion | `0.5D + 0.5M` | naive fusion baseline |",
 356:         "| quality_weighted_crop | `0.5D + 0.5(M * (0.5 + 0.5Q))` | candidate quality modulates VLM evidence |",
 357:         "| quality_consistency_fusion | `0.4D + 0.4M + 0.1Q + 0.1K` | fixed Q+C fusion variant |",
 358:         "",
 359:         "Where `D` is detector score, `M` is crop VLM abnormal score, `Q` is candidate quality, and `K` is detector-VLM high-high consistency.",
 360:         "",
 361:         "## 4. Best Protocols by Q+C Fusion AUROC",
 362:         "",
 363:         "| Rank | Backbone | Dataset | Strategy | Eval Mode | V5 AUROC | V5 AP | V5 Best F1 |",
 364:         "|---:|---|---|---|---|---:|---:|---:|",
 365:     ]
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
 378:         "",
 379:         "## 5. Variant Comparison Within the Best Protocol",
 380:         "",
 381:     ]
 382: 
 383:     if not best_protocol.empty:
 384:         best = best_protocol.iloc[0]
 385:         mask = (
 386:             (per_config["backbone"] == best["backbone"])
 387:             & (per_config["dataset"] == best["dataset"])
 388:             & (per_config["strategy"] == best["strategy"])
 389:             & (per_config["eval_mode"] == best["eval_mode"])
 390:         )
 391:         comp = per_config[mask].sort_values("variant_id")
 392: 
 393:         lines += [
 394:             f"Best protocol by V5 AUROC: `{best['backbone']} / {best['dataset']} / {best['strategy']} / {best['eval_mode']}`.",
 395:             "",
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
... 函数剩余 44 行已省略 ...
```

**函数 `build_base_table`：`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 155–231 行**

```python
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
 166:         "image_key",
 167:         "is_anomaly_final",
 168:         "fallback",
 169:         "has_candidate",
 170:         "num_candidates",
 171:         "vlm_score_norm",
 172:         "detector_score_norm",
 173:         "candidate_quality_norm",
 174:         "high_high_consistency",
 175:     ]
 176:     base_cols = [c for c in base_cols if c in df.columns]
 177: 
 178:     base = df[base_cols].copy()
 179:     base = base.drop_duplicates(
 180:         subset=["backbone", "dataset", "category", "strategy", "eval_mode", "image_key"]
 181:     ).reset_index(drop=True)
 182: 
 183:     for c in [
 184:         "is_anomaly_final",
 185:         "vlm_score_norm",
 186:         "detector_score_norm",
 187:         "candidate_quality_norm",
 188:         "high_high_consistency",
 189:         "num_candidates",
 190:     ]:
 191:         if c in base.columns:
 192:             base[c] = pd.to_numeric(base[c], errors="coerce")
 193: 
 194:     base["D"] = base["detector_score_norm"].fillna(0.0)
 195:     base["M"] = base["vlm_score_norm"].fillna(0.0)
 196:     base["Q"] = base["candidate_quality_norm"].fillna(0.0)
 197:     base["K"] = base["high_high_consistency"].fillna(0.0)
 198: 
 199:     # Existing baselines.
 200:     base["score_naive"] = 0.5 * base["D"] + 0.5 * base["M"]
 201:     base["score_quality_raw"] = 0.5 * base["D"] + 0.5 * (base["M"] * (0.5 + 0.5 * base["Q"]))
 202:     base["score_fixed_qc_raw"] = 0.40 * base["D"] + 0.40 * base["M"] + 0.10 * base["Q"] + 0.10 * base["K"]
 203: 
 204:     # Adaptive QCR-U:
 205:     # Start from quality-weighted core.
 206:     # Add a conservative consistency bonus only when:
 207:     # - candidate quality is high,
 208:     # - detector and VLM agree,
 209:     # - both detector and VLM provide high anomaly evidence.
 210:     #
 211:     # This is label-free and intentionally conservative.
 212:     agreement = 1.0 - (base["D"] - base["M"]).abs()
 213:     agreement = agreement.clip(lower=0.0, upper=1.0)
 214: 
 215:     mutual_anomaly_evidence = np.minimum(base["D"], base["M"])
 216:     adaptive_gate = base["Q"] * base["K"] * agreement * mutual_anomaly_evidence
 217: 
 218:     base["adaptive_gate"] = adaptive_gate
 219: 
 220:     # Small fixed coefficient. This is not selected from test labels.
 221:     base["score_adaptive_qcru_raw"] = base["score_quality_raw"] + 0.05 * adaptive_gate
 222: 
 223:     group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
 224:     for raw_col, out_col in [
 225:         ("score_quality_raw", "score_quality"),
 226:         ("score_fixed_qc_raw", "score_fixed_qc"),
 227:         ("score_adaptive_qcru_raw", "score_adaptive_qcru"),
 228:     ]:
 229:         base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)
 230: 
 231:     return base
```

**函数 `write_report`：`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 351–444 行**

```python
 351: def write_report(per_config: pd.DataFrame, delta: pd.DataFrame, summary: pd.DataFrame) -> None:
 352:     best = per_config[per_config["variant_id"] == "V6"].sort_values("auroc", ascending=False).reset_index(drop=True)
 353:     best["rank_by_v6_auroc"] = range(1, len(best) + 1)
 354: 
 355:     lines = []
 356:     lines += [
 357:         "# Stage 16-A3 Adaptive QCR-U",
 358:         "",
 359:         "## 1. Purpose",
 360:         "",
 361:         "Stage 16-A2 showed that candidate quality is stable, while fixed consistency is not universally beneficial.",
 362:         "",
 363:         "This stage tests an adaptive QCR-U score that uses quality-weighted crop scoring as the stable core and applies consistency only as a conservative reliability-gated bonus.",
 364:         "",
 365:         "## 2. Formula",
 366:         "",
 367:         "```text",
 368:         "D = detector anomaly score",
 369:         "M = crop VLM anomaly score",
 370:         "Q = candidate quality",
 371:         "K = high-high detector/VLM consistency",
 372:         "",
 373:         "S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)",
 374:         "agreement = 1 - |D - M|",
 375:         "mutual_anomaly_evidence = min(D, M)",
 376:         "gate = Q * K * agreement * mutual_anomaly_evidence",
 377:         "S_adaptive = S_quality + 0.05 * gate",
 378:         "```",
 379:         "",
 380:         "The coefficient `0.05` is fixed and intentionally conservative. It is not selected by test-set tuning.",
 381:         "",
 382:         "## 3. Robustness Summary",
 383:         "",
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
 396:         "",
 397:         "## 4. Adaptive QCR-U Protocol Ranking",
 398:         "",
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
 411:         "## 5. Protocol-level Delta Table",
 412:         "",
 413:         "| Backbone | Strategy | Eval Mode | V3 | V4 | V5 | V6 | V6-V3 | V6-V4 | V6-V5 |",
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
 426: 
 427:     lines += [
 428:         "",
 429:         "## 6. Decision Rule",
 430:         "",
... 函数剩余 14 行已省略 ...
```

**函数 `scan_file`：`experiments/stage18_ad2_qcr_ablation/build_stage18_b0_ad2_qcr_source_inventory.py`，第 170–270 行**

```python
 170: def scan_file(path: Path) -> dict:
 171:     df = read_table_sample(path)
 172: 
 173:     base = {
 174:         "file": str(path.relative_to(ROOT)),
 175:         "suffix": path.suffix.lower(),
 176:         "readable": False,
 177:         "num_sample_rows": None,
 178:         "num_cols": None,
 179:         "category_col": "",
 180:         "ad2_categories_found": "",
 181:         "ad2_coverage_count": 0,
 182:         "image_id_col": "",
 183:         "label_col": "",
 184:         "detector_score_col": "",
 185:         "vlm_score_col": "",
 186:         "quality_col": "",
 187:         "consistency_col": "",
 188:         "has_image_id": False,
 189:         "has_label": False,
 190:         "has_detector_score": False,
 191:         "has_vlm_score": False,
 192:         "has_quality": False,
 193:         "has_consistency": False,
 194:         "source_role": "unreadable_or_irrelevant",
 195:         "qcr_assembly_value": "none",
 196:         "notes": "",
 197:     }
 198: 
 199:     if df is None or df.empty or len(df.columns) <= 1:
 200:         return base
 201: 
 202:     cols = set(df.columns)
 203: 
 204:     category_col, ad2_found = detect_categories(df, path)
 205: 
 206:     image_id_col = find_first_existing(cols, IMAGE_ID_COLS)
 207:     label_col = find_first_existing(cols, LABEL_COLS)
 208:     detector_col = find_first_existing(cols, DETECTOR_SCORE_COLS)
 209:     vlm_col = find_first_existing(cols, VLM_SCORE_COLS)
 210:     quality_col = find_first_existing(cols, QUALITY_COLS)
 211:     consistency_col = find_first_existing(cols, CONSISTENCY_COLS)
 212: 
 213:     has_image = bool(image_id_col)
 214:     has_label = bool(label_col)
 215:     has_detector = bool(detector_col)
 216:     has_vlm = bool(vlm_col)
 217:     has_quality = bool(quality_col)
 218:     has_consistency = bool(consistency_col)
 219: 
 220:     role_parts = []
 221:     if has_detector:
 222:         role_parts.append("detector")
 223:     if has_vlm:
 224:         role_parts.append("vlm")
 225:     if has_quality:
 226:         role_parts.append("quality")
 227:     if has_consistency:
 228:         role_parts.append("consistency")
 229:     if has_label:
 230:         role_parts.append("label")
 231: 
 232:     if len(ad2_found) == 0:
 233:         source_role = "non_ad2_or_summary"
 234:         qcr_value = "none"
 235:     elif has_image and has_label and has_detector and has_vlm and has_quality:
 236:         source_role = "ad2_qcr_ready_or_near_ready"
 237:         qcr_value = "high"
 238:     elif has_image and has_label and (has_detector or has_vlm or has_quality):
 239:         source_role = "ad2_partial_per_image_source"
 240:         qcr_value = "medium"
 241:     elif len(ad2_found) > 0:
 242:         source_role = "ad2_summary_or_category_level_source"
 243:         qcr_value = "low"
 244:     else:
 245:         source_role = "unclassified"
 246: 
 247:     return {
 248:         **base,
 249:         "readable": True,
... 函数剩余 21 行已省略 ...
```

**函数 `classify_col`：`experiments/stage18_ad2_qcr_ablation/build_stage18_b1_ad2_qcr_schema_profile.py`，第 102–118 行**

```python
 102: def classify_col(col: str) -> str:
 103:     c = col.lower()
 104: 
 105:     if has_any(c, KEY_PATTERNS):
 106:         return "key_or_geometry"
 107:     if has_any(c, LABEL_PATTERNS):
 108:         return "label"
 109:     if has_any(c, VLM_PATTERNS):
 110:         return "vlm_or_text_score"
 111:     if has_any(c, QUALITY_PATTERNS):
 112:         return "quality_or_region_feature"
 113:     if has_any(c, CONSISTENCY_PATTERNS):
 114:         return "consistency"
 115:     if has_any(c, DETECTOR_PATTERNS):
 116:         return "detector_score"
 117: 
 118:     return "other"
```

**函数 `profile_file`：`experiments/stage18_ad2_qcr_ablation/build_stage18_b1_ad2_qcr_schema_profile.py`，第 172–274 行**

```python
 172: def profile_file(path: Path) -> tuple[dict, list[dict]]:
 173:     df = read_csv_robust(path)
 174: 
 175:     if df is None:
 176:         return {
 177:             "file": str(path.relative_to(ROOT)),
 178:             "exists_readable": False,
 179:             "num_rows": None,
 180:             "num_cols": None,
 181:             "ad2_categories_found": "",
 182:             "ad2_coverage_count": 0,
 183:             "rows_fruit_jelly": 0,
 184:             "rows_sheet_metal": 0,
 185:             "rows_vial": 0,
 186:             "rows_walnuts": 0,
 187:             "key_cols": "",
 188:             "label_cols": "",
 189:             "detector_like_cols": "",
 190:             "vlm_like_cols": "",
 191:             "quality_like_cols": "",
 192:             "consistency_like_cols": "",
 193:             "numeric_cols": "",
 194:             "qcr_readiness": "unreadable",
 195:             "notes": "missing or unreadable",
 196:         }, []
 197: 
 198:     cats = ad2_categories_in_df(df, path)
 199:     row_counts = rows_by_category(df)
 200: 
 201:     classified = {c: classify_col(c) for c in df.columns}
 202: 
 203:     key_cols = [c for c, t in classified.items() if t == "key_or_geometry"]
 204:     label_cols = [c for c, t in classified.items() if t == "label"]
 205:     detector_cols = [c for c, t in classified.items() if t == "detector_score"]
 206:     vlm_cols = [c for c, t in classified.items() if t == "vlm_or_text_score"]
 207:     quality_cols = [c for c, t in classified.items() if t == "quality_or_region_feature"]
 208:     consistency_cols = [c for c, t in classified.items() if t == "consistency"]
 209: 
 210:     numeric_cols = []
 211:     col_rows = []
 212: 
 213:     for c in df.columns:
 214:         s = pd.to_numeric(df[c], errors="coerce")
 215:         is_numeric = s.notna().sum() > 0 and s.notna().sum() >= max(3, int(0.2 * len(df)))
 216: 
 217:         if is_numeric:
 218:             numeric_cols.append(c)
 219: 
 220:         mn, mx, mean = numeric_summary(df, c)
 221: 
 222:         col_rows.append(
 223:             {
 224:                 "file": str(path.relative_to(ROOT)),
 225:                 "column": c,
 226:                 "classified_as": classified[c],
 227:                 "dtype": str(df[c].dtype),
 228:                 "non_null": int(df[c].notna().sum()),
 229:                 "unique_sample_count": int(df[c].dropna().astype(str).nunique()),
 230:                 "numeric_min": mn,
 231:                 "numeric_max": mx,
 232:                 "numeric_mean": mean,
 233:                 "sample_values": "; ".join(df[c].dropna().astype(str).unique()[:5]),
 234:             }
 235:         )
 236: 
 237:     has_label = len(label_cols) > 0
 238:     has_detector = len(detector_cols) > 0
 239:     has_vlm = len(vlm_cols) > 0
 240:     has_quality = len(quality_cols) > 0
 241:     has_key = any(c in df.columns for c in ["image_path", "image_key", "path", "filename", "category"])
 242: 
 243:     if len(cats) == 4 and has_key and has_label and has_detector and has_vlm and has_quality:
 244:         readiness = "qcr_ready"
 245:     elif len(cats) == 4 and has_key and has_label and (has_detector or has_vlm or has_quality):
 246:         readiness = "partial_join_source"
 247:     elif len(cats) > 0:
 248:         readiness = "ad2_summary_or_auxiliary"
 249:     else:
 250:         readiness = "non_ad2_or_irrelevant"
 251: 
... 函数剩余 23 行已省略 ...
```

**函数 `main`：`experiments/stage18_ad2_qcr_ablation/build_stage18_b7_qcr_final_claim_update.py`，第 45–245 行**

```python
  45: def main() -> None:
  46:     OUT_DIR.mkdir(parents=True, exist_ok=True)
  47:     DOC_DIR.mkdir(parents=True, exist_ok=True)
  48: 
  49:     summary = read_csv(IN_B6_SUMMARY)
  50:     folds = read_csv(IN_B6_FOLDS)
  51: 
  52:     s = summary[summary["selector"] == LOCKED_SELECTOR]
  53:     if s.empty:
  54:         raise RuntimeError(f"Missing locked selector in B6 summary: {LOCKED_SELECTOR}")
  55:     s = s.iloc[0]
  56: 
  57:     f = folds[folds["selector"] == LOCKED_SELECTOR].copy()
  58:     if f.empty:
  59:         raise RuntimeError(f"Missing locked selector folds: {LOCKED_SELECTOR}")
  60: 
  61:     # Decide final AD2 score variant.
  62:     quality_mean = float(s["mean_test_quality_qcr"])
  63:     adaptive_mean = float(s["mean_test_adaptive_qcr"])
  64: 
  65:     if quality_mean >= adaptive_mean:
  66:         final_variant = "Quality-Calibrated QCR"
  67:         final_score = quality_mean
  68:         final_delta = float(s["mean_delta_quality_minus_V3"])
  69:         final_wins = int(s["wins_quality_over_V3"])
  70:         final_worst_delta = float(s["worst_quality_delta"])
  71:         final_note = "Quality-only calibration is selected because it is slightly stronger than adaptive refinement on AD2."
  72:     else:
  73:         final_variant = "Quality-Calibrated QCR + adaptive refinement"
  74:         final_score = adaptive_mean
  75:         final_delta = float(s["mean_delta_adaptive_minus_V3"])
  76:         final_wins = int(s["wins_adaptive_over_V3"])
  77:         final_worst_delta = float(s["worst_adaptive_delta"])
  78:         final_note = "Adaptive refinement is selected because it is stronger than quality-only calibration on AD2."
  79: 
  80:     final_table = pd.DataFrame(
  81:         [
  82:             {
  83:                 "setting": "AD2 four-category LOCO policy",
  84:                 "method": "Naive detector-crop fusion",
  85:                 "mean_image_auroc": float(s["mean_test_V3"]),
  86:                 "delta_vs_naive": 0.0,
  87:                 "wins_vs_naive": "",
  88:                 "paper_role": "baseline",
  89:             },
  90:             {
  91:                 "setting": "AD2 four-category LOCO policy",
  92:                 "method": "Quality-Calibrated QCR",
  93:                 "mean_image_auroc": quality_mean,
  94:                 "delta_vs_naive": float(s["mean_delta_quality_minus_V3"]),
  95:                 "wins_vs_naive": f"{int(s['wins_quality_over_V3'])}/4",
  96:                 "paper_role": "main_qcr_support",
  97:             },
  98:             {
  99:                 "setting": "AD2 four-category LOCO policy",
 100:                 "method": "Quality-Calibrated QCR + adaptive refinement",
 101:                 "mean_image_auroc": adaptive_mean,
 102:                 "delta_vs_naive": float(s["mean_delta_adaptive_minus_V3"]),
 103:                 "wins_vs_naive": f"{int(s['wins_adaptive_over_V3'])}/4",
 104:                 "paper_role": "auxiliary_refinement",
 105:             },
 106:         ]
 107:     )
 108: 
 109:     final_table.to_csv(OUT_FINAL_TABLE, index=False, lineterminator="\n")
 110: 
 111:     fold_table = f[
 112:         [
 113:             "heldout_category",
 114:             "selected_q_source",
 115:             "selected_q_direction",
 116:             "selected_eta",
 117:             "selected_gamma",
 118:             "test_V3",
 119:             "test_quality_qcr",
 120:             "test_adaptive_qcr",
 121:             "test_delta_quality_minus_V3",
 122:             "test_delta_adaptive_minus_V3",
 123:         ]
 124:     ].copy()
... 函数剩余 121 行已省略 ...
```

**函数 `build_predictions`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 98–239 行**

```python
  98: def build_predictions() -> pd.DataFrame:
  99:     img = read_csv_strict(IN_IMAGE)
 100:     cand = read_csv_strict(IN_CAND)
 101: 
 102:     img = img[img["category"].isin(AD2_CATEGORIES)].copy()
 103:     cand = cand[cand["category"].isin(AD2_CATEGORIES)].copy()
 104: 
 105:     required_img = ["category", "image_path", "gt_binary", "patchcore_score"]
 106:     for c in required_img:
 107:         if c not in img.columns:
 108:             raise RuntimeError(f"Missing required image-level column: {c}")
 109: 
 110:     # Aggregate candidate quality per image. Do not use GT coverage columns as quality.
 111:     q_cols = [
 112:         c for c in [
 113:             "candidate_score_max",
 114:             "candidate_score_mean",
 115:             "tight_candidate_mask_density",
 116:             "context_candidate_mask_density",
 117:             "map_area",
 118:         ]
 119:         if c in cand.columns
 120:     ]
 121: 
 122:     if not q_cols:
 123:         raise RuntimeError("No candidate quality/source columns found in candidate score file.")
 124: 
 125:     agg_dict = {}
 126:     if "candidate_score_mean" in cand.columns:
 127:         agg_dict["candidate_score_mean_max"] = ("candidate_score_mean", "max")
 128:         agg_dict["candidate_score_mean_mean"] = ("candidate_score_mean", "mean")
 129:     if "candidate_score_max" in cand.columns:
 130:         agg_dict["candidate_score_max_max"] = ("candidate_score_max", "max")
 131:     if "tight_candidate_mask_density" in cand.columns:
 132:         agg_dict["tight_candidate_mask_density_max"] = ("tight_candidate_mask_density", "max")
 133:     if "context_candidate_mask_density" in cand.columns:
 134:         agg_dict["context_candidate_mask_density_max"] = ("context_candidate_mask_density", "max")
 135:     if "candidate_rank" in cand.columns:
 136:         agg_dict["num_candidates"] = ("candidate_rank", "count")
 137: 
 138:     q = cand.groupby(["category", "image_path"], as_index=False).agg(**agg_dict)
 139: 
 140:     df = img.merge(q, on=["category", "image_path"], how="left", validate="one_to_one")
 141: 
 142:     # Raw evidence.
 143:     df["D_raw_patchcore"] = pd.to_numeric(df["patchcore_score"], errors="coerce")
 144: 
 145:     if "context_topk_mean_score" in df.columns:
 146:         df["M_raw_crop_topk"] = pd.to_numeric(df["context_topk_mean_score"], errors="coerce")
 147:         m_source = "context_topk_mean_score"
 148:     elif "context_topk_max_score" in df.columns:
 149:         df["M_raw_crop_topk"] = pd.to_numeric(df["context_topk_max_score"], errors="coerce")
 150:         m_source = "context_topk_max_score"
 151:     elif "context_top1_score" in df.columns:
 152:         df["M_raw_crop_topk"] = pd.to_numeric(df["context_top1_score"], errors="coerce")
 153:         m_source = "context_top1_score"
 154:     else:
 155:         raise RuntimeError("No context crop VLM score column found in image-level predictions.")
 156: 
 157:     if "full_image_score" in df.columns:
 158:         df["F_raw_full_image_vlm"] = pd.to_numeric(df["full_image_score"], errors="coerce")
 159:         full_source = "full_image_score"
 160:     elif "full_image_anomaly_score" in df.columns:
 161:         df["F_raw_full_image_vlm"] = pd.to_numeric(df["full_image_anomaly_score"], errors="coerce")
 162:         full_source = "full_image_anomaly_score"
 163:     else:
 164:         df["F_raw_full_image_vlm"] = np.nan
 165:         full_source = "missing"
 166: 
 167:     if "candidate_score_mean_max" in df.columns:
 168:         df["Q_raw_candidate_quality"] = pd.to_numeric(df["candidate_score_mean_max"], errors="coerce")
 169:         q_source = "max(candidate_score_mean)"
 170:     elif "candidate_score_max_max" in df.columns:
 171:         df["Q_raw_candidate_quality"] = pd.to_numeric(df["candidate_score_max_max"], errors="coerce")
 172:         q_source = "max(candidate_score_max)"
 173:     else:
 174:         raise RuntimeError("No usable non-GT candidate quality column found after aggregation.")
 175: 
 176:     # Per-category normalization to avoid cross-category scale leakage.
 177:     df = norm_by_category(df, "D_raw_patchcore", "D")
... 函数剩余 62 行已省略 ...
```

**函数 `write_report`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 362–441 行**

```python
 362: def write_report(pred: pd.DataFrame, per_cat: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
 363:     lines = [
 364:         "# Stage 18-B2 AD2 Four-category QCR Ablation",
 365:         "",
 366:         "## Purpose",
 367:         "",
 368:         "Assemble AD2 four-category QCR ablation from existing Stage11 image-level VLM predictions and candidate-level quality evidence.",
 369:         "",
 370:         "This aligns the QCR ablation with the AD2 four-category system-level baseline setting.",
 371:         "",
 372:         "## Data",
 373:         "",
 374:         f"- input image-level predictions: `{IN_IMAGE.relative_to(ROOT)}`",
 375:         f"- input candidate scores: `{IN_CAND.relative_to(ROOT)}`",
 376:         f"- assembled images: `{len(pred)}`",
 377:         f"- categories: `{'; '.join(sorted(pred['category'].unique()))}`",
 378:         "- detector evidence `D`: normalized `patchcore_score`",
 379:         "- crop VLM evidence `M`: normalized context top-k VLM score",
 380:         "- candidate quality `Q`: normalized non-GT candidate score evidence",
 381:         "- consistency `K`: soft high-high consistency `D*M`",
 382:         "",
 383:         "## Summary table",
 384:         "",
 385:         "| Variant | Method | Role | Mean AUROC | Mean F1 |",
 386:         "|---|---|---|---:|---:|",
 387:     ]
 388: 
 389:     for _, r in summary.iterrows():
 390:         lines.append(
 391:             f"| {r['variant_id']} | {r['method']} | {r['paper_role']} | "
 392:             f"{fmt(r['mean_image_auroc'])} | {fmt(r['mean_best_f1'])} |"
 393:         )
 394: 
 395:     lines += [
 396:         "",
 397:         "## Claim-ready deltas",
 398:         "",
 399:         "| Comparison | Delta AUROC | A | B |",
 400:         "|---|---:|---:|---:|",
 401:     ]
 402: 
 403:     for _, r in deltas.iterrows():
 404:         lines.append(
 405:             f"| {r['comparison']} | {signed(r['delta_a_minus_b'])} | "
 406:             f"{fmt(r['mean_image_auroc_a'])} | {fmt(r['mean_image_auroc_b'])} |"
 407:         )
 408: 
 409:     lines += [
 410:         "",
 411:         "## Per-category AUROC",
 412:         "",
 413:         "| Category | Variant | Method | AUROC | F1 |",
 414:         "|---|---|---|---:|---:|",
 415:     ]
 416: 
 417:     for _, r in per_cat.iterrows():
 418:         lines.append(
 419:             f"| {r['category']} | {r['variant_id']} | {r['method']} | "
 420:             f"{fmt(r['image_auroc'])} | {fmt(r['best_f1'])} |"
 421:         )
 422: 
 423:     lines += [
 424:         "",
 425:         "## Interpretation rules",
 426:         "",
 427:         "- If V4 improves over V3, AD2 supports candidate quality calibration.",
 428:         "- If V6 only slightly improves over V4, keep adaptive consistency as refinement.",
 429:         "- If V5 is strong but unstable or not selected, keep fixed Q+C as diagnostic.",
 430:         "- Do not use this table to claim pixel-level segmentation SOTA.",
 431:         "",
 432:         "## Outputs",
 433:         "",
 434:         f"- `{OUT_PRED.relative_to(ROOT)}`",
 435:         f"- `{OUT_PER_CATEGORY.relative_to(ROOT)}`",
 436:         f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
 437:         f"- `{OUT_DELTAS.relative_to(ROOT)}`",
 438:         "",
 439:     ]
 440: 
 441:     OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
```

**函数 `evaluate_one`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b3_ad2_q_source_sweep.py`，第 156–232 行**

```python
 156: def evaluate_one(df: pd.DataFrame, q_raw_col: str, invert_q: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
 157:     work = df.copy()
 158: 
 159:     q_name = q_raw_col + ("__inverted" if invert_q else "__direct")
 160: 
 161:     work = norm_by_category(work, q_raw_col, "Q")
 162:     if invert_q:
 163:         work["Q"] = 1.0 - work["Q"]
 164: 
 165:     work["K"] = work["D"] * work["M"]
 166:     work["agreement"] = 1.0 - (work["D"] - work["M"]).abs()
 167:     work["mutual_anomaly_evidence"] = np.minimum(work["D"], work["M"])
 168:     work["adaptive_gate"] = work["Q"] * work["K"] * work["agreement"] * work["mutual_anomaly_evidence"]
 169: 
 170:     # Keep the same formulas as the current paper method.
 171:     work["V4_quality"] = 0.5 * work["D"] + 0.5 * work["M"] * (0.5 + 0.5 * work["Q"])
 172:     work["V5_fixed_qc"] = 0.4 * work["D"] + 0.4 * work["M"] + 0.1 * work["Q"] + 0.1 * work["K"]
 173:     work["V6_adaptive"] = work["V4_quality"] + 0.05 * work["adaptive_gate"]
 174: 
 175:     rows = []
 176: 
 177:     for cat, sub in work.groupby("category"):
 178:         y = sub["gt_binary"]
 179: 
 180:         au_v3 = safe_auroc(y, sub["V3_naive"])
 181:         au_v4 = safe_auroc(y, sub["V4_quality"])
 182:         au_v5 = safe_auroc(y, sub["V5_fixed_qc"])
 183:         au_v6 = safe_auroc(y, sub["V6_adaptive"])
 184:         au_d = safe_auroc(y, sub["D"])
 185:         au_m = safe_auroc(y, sub["M"])
 186:         au_q = safe_auroc(y, sub["Q"])
 187: 
 188:         rows.append(
 189:             {
 190:                 "q_source": q_raw_col,
 191:                 "q_direction": "inverted" if invert_q else "direct",
 192:                 "q_name": q_name,
 193:                 "category": cat,
 194:                 "num_images": len(sub),
 195:                 "auroc_detector_D": au_d,
 196:                 "auroc_crop_M": au_m,
 197:                 "auroc_quality_Q_alone": au_q,
 198:                 "auroc_V3_naive": au_v3,
 199:                 "auroc_V4_quality": au_v4,
 200:                 "auroc_V5_fixed_qc": au_v5,
 201:                 "auroc_V6_adaptive": au_v6,
 202:                 "delta_V4_minus_V3": au_v4 - au_v3,
 203:                 "delta_V6_minus_V4": au_v6 - au_v4,
 204:                 "delta_V6_minus_V3": au_v6 - au_v3,
 205:             }
 206:         )
 207: 
 208:     per_cat = pd.DataFrame(rows)
 209: 
 210:     summary = {
 211:         "q_source": q_raw_col,
 212:         "q_direction": "inverted" if invert_q else "direct",
 213:         "q_name": q_name,
 214:         "num_categories": int(per_cat["category"].nunique()),
 215:         "mean_auroc_detector_D": float(per_cat["auroc_detector_D"].mean()),
 216:         "mean_auroc_crop_M": float(per_cat["auroc_crop_M"].mean()),
 217:         "mean_auroc_quality_Q_alone": float(per_cat["auroc_quality_Q_alone"].mean()),
 218:         "mean_auroc_V3_naive": float(per_cat["auroc_V3_naive"].mean()),
 219:         "mean_auroc_V4_quality": float(per_cat["auroc_V4_quality"].mean()),
 220:         "mean_auroc_V5_fixed_qc": float(per_cat["auroc_V5_fixed_qc"].mean()),
 221:         "mean_auroc_V6_adaptive": float(per_cat["auroc_V6_adaptive"].mean()),
 222:         "mean_delta_V4_minus_V3": float(per_cat["delta_V4_minus_V3"].mean()),
 223:         "mean_delta_V6_minus_V4": float(per_cat["delta_V6_minus_V4"].mean()),
 224:         "mean_delta_V6_minus_V3": float(per_cat["delta_V6_minus_V3"].mean()),
 225:         "wins_V4_over_V3": int((per_cat["delta_V4_minus_V3"] > 0).sum()),
 226:         "wins_V6_over_V4": int((per_cat["delta_V6_minus_V4"] > 0).sum()),
 227:         "wins_V6_over_V3": int((per_cat["delta_V6_minus_V3"] > 0).sum()),
 228:         "worst_category_delta_V4_minus_V3": float(per_cat["delta_V4_minus_V3"].min()),
 229:         "worst_category": str(per_cat.sort_values("delta_V4_minus_V3").iloc[0]["category"]),
 230:     }
 231: 
 232:     return per_cat, pd.DataFrame([summary])
```

**函数 `score_config`：`experiments/stage18_ad2_qcr_ablation/run_stage18_b5_ad2_loco_qcr_policy_optimization.py`，第 161–202 行**

```python
 161: def score_config(base: pd.DataFrame, q_source: str, q_direction: str, eta: float, gamma: float) -> pd.DataFrame:
 162:     work = base.copy()
 163: 
 164:     work = norm_by_category(work, q_source, "Q")
 165:     if q_direction == "inverted":
 166:         work["Q"] = 1.0 - work["Q"]
 167: 
 168:     work["K"] = work["D"] * work["M"]
 169:     work["agreement"] = 1.0 - (work["D"] - work["M"]).abs()
 170:     work["mutual_anomaly_evidence"] = np.minimum(work["D"], work["M"])
 171:     work["adaptive_gate"] = work["Q"] * work["K"] * work["agreement"] * work["mutual_anomaly_evidence"]
 172: 
 173:     # Generalized QCR policy.
 174:     work["S_quality"] = work["V3_naive"] - eta * work["M"] * (1.0 - work["Q"])
 175:     work["S_adaptive"] = work["S_quality"] + gamma * work["adaptive_gate"]
 176: 
 177:     rows = []
 178: 
 179:     for cat, sub in work.groupby("category"):
 180:         au_v3 = safe_auroc(sub["gt_binary"], sub["V3_naive"])
 181:         au_q = safe_auroc(sub["gt_binary"], sub["S_quality"])
 182:         au_a = safe_auroc(sub["gt_binary"], sub["S_adaptive"])
 183:         au_qalone = safe_auroc(sub["gt_binary"], sub["Q"])
 184: 
 185:         rows.append(
 186:             {
 187:                 "category": cat,
 188:                 "q_source": q_source,
 189:                 "q_direction": q_direction,
 190:                 "eta": eta,
 191:                 "gamma": gamma,
 192:                 "auroc_V3_naive": au_v3,
 193:                 "auroc_quality_qcr": au_q,
 194:                 "auroc_adaptive_qcr": au_a,
 195:                 "auroc_Q_alone": au_qalone,
 196:                 "delta_quality_minus_V3": au_q - au_v3,
 197:                 "delta_adaptive_minus_V3": au_a - au_v3,
 198:                 "delta_adaptive_minus_quality": au_a - au_q,
 199:             }
 200:         )
 201: 
 202:     return pd.DataFrame(rows)
```

## 3. 旧 QCR 与 V3–V6 融合公式上下文

**`experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 84–96 行**

```python
  84:             "uses_crop_vlm": False,
  85:             "uses_quality": False,
  86:             "uses_consistency": False,
  87:             "uses_unknown": False,
  88:             "purpose": "Anchor baseline; proves whether QCR-U beats the detector alone.",
  89:         },
  90:         {
  91:             "variant_id": "V1",
  92:             "variant": "full_image_vlm",
  93:             "uses_detector_score": False,
  94:             "uses_full_image_vlm": True,
  95:             "uses_crop_vlm": False,
  96:             "uses_quality": False,
```

**`experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 109–154 行**

```python
 109:             "uses_unknown": False,
 110:             "purpose": "Tests whether localization-guided crops improve VLM scoring.",
 111:         },
 112:         {
 113:             "variant_id": "V3",
 114:             "variant": "naive_detector_crop_fusion",
 115:             "uses_detector_score": True,
 116:             "uses_full_image_vlm": False,
 117:             "uses_crop_vlm": True,
 118:             "uses_quality": False,
 119:             "uses_consistency": False,
 120:             "uses_unknown": False,
 121:             "purpose": "Naive fusion baseline; QCR-U must beat this or the method is not justified.",
 122:         },
 123:         {
 124:             "variant_id": "V4",
 125:             "variant": "quality_weighted_crop",
 126:             "uses_detector_score": True,
 127:             "uses_full_image_vlm": False,
 128:             "uses_crop_vlm": True,
 129:             "uses_quality": True,
 130:             "uses_consistency": False,
 131:             "uses_unknown": False,
 132:             "purpose": "Tests whether candidate quality contributes beyond crop scoring.",
 133:         },
 134:         {
 135:             "variant_id": "V5",
 136:             "variant": "quality_consistency_fusion",
 137:             "uses_detector_score": True,
 138:             "uses_full_image_vlm": False,
 139:             "uses_crop_vlm": True,
 140:             "uses_quality": True,
 141:             "uses_consistency": True,
 142:             "uses_unknown": False,
 143:             "purpose": "Core QCR-U binary anomaly recognition variant.",
 144:         },
 145:         {
 146:             "variant_id": "V6",
 147:             "variant": "qcr_u_full_optional_unknown",
 148:             "uses_detector_score": True,
 149:             "uses_full_image_vlm": False,
 150:             "uses_crop_vlm": True,
 151:             "uses_quality": True,
 152:             "uses_consistency": True,
 153:             "uses_unknown": True,
 154:             "purpose": "Only valid if a strict known/unknown protocol is available.",
```

**`experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 166–188 行**

```python
 166:     ]
 167: 
 168:     lines = []
 169:     lines += [
 170:         "# Stage 16-A0 QCR-U 输入审计与消融计划",
 171:         "",
 172:         "## 1. 本阶段目的",
 173:         "",
 174:         "Stage 15 已经完成强基线结论锁定。下一阶段进入 QCR-U，但不能直接写新方法或乱调融合权重。",
 175:         "",
 176:         "本阶段只做三件事：",
 177:         "",
 178:         "1. 审计 Stage 9 / Stage 13 / Stage 15 已有结果文件。",
 179:         "2. 判断哪些文件可以作为 QCR-U ablation 的输入。",
 180:         "3. 锁定 QCR-U 消融变量，避免把 heuristic fusion 包装成方法。",
 181:         "",
 182:         "## 2. 输入文件审计结果",
 183:         "",
 184:         f"- total_sources_checked: `{len(inventory)}`",
 185:         f"- usable_sources: `{len(usable)}`",
 186:         f"- missing_sources: `{len(missing)}`",
 187:         f"- malformed_or_empty_sources: `{len(malformed)}`",
 188:         "",
```

**`experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 191–216 行**

```python
 191:         f"`{OUT_INVENTORY.relative_to(ROOT)}`",
 192:         "",
 193:         "## 3. 必须解决的硬问题",
 194:         "",
 195:         "QCR-U 不能只是：",
 196:         "",
 197:         "```text",
 198:         "score = alpha * detector + beta * vlm",
 199:         "```",
 200:         "",
 201:         "它必须至少证明：",
 202:         "",
 203:         "1. candidate quality 是否有效。",
 204:         "2. detector-VLM consistency 是否有效。",
 205:         "3. QCR-U 是否稳定优于 naive fusion。",
 206:         "4. 参数是否来自固定协议，而不是 test-set 调参。",
 207:         "",
 208:         "## 4. QCR-U 消融计划",
 209:         "",
 210:         "| Variant | Detector | Crop VLM | Quality | Consistency | Unknown | Purpose |",
 211:         "|---|---:|---:|---:|---:|---:|---|",
 212:     ]
 213: 
 214:     for _, r in plan.iterrows():
 215:         lines.append(
 216:             f"| {r['variant']} | "
```

**`experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 225–240 行**

```python
 225:     lines += [
 226:         "",
 227:         "## 5. 下一步决策",
 228:         "",
 229:         "如果 Stage 9 的旧 QCR-U 文件已经包含足够字段，下一步进入：",
 230:         "",
 231:         "```text",
 232:         "Stage 16-A1: QCR-U fixed-protocol ablation implementation",
 233:         "```",
 234:         "",
 235:         "如果字段不够，先补：",
 236:         "",
 237:         "```text",
 238:         "Stage 16-A1-input: 构建统一 prediction table",
 239:         "```",
 240:         "",
```

**`experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 20–34 行**

```python
  20: 
  21: VARIANT_NAMES = {
  22:     "V0": "detector_only",
  23:     "V2": "crop_topk_vlm",
  24:     "V3": "naive_detector_crop_fusion",
  25:     "V4": "quality_weighted_crop",
  26:     "V5": "quality_consistency_fusion",
  27: }
  28: 
  29: 
  30: def read_csv_robust(path: Path) -> pd.DataFrame:
  31:     df = pd.read_csv(path)
  32:     if len(df.columns) <= 1:
  33:         raise RuntimeError(f"{path} read as <=1 column. Fix CSV line breaks before running Stage 16-A2.")
  34:     return df
```

**`experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 43–65 行**

```python
  43: 
  44: def compute_protocol_deltas(per_config: pd.DataFrame) -> pd.DataFrame:
  45:     piv = pivot_metric(per_config, "auroc")
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
  58: 
  59:     out["v5_beats_naive"] = out["delta_v5_minus_v3_naive"] > 0
  60:     out["v5_beats_quality_only"] = out["delta_v5_minus_v4_quality"] > 0
  61:     out["v5_beats_detector"] = out["delta_v5_minus_v0_detector"] > 0
  62:     out["v5_beats_crop"] = out["delta_v5_minus_v2_crop"] > 0
  63:     out["quality_beats_naive"] = out["delta_v4_minus_v3_naive"] > 0
  64: 
  65:     return out
```

**`experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 75–91 行**

```python
  75: def make_summary(delta: pd.DataFrame) -> pd.DataFrame:
  76:     rows = []
  77: 
  78:     checks = [
  79:         ("V5 > V3 naive fusion", "v5_beats_naive", "delta_v5_minus_v3_naive"),
  80:         ("V5 > V4 quality-only", "v5_beats_quality_only", "delta_v5_minus_v4_quality"),
  81:         ("V5 > V0 detector-only", "v5_beats_detector", "delta_v5_minus_v0_detector"),
  82:         ("V5 > V2 crop-VLM-only", "v5_beats_crop", "delta_v5_minus_v2_crop"),
  83:         ("V4 > V3 naive fusion", "quality_beats_naive", "delta_v4_minus_v3_naive"),
  84:     ]
  85: 
  86:     for name, bool_col, delta_col in checks:
  87:         wins, total, rate = summarize_boolean(delta, bool_col)
  88:         rows.append(
  89:             {
  90:                 "check": name,
  91:                 "wins": wins,
```

**`experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 102–128 行**

```python
 102:     for eval_mode, g in delta.groupby("eval_mode"):
 103:         wins, total, rate = summarize_boolean(g, "v5_beats_naive")
 104:         rows.append(
 105:             {
 106:                 "check": f"V5 > V3 naive fusion by eval_mode={eval_mode}",
 107:                 "wins": wins,
 108:                 "total_protocols": total,
 109:                 "win_rate": rate,
 110:                 "mean_delta": g["delta_v5_minus_v3_naive"].mean(),
 111:                 "median_delta": g["delta_v5_minus_v3_naive"].median(),
 112:                 "min_delta": g["delta_v5_minus_v3_naive"].min(),
 113:                 "max_delta": g["delta_v5_minus_v3_naive"].max(),
 114:             }
 115:         )
 116: 
 117:         wins, total, rate = summarize_boolean(g, "v5_beats_quality_only")
 118:         rows.append(
 119:             {
 120:                 "check": f"V5 > V4 quality-only by eval_mode={eval_mode}",
 121:                 "wins": wins,
 122:                 "total_protocols": total,
 123:                 "win_rate": rate,
 124:                 "mean_delta": g["delta_v5_minus_v4_quality"].mean(),
 125:                 "median_delta": g["delta_v5_minus_v4_quality"].median(),
 126:                 "min_delta": g["delta_v5_minus_v4_quality"].min(),
 127:                 "max_delta": g["delta_v5_minus_v4_quality"].max(),
 128:             }
```

**`experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 163–175 行**

```python
 163: 
 164: def write_report(delta: pd.DataFrame, summary: pd.DataFrame, failures: pd.DataFrame) -> None:
 165:     lines = []
 166:     lines += [
 167:         "# Stage 16-A2 QCR-U Robustness Check",
 168:         "",
 169:         "## 1. Purpose",
 170:         "",
 171:         "Stage 16-A1 showed that fixed quality-consistency fusion can improve the best protocol.",
 172:         "",
 173:         "Stage 16-A2 checks whether that gain is robust across all protocols, instead of only appearing in the best protocol.",
 174:         "",
 175:         "## 2. Overall Robustness Summary",
```

**`experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 188–222 行**

```python
 188:     lines += [
 189:         "",
 190:         "## 3. Protocol-level Deltas",
 191:         "",
 192:         "| Backbone | Strategy | Eval Mode | V5 AUROC | V3 AUROC | V4 AUROC | V5-V3 | V5-V4 |",
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
 205:         "",
 206:         "## 4. Failure / Weakness Cases",
 207:         "",
 208:     ]
 209: 
 210:     if failures.empty:
 211:         lines.append("No failure case found under the current checks.")
 212:     else:
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
```

**`experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 227–241 行**

```python
 227:     lines += [
 228:         "",
 229:         "## 5. Decision Rule",
 230:         "",
 231:         "If V5 is consistently better than V3 naive fusion but often worse than V4 quality-only, the consistency term should not be claimed as universally beneficial.",
 232:         "",
 233:         "In that case, the next method should be revised from fixed Q+C fusion to adaptive QCR-U:",
 234:         "",
 235:         "```text",
 236:         "use quality-weighted crop as the stable core;",
 237:         "apply consistency only when detector and VLM evidence are both reliable;",
 238:         "avoid adding consistency under weak/full-image protocols where it hurts.",
 239:         "```",
 240:         "",
 241:         "## 6. Outputs",
```

**`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 36–51 行**

```python
  36: 
  37: VARIANTS = [
  38:     ("V0", "detector_only", "score_detector_only"),
  39:     ("V2", "crop_topk_vlm", "score_crop_vlm"),
  40:     ("V3", "naive_detector_crop_fusion", "score_naive"),
  41:     ("V4", "quality_weighted_crop", "score_quality"),
  42:     ("V5", "fixed_quality_consistency", "score_fixed_qc"),
  43:     ("V6", "adaptive_qcru", "score_adaptive_qcru"),
  44: ]
  45: 
  46: 
  47: def read_csv_strict(path: Path) -> pd.DataFrame:
  48:     df = pd.read_csv(path)
  49:     if len(df.columns) <= 1:
  50:         raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
  51:     return df
```

**`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 283–295 行**

```python
 283: 
 284: 
 285: def build_primary_table(per_config: pd.DataFrame) -> pd.DataFrame:
 286:     # Paper-facing primary test:
 287:     # QCR-U is a crop/candidate method, so crop_topk_ensemble is the relevant primary setting.
 288:     primary = per_config[
 289:         (per_config["dataset"] == "VisA")
 290:         & (per_config["strategy"] == "inspection_binary")
 291:         & (per_config["eval_mode"] == "crop_topk_ensemble")
 292:     ].copy()
 293: 
 294:     if primary.empty:
 295:         primary = per_config.copy()
```

**`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 314–329 行**

```python
 314:         return piv
 315: 
 316:     for scope, df in [("primary_protocol", primary), ("all_protocols", per_config)]:
 317:         for left, right, label in [
 318:             ("V6", "V3", "adaptive_qcru_minus_naive"),
 319:             ("V6", "V4", "adaptive_qcru_minus_quality"),
 320:             ("V6", "V5", "adaptive_qcru_minus_fixed_qc"),
 321:             ("V4", "V3", "quality_minus_naive"),
 322:         ]:
 323:             d = get_delta(df, left, right)
 324:             if d.empty:
 325:                 continue
 326:             delta_col = f"delta_{left}_minus_{right}"
 327:             rows.append(
 328:                 {
 329:                     "scope": scope,
```

**`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 348–366 行**

```python
 348: 
 349:     if not primary_v6_v4.empty:
 350:         mean_delta = float(primary_v6_v4.iloc[0]["mean_delta"])
 351:         if mean_delta >= 0.005:
 352:             recommendation = "Adaptive QCR-U can be presented as the final candidate method."
 353:             method_name = "Adaptive QCR-U"
 354:         elif mean_delta > 0:
 355:             recommendation = "Use Quality-Calibrated QCR as the main method; describe adaptive consistency as a small gated refinement."
 356:             method_name = "Quality-Calibrated QCR with adaptive consistency refinement"
 357:         else:
 358:             recommendation = "Do not use Adaptive QCR-U as final method; use quality-weighted fusion."
 359:             method_name = "Quality-Calibrated Localization-Guided Fusion"
 360:     else:
 361:         recommendation = "Insufficient primary comparison."
 362:         method_name = "undecided"
 363: 
 364:     decision["final_recommendation"] = ""
 365:     decision["recommended_method_name"] = ""
 366:     if len(decision) > 0:
```

**`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 377–407 行**

```python
 377:     decision: pd.DataFrame,
 378: ) -> None:
 379:     lines = []
 380:     lines += [
 381:         "# Stage 16-B Adaptive QCR-U Paper-facing Comparison",
 382:         "",
 383:         "## 1. Purpose",
 384:         "",
 385:         "This stage connects the Adaptive QCR-U candidate back to a paper-facing comparison table.",
 386:         "",
 387:         "It tests whether Adaptive QCR-U should be the final method name, or whether the method should be downgraded to quality-calibrated localization-guided fusion.",
 388:         "",
 389:         "## 2. Primary Protocol",
 390:         "",
 391:         "The primary protocol is:",
 392:         "",
 393:         "```text",
 394:         "dataset = VisA",
 395:         "strategy = inspection_binary",
 396:         "eval_mode = crop_topk_ensemble",
 397:         "```",
 398:         "",
 399:         "Reason: QCR-U is a candidate/crop reliability method. `full_all` is useful for diagnostics but is not the correct primary protocol for a crop-based reliability module.",
 400:         "",
 401:         "## 3. Primary Protocol Table",
 402:         "",
 403:         "| Backbone | Variant | AUROC | AP | Best F1 | Best Acc |",
 404:         "|---|---|---:|---:|---:|---:|",
 405:     ]
 406: 
 407:     for _, r in primary.iterrows():
```

**`experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 440–452 行**

```python
 440:         f"- recommendation: {rec}",
 441:         "",
 442:         "## 6. Interpretation Rule",
 443:         "",
 444:         "If Adaptive QCR-U only improves over quality-only by a negligible margin, the paper should not overclaim adaptive consistency.",
 445:         "",
 446:         "In that case, the correct claim is:",
 447:         "",
 448:         "```text",
 449:         "Candidate quality provides the main reliability calibration gain, while adaptive consistency is a conservative refinement that avoids fixed-consistency degradation.",
 450:         "```",
 451:         "",
 452:         "## 7. Outputs",
```

**`experiments/stage16_qcru_ablation/build_stage16_c_final_method_claims.py`，第 102–118 行**

```python
 102:     primary = read_csv_strict(IN_PRIMARY)
 103:     decision = read_csv_strict(IN_DECISION)
 104: 
 105:     # Primary-protocol deltas.
 106:     d_v4_v3 = get_primary_delta(primary, "V4", "V3")
 107:     d_v6_v3 = get_primary_delta(primary, "V6", "V3")
 108:     d_v6_v4 = get_primary_delta(primary, "V6", "V4")
 109:     d_v5_v4 = get_primary_delta(primary, "V5", "V4")
 110:     d_v6_v5 = get_primary_delta(primary, "V6", "V5")
 111: 
 112:     # All-protocol summaries from Stage 16-B decision file.
 113:     all_v4_v3 = lookup_decision(decision, "all_protocols", "quality_minus_naive")
 114:     all_v6_v3 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_naive")
 115:     all_v6_v4 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_quality")
 116:     all_v6_v5 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_fixed_qc")
 117: 
 118:     # Recommendation from Stage 16-B.
```

**`experiments/stage16_qcru_ablation/build_stage16_c_final_method_claims.py`，第 128–156 行**

```python
 128:             final_recommendation = vals[0]
 129: 
 130:     if not recommended_name:
 131:         if d_v6_v4["mean_delta"] >= 0.005:
 132:             recommended_name = "Adaptive QCR-U"
 133:         elif d_v6_v4["mean_delta"] > 0:
 134:             recommended_name = "Quality-Calibrated QCR with adaptive consistency refinement"
 135:         else:
 136:             recommended_name = "Quality-Calibrated Localization-Guided Fusion"
 137: 
 138:     if not final_recommendation:
 139:         final_recommendation = (
 140:             "Use the quality-calibrated method as the main paper-facing method; "
 141:             "treat adaptive consistency as a conservative refinement."
 142:         )
 143: 
 144:     rows = [
 145:         {
 146:             "claim_id": "C1",
 147:             "claim_type": "final_method_name",
 148:             "claim": "Use Quality-Calibrated QCR as the main paper-facing method family.",
 149:             "evidence": (
 150:                 f"Stage 16-B recommends `{recommended_name}`. "
 151:                 f"Primary adaptive-minus-quality mean delta is {fmt(d_v6_v4['mean_delta'])} AUROC."
 152:             ),
 153:             "paper_status": "use",
 154:         },
 155:         {
 156:             "claim_id": "C2",
```

**`experiments/stage16_qcru_ablation/build_stage16_c_final_method_claims.py`，第 228–245 行**

```python
 228:         "# Stage 16-C Final Method Claims",
 229:         "",
 230:         "## 1. Decision",
 231:         "",
 232:         "The final method should not be written as fixed QCR-U or as a consistency-driven method.",
 233:         "",
 234:         "The paper-facing method family is:",
 235:         "",
 236:         "```text",
 237:         "Quality-Calibrated QCR",
 238:         "```",
 239:         "",
 240:         "A more descriptive paper title/method phrase is:",
 241:         "",
 242:         "```text",
 243:         "Quality-Calibrated Localization-Guided VLM Reasoning",
 244:         "```",
 245:         "",
```

**`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 116–135 行**

```python
 116:     if variant_id == "V0":
 117:         return "Detector only", "anchor_baseline", True
 118:     if variant_id == "V2":
 119:         return "Crop VLM only", "vlm_crop_baseline", True
 120:     if variant_id == "V3":
 121:         return "Naive detector-crop fusion", "naive_fusion_baseline", True
 122:     if variant_id == "V4":
 123:         return "Quality-Calibrated QCR", "main_effective_method_core", True
 124:     if variant_id == "V5":
 125:         return "Fixed Q+C fusion", "diagnostic_not_final", False
 126:     if variant_id == "V6":
 127:         return "Quality-Calibrated QCR + adaptive consistency refinement", "final_refinement_variant", True
 128:     return variant, "other", True
 129: 
 130: 
 131: def build_qcr_table(primary: pd.DataFrame) -> pd.DataFrame:
 132:     df = primary.copy()
 133:     df = to_num(df, ["auroc", "ap", "best_f1", "best_accuracy", "best_threshold"])
 134: 
 135:     rows = []
```

**`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 154–166 行**

```python
 154:                 "best_f1": float(r["best_f1"]),
 155:                 "best_accuracy": float(r["best_accuracy"]),
 156:                 "paper_role": paper_role,
 157:                 "use_in_main_claim": use_in_main_claim,
 158:                 "comparison_scope": "Stage16-B QCR primary protocol",
 159:                 "directly_comparable_with_system_panel": False,
 160:             }
 161:         )
 162: 
 163:     out = pd.DataFrame(rows)
 164:     out = out.sort_values(["backbone", "variant_id"]).reset_index(drop=True)
 165:     return out
 166: 
```

**`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 205–231 行**

```python
 205:                 "paper_interpretation": interpret_system_delta(name),
 206:             }
 207:         )
 208: 
 209:     # QCR primary protocol mean over backbones.
 210:     piv = qcr_table.pivot_table(
 211:         index=["dataset", "strategy", "eval_mode", "backbone"],
 212:         columns="variant_id",
 213:         values="image_auroc",
 214:         aggfunc="first",
 215:     ).reset_index()
 216:     piv.columns.name = None
 217: 
 218:     qcr_pairs = [
 219:         ("Quality-Calibrated QCR vs naive fusion", "V4", "V3", "qcr_core_delta"),
 220:         ("Adaptive refinement vs Quality-Calibrated QCR", "V6", "V4", "adaptive_refinement_delta"),
 221:         ("Adaptive refinement vs naive fusion", "V6", "V3", "qcr_final_delta"),
 222:         ("Fixed Q+C vs Quality-Calibrated QCR", "V5", "V4", "diagnostic_fixed_consistency_delta"),
 223:         ("Adaptive refinement vs fixed Q+C", "V6", "V5", "robustness_tradeoff_delta"),
 224:     ]
 225: 
 226:     for name, left_col, right_col, delta_type in qcr_pairs:
 227:         if left_col not in piv.columns or right_col not in piv.columns:
 228:             continue
 229:         d = piv[left_col] - piv[right_col]
 230:         rows.append(
 231:             {
```

**`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 275–299 行**

```python
 275:     return "Claim-supporting delta."
 276: 
 277: 
 278: def interpret_qcr_delta(name: str, delta: float) -> str:
 279:     if "Quality-Calibrated QCR vs naive" in name:
 280:         return "Candidate quality calibration is the main method gain."
 281:     if "Adaptive refinement vs Quality" in name:
 282:         if abs(delta) < 0.005:
 283:             return "Adaptive consistency is only a small refinement, not a main contribution."
 284:         return "Adaptive consistency provides a meaningful refinement."
 285:     if "Adaptive refinement vs naive" in name:
 286:         return "Final refinement variant improves over naive fusion."
 287:     if "Fixed Q+C" in name:
 288:         return "Fixed consistency is diagnostic only because robustness is not stable across protocols."
 289:     if "Adaptive refinement vs fixed" in name:
 290:         return "Adaptive refinement trades peak primary-protocol AUROC for robustness."
 291:     return "QCR delta."
 292: 
 293: 
 294: def write_report(
 295:     system_table: pd.DataFrame,
 296:     qcr_table: pd.DataFrame,
 297:     deltas: pd.DataFrame,
 298:     claims: pd.DataFrame,
 299: ) -> None:
```

**`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 307–329 行**

```python
 307:         "",
 308:         "The final method family is:",
 309:         "",
 310:         "```text",
 311:         "Quality-Calibrated QCR",
 312:         "```",
 313:         "",
 314:         "The adaptive consistency term is treated only as a conservative refinement, not as the main performance source.",
 315:         "",
 316:         "## 2. Important Comparison Rule",
 317:         "",
 318:         "This report uses two panels because Stage 15 system baselines and Stage 16 QCR ablations are not the same protocol.",
 319:         "",
 320:         "- Panel A compares system-level baselines from Stage 15.",
 321:         "- Panel B compares QCR variants under the Stage 16-B QCR primary protocol.",
 322:         "",
 323:         "Do not merge the two panels into a single global ranking.",
 324:         "",
 325:         "## 3. Panel A: System-level Strong Baseline Comparison",
 326:         "",
 327:         "| Rank | Method | Mean Image AUROC | Role | Fairness Tag |",
 328:         "|---:|---|---:|---|---|",
 329:     ]
```

**`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 341–353 行**

```python
 341:         "- Use `PatchCore + context VLM, LOCO` as the fair system-level result.",
 342:         "- Use `same-set` only as an upper-bound diagnostic.",
 343:         "- Keep `EfficientAD-30` explicitly labeled as fixed-budget.",
 344:         "",
 345:         "## 4. Panel B: QCR Primary-protocol Ablation",
 346:         "",
 347:         "| Backbone | Method | Variant | Image AUROC | AP | Best F1 | Role |",
 348:         "|---|---|---|---:|---:|---:|---|",
 349:     ]
 350: 
 351:     for _, r in qcr_table.iterrows():
 352:         lines.append(
 353:             f"| {r['backbone']} | {r['method']} | {r['variant_id']} | "
```

**`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 358–371 行**

```python
 358:     lines += [
 359:         "",
 360:         "Paper use:",
 361:         "",
 362:         "- Treat `Quality-Calibrated QCR` as the main effective method core.",
 363:         "- Treat `Quality-Calibrated QCR + adaptive consistency refinement` as the final conservative refinement.",
 364:         "- Treat `Fixed Q+C fusion` as diagnostic only, because it is not robust across protocols.",
 365:         "",
 366:         "## 5. Claim-ready Deltas",
 367:         "",
 368:         "| Scope | Comparison | Left Score | Right Score | Delta | Interpretation |",
 369:         "|---|---|---:|---:|---:|---|",
 370:     ]
 371: 
```

**`experiments/stage16_qcru_ablation/build_stage16_d_paper_facing_final_comparison.py`，第 470–482 行**

```python
 470:     print()
 471:     print("===== Panel A: system baselines =====")
 472:     print(system_table[["rank_by_mean_image_auroc", "method", "mean_image_auroc", "paper_role"]].to_string(index=False))
 473:     print()
 474:     print("===== Panel B: QCR ablation =====")
 475:     print(qcr_table[["backbone", "variant_id", "method", "image_auroc", "paper_role"]].to_string(index=False))
 476:     print()
 477:     print("===== claim-ready deltas =====")
 478:     print(deltas[["scope", "comparison", "delta", "paper_interpretation"]].to_string(index=False))
 479: 
 480: 
 481: if __name__ == "__main__":
 482:     main()
```

**`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 312–324 行**

```python
 312: 
 313:         rows.append(
 314:             select_top_cases(
 315:                 g,
 316:                 "adaptive_refinement_high_gate",
 317:                 "adaptive_gate",
 318:                 ascending=False,
 319:             )
 320:         )
 321:         rows.append(
 322:             select_top_cases(
 323:                 g,
 324:                 "detector_vlm_disagreement_boundary",
```

**`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 335–350 行**

```python
 335: 
 336: 
 337: def build_category_summary(primary: pd.DataFrame) -> pd.DataFrame:
 338:     variants = [
 339:         ("V3", "naive_detector_crop_fusion", "score_naive"),
 340:         ("V4", "Quality-Calibrated QCR", "score_quality"),
 341:         ("V5", "Fixed Q+C fusion", "score_fixed_qc"),
 342:         ("V6", "Quality-Calibrated QCR + adaptive consistency refinement", "score_adaptive"),
 343:     ]
 344: 
 345:     rows = []
 346:     group_cols = ["backbone", "dataset", "strategy", "eval_mode", "category"]
 347: 
 348:     for keys, g in primary.groupby(group_cols, dropna=False):
 349:         base_row = dict(zip(group_cols, keys))
 350: 
```

**`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 366–395 行**

```python
 366:     idx = group_cols
 367:     piv = long.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
 368:     piv.columns.name = None
 369: 
 370:     for col in ["V3", "V4", "V5", "V6"]:
 371:         if col not in piv.columns:
 372:             piv[col] = np.nan
 373: 
 374:     piv["delta_v4_quality_minus_v3_naive"] = piv["V4"] - piv["V3"]
 375:     piv["delta_v6_adaptive_minus_v4_quality"] = piv["V6"] - piv["V4"]
 376:     piv["delta_v5_fixed_minus_v4_quality"] = piv["V5"] - piv["V4"]
 377:     piv["delta_v6_adaptive_minus_v5_fixed"] = piv["V6"] - piv["V5"]
 378: 
 379:     def boundary_label(r):
 380:         labels = []
 381:         if pd.notna(r["delta_v4_quality_minus_v3_naive"]) and r["delta_v4_quality_minus_v3_naive"] <= 0:
 382:             labels.append("quality_not_helpful")
 383:         if pd.notna(r["delta_v6_adaptive_minus_v4_quality"]) and abs(r["delta_v6_adaptive_minus_v4_quality"]) < 0.001:
 384:             labels.append("adaptive_gain_negligible")
 385:         if pd.notna(r["delta_v5_fixed_minus_v4_quality"]) and r["delta_v5_fixed_minus_v4_quality"] > 0:
 386:             labels.append("fixed_consistency_can_peak_but_diagnostic")
 387:         if pd.notna(r["V6"]) and r["V6"] < 0.90:
 388:             labels.append("low_absolute_qcr_auc")
 389:         return ";".join(labels) if labels else "no_major_boundary"
 390: 
 391:     piv["boundary_label"] = piv.apply(boundary_label, axis=1)
 392: 
 393:     return piv.sort_values(
 394:         ["backbone", "delta_v4_quality_minus_v3_naive", "delta_v6_adaptive_minus_v4_quality"],
 395:         ascending=[True, True, True],
```

**`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 417–445 行**

```python
 417:     add(
 418:         "E1",
 419:         "quality_calibration",
 420:         "Keep candidate quality calibration as the main method core.",
 421:         f"Per-category mean V4-V3 AUROC delta={q_delta.mean():+.4f}; wins={(q_delta > 0).sum()}/{len(q_delta)}.",
 422:         "Use as main contribution.",
 423:     )
 424: 
 425:     add(
 426:         "E2",
 427:         "adaptive_consistency",
 428:         "Keep adaptive consistency only as a refinement.",
 429:         f"Per-category mean V6-V4 AUROC delta={a_delta.mean():+.4f}; wins={(a_delta > 0).sum()}/{len(a_delta)}.",
 430:         "Use with caution; do not call it the main source of improvement.",
 431:     )
 432: 
 433:     add(
 434:         "E3",
 435:         "fixed_consistency",
 436:         "Do not use fixed Q+C as the final method even if it peaks on some categories.",
 437:         f"Per-category mean V5-V4 AUROC delta={f_delta.mean():+.4f}; positive cases={(f_delta > 0).sum()}/{len(f_delta)}.",
 438:         "Mention as diagnostic only.",
 439:     )
 440: 
 441:     if not case_inventory.empty:
 442:         counts = case_inventory["case_type"].value_counts().to_dict()
 443:         add(
 444:             "E4",
 445:             "case_inventory",
```

**`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 471–500 行**

```python
 471:         "This stage does not train models or rerun VLM inference. It mines the existing Stage 9 prediction table for representative boundary cases.",
 472:         "",
 473:         "## 2. Primary Scope",
 474:         "",
 475:         "The case inventory uses the QCR primary protocol:",
 476:         "",
 477:         "```text",
 478:         "dataset = VisA",
 479:         "strategy = inspection_binary",
 480:         "eval_mode = crop_topk_ensemble",
 481:         "```",
 482:         "",
 483:         "## 3. Category-level Boundary Summary",
 484:         "",
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
 497:         )
 498: 
 499:     lines += [
 500:         "",
```

**`experiments/stage16_qcru_ablation/build_stage16_e_failure_boundary_analysis.py`，第 507–519 行**

```python
 507:         "| quality_boundary_anomaly_suppression | anomaly images suppressed by quality calibration | boundary / failure case |",
 508:         "| quality_boundary_normal_boost | normal images boosted by quality calibration | boundary / failure case |",
 509:         "| fixed_consistency_boundary_anomaly_suppression | anomaly images where fixed consistency hurts | explains why fixed Q+C is not final |",
 510:         "| fixed_consistency_boundary_normal_boost | normal images where fixed consistency increases risk | explains false-positive boundary |",
 511:         "| adaptive_refinement_high_gate | images with strongest adaptive gate | explains refinement behavior |",
 512:         "| detector_vlm_disagreement_boundary | images with high detector/VLM disagreement | explains detector-VLM conflict |",
 513:         "",
 514:     ]
 515: 
 516:     if case_inventory.empty:
 517:         lines.append("No case inventory generated.")
 518:     else:
 519:         counts = case_inventory["case_type"].value_counts().reset_index()
```

**`experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py`，第 100–115 行**

```python
 100:     d_loco_patch = delta_row(deltas, "LOCO fusion vs PatchCore")
 101:     d_loco_ead = delta_row(deltas, "LOCO fusion vs EfficientAD-30")
 102:     d_loco_winclip = delta_row(deltas, "LOCO fusion vs WinCLIP")
 103:     d_context_full = delta_row(deltas, "context-aware VLM vs full-image VLM")
 104:     d_quality_naive = delta_row(deltas, "Quality-Calibrated QCR vs naive fusion")
 105:     d_adaptive_quality = delta_row(deltas, "Adaptive refinement vs Quality-Calibrated QCR")
 106:     d_adaptive_naive = delta_row(deltas, "Adaptive refinement vs naive fusion")
 107:     d_fixed_quality = delta_row(deltas, "Fixed Q+C vs Quality-Calibrated QCR")
 108: 
 109:     e_quality = boundary_decision(boundary, "E1")
 110:     e_adaptive = boundary_decision(boundary, "E2")
 111:     e_fixed = boundary_decision(boundary, "E3")
 112:     e_cases = boundary_decision(boundary, "E4")
 113:     e_boundary = boundary_decision(boundary, "E5")
 114: 
 115:     cat_stats = compute_category_stats(category)
```

**`experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py`，第 171–183 行**

```python
 171:             "allowed_wording": "Candidate quality calibration provides the main gain over naive detector-crop fusion.",
 172:             "forbidden_wording": "Every category benefits from quality calibration.",
 173:             "evidence_files": "stage16_d_paper_facing_claim_ready_deltas.csv; stage16_e_boundary_decision_summary.csv; stage16_e_category_boundary_summary.csv",
 174:             "evidence_summary": (
 175:                 f"primary QCR quality-minus-naive delta={fmt(d_quality_naive.get('delta', None))}; "
 176:                 f"{e_quality.get('evidence', '')}; "
 177:                 f"per-category mean={fmt(cat_stats.get('quality_minus_naive_mean'))}, "
 178:                 f"wins={cat_stats.get('quality_minus_naive_wins')}/{cat_stats.get('quality_minus_naive_total')}."
 179:             ),
 180:             "support_level": "strong_as_core_but_not_universal",
 181:             "paper_section": "Method; Ablation",
 182:             "caveat": "Per-category wins are not universal; use boundary-aware wording.",
 183:             "status": "use",
```

**`experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py`，第 289–301 行**

```python
 289:             {
 290:                 "status_group": "paper_ready_method_name",
 291:                 "num_claims": 1,
 292:                 "claim_ids": "P4;P5;P6",
 293:                 "summary": "Use Quality-Calibrated QCR as the method family; adaptive consistency is refinement; fixed Q+C is diagnostic only.",
 294:             },
 295:             {
 296:                 "status_group": "remaining_experiment_risks",
 297:                 "num_claims": 3,
 298:                 "claim_ids": "R1;R2;R3",
 299:                 "summary": "EfficientAD remains fixed-budget; AnomalyCLIP is absent; representative failure figures still need manual visual inspection.",
 300:             },
 301:             {
```

**`experiments/stage16_qcru_ablation/build_stage16_f_final_claim_evidence_map.py`，第 345–372 行**

```python
 345:         "",
 346:         "Use this method family name:",
 347:         "",
 348:         "```text",
 349:         "Quality-Calibrated QCR",
 350:         "```",
 351:         "",
 352:         "Use this longer descriptive phrase when needed:",
 353:         "",
 354:         "```text",
 355:         "Quality-Calibrated Localization-Guided VLM Reasoning",
 356:         "```",
 357:         "",
 358:         "Use this only as the full variant name:",
 359:         "",
 360:         "```text",
 361:         "Quality-Calibrated QCR with adaptive consistency refinement",
 362:         "```",
 363:         "",
 364:         "Do not write the method as fixed Q+C QCR-U.",
 365:         "",
 366:         "## 3. Claim-Evidence Map",
 367:         "",
 368:         "| Claim ID | Category | Paper Claim | Support | Status | Section |",
 369:         "|---|---|---|---|---|---|",
 370:     ]
 371: 
 372:     for _, r in claim_map.iterrows():
```

**`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 252–266 行**

```python
 252: def make_variant_long(base: pd.DataFrame) -> pd.DataFrame:
 253:     variants = [
 254:         ("V0", "detector_only", "score_detector_only", False, False),
 255:         ("V2", "crop_topk_vlm", "score_crop_topk_vlm", False, False),
 256:         ("V3", "naive_detector_crop_fusion", "score_naive_detector_crop_fusion", False, False),
 257:         ("V4", "quality_weighted_crop", "score_quality_weighted_crop", True, False),
 258:         ("V5", "quality_consistency_fusion", "score_quality_consistency_fusion", True, True),
 259:     ]
 260: 
 261:     rows = []
 262:     id_cols = [
 263:         "backbone",
 264:         "dataset",
 265:         "category",
 266:         "strategy",
```

**`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 304–323 行**

```python
 304: 
 305:     per_config = pd.DataFrame(per_config_rows)
 306:     per_category = pd.DataFrame(per_category_rows)
 307: 
 308:     # Choose best fixed-protocol row for each backbone/dataset/strategy/eval_mode by V5 AUROC,
 309:     # but report every variant in that same protocol. This avoids cherry-picking variant formulas.
 310:     best_rows = []
 311:     for keys, g in per_config.groupby(["backbone", "dataset", "strategy", "eval_mode"], dropna=False):
 312:         v5 = g[g["variant"] == "quality_consistency_fusion"]
 313:         if v5.empty:
 314:             continue
 315:         row = v5.sort_values("auroc", ascending=False).iloc[0].copy()
 316:         best_rows.append(row)
 317: 
 318:     best_protocol = pd.DataFrame(best_rows)
 319:     if not best_protocol.empty:
 320:         best_protocol = best_protocol.sort_values("auroc", ascending=False).reset_index(drop=True)
 321:         best_protocol["rank_by_v5_auroc"] = range(1, len(best_protocol) + 1)
 322: 
 323:     return per_config, per_category, best_protocol
```

**`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 330–346 行**

```python
 330:     best_protocol: pd.DataFrame,
 331: ) -> None:
 332:     lines = []
 333:     lines += [
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
 346:         "",
```

**`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 359–371 行**

```python
 359:         "Where `D` is detector score, `M` is crop VLM abnormal score, `Q` is candidate quality, and `K` is detector-VLM high-high consistency.",
 360:         "",
 361:         "## 4. Best Protocols by Q+C Fusion AUROC",
 362:         "",
 363:         "| Rank | Backbone | Dataset | Strategy | Eval Mode | V5 AUROC | V5 AP | V5 Best F1 |",
 364:         "|---:|---|---|---|---|---:|---:|---:|",
 365:     ]
 366: 
 367:     if best_protocol.empty:
 368:         lines.append("| - | - | - | - | - | - | - | - |")
 369:     else:
 370:         for _, r in best_protocol.head(20).iterrows():
 371:             lines.append(
```

**`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 390–402 行**

```python
 390:         )
 391:         comp = per_config[mask].sort_values("variant_id")
 392: 
 393:         lines += [
 394:             f"Best protocol by V5 AUROC: `{best['backbone']} / {best['dataset']} / {best['strategy']} / {best['eval_mode']}`.",
 395:             "",
 396:             "| Variant | AUROC | AP | Best F1 | Best Accuracy |",
 397:             "|---|---:|---:|---:|---:|",
 398:         ]
 399: 
 400:         for _, r in comp.iterrows():
 401:             lines.append(
 402:                 f"| {r['variant']} | {r['auroc']:.4f} | {r['ap']:.4f} | "
```

**`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 430–442 行**

```python
 430:         "## 6. Interpretation Rules",
 431:         "",
 432:         "This stage is diagnostic. A positive result only means fixed Q+C evidence is useful under the existing Stage 9 signals.",
 433:         "",
 434:         "It is not yet the final QCR-U method unless:",
 435:         "",
 436:         "1. Q+C improves over naive fusion consistently, not only in one protocol.",
 437:         "2. The selected protocol is justified without test-set tuning.",
 438:         "3. Per-category results do not collapse on one or more primary categories.",
 439:         "",
 440:         "## 7. Outputs",
 441:         "",
 442:         f"- `{OUT_PER_CONFIG.relative_to(ROOT)}`",
```

**`experiments/stage16_qcru_ablation/run_stage16_a1_qcru_fixed_protocol_ablation.py`，第 468–480 行**

```python
 468:     print("[DONE]", OUT_PER_CATEGORY)
 469:     print("[DONE]", OUT_BEST)
 470:     print("[DONE]", OUT_REPORT)
 471:     print()
 472:     print("===== top protocols by V5 quality_consistency_fusion AUROC =====")
 473:     if best_protocol.empty:
 474:         print("EMPTY")
 475:     else:
 476:         print(
 477:             best_protocol[
 478:                 [
 479:                     "rank_by_v5_auroc",
 480:                     "backbone",
```

**`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 200–212 行**

```python
 200:     base["score_naive"] = 0.5 * base["D"] + 0.5 * base["M"]
 201:     base["score_quality_raw"] = 0.5 * base["D"] + 0.5 * (base["M"] * (0.5 + 0.5 * base["Q"]))
 202:     base["score_fixed_qc_raw"] = 0.40 * base["D"] + 0.40 * base["M"] + 0.10 * base["Q"] + 0.10 * base["K"]
 203: 
 204:     # Adaptive QCR-U:
 205:     # Start from quality-weighted core.
 206:     # Add a conservative consistency bonus only when:
 207:     # - candidate quality is high,
 208:     # - detector and VLM agree,
 209:     # - both detector and VLM provide high anomaly evidence.
 210:     #
 211:     # This is label-free and intentionally conservative.
 212:     agreement = 1.0 - (base["D"] - base["M"]).abs()
```

**`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 232–247 行**

```python
 232: 
 233: 
 234: def make_long(base: pd.DataFrame) -> pd.DataFrame:
 235:     variants = [
 236:         ("V3", "naive_detector_crop_fusion", "score_naive"),
 237:         ("V4", "quality_weighted_crop", "score_quality"),
 238:         ("V5", "fixed_quality_consistency", "score_fixed_qc"),
 239:         ("V6", "adaptive_qcru", "score_adaptive_qcru"),
 240:     ]
 241: 
 242:     id_cols = [
 243:         "backbone",
 244:         "dataset",
 245:         "category",
 246:         "strategy",
 247:         "eval_mode",
```

**`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 277–311 行**

```python
 277:     idx = ["backbone", "dataset", "strategy", "eval_mode"]
 278:     piv = per_config.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
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
 291:     piv["v6_beats_quality"] = piv["delta_v6_minus_v4_quality"] > 0
 292:     piv["v6_beats_fixed_qc"] = piv["delta_v6_minus_v5_fixed_qc"] > 0
 293:     piv["v5_beats_quality"] = piv["delta_v5_minus_v4_quality"] > 0
 294: 
 295:     return piv
 296: 
 297: 
 298: def summarize_delta(delta: pd.DataFrame) -> pd.DataFrame:
 299:     checks = [
 300:         ("V6 > V3 naive", "v6_beats_naive", "delta_v6_minus_v3_naive"),
 301:         ("V6 > V4 quality", "v6_beats_quality", "delta_v6_minus_v4_quality"),
 302:         ("V6 > V5 fixed Q+C", "v6_beats_fixed_qc", "delta_v6_minus_v5_fixed_qc"),
 303:         ("V5 > V4 quality", "v5_beats_quality", "delta_v5_minus_v4_quality"),
 304:     ]
 305: 
 306:     rows = []
 307:     for name, win_col, delta_col in checks:
 308:         rows.append(
 309:             {
 310:                 "check": name,
 311:                 "wins": int(delta[win_col].sum()),
```

**`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 320–345 行**

```python
 320: 
 321:     for eval_mode, g in delta.groupby("eval_mode"):
 322:         rows.append(
 323:             {
 324:                 "check": f"V6 > V4 quality by eval_mode={eval_mode}",
 325:                 "wins": int(g["v6_beats_quality"].sum()),
 326:                 "total_protocols": int(len(g)),
 327:                 "win_rate": float(g["v6_beats_quality"].mean()),
 328:                 "mean_delta": float(g["delta_v6_minus_v4_quality"].mean()),
 329:                 "median_delta": float(g["delta_v6_minus_v4_quality"].median()),
 330:                 "min_delta": float(g["delta_v6_minus_v4_quality"].min()),
 331:                 "max_delta": float(g["delta_v6_minus_v4_quality"].max()),
 332:             }
 333:         )
 334: 
 335:         rows.append(
 336:             {
 337:                 "check": f"V6 > V5 fixed Q+C by eval_mode={eval_mode}",
 338:                 "wins": int(g["v6_beats_fixed_qc"].sum()),
 339:                 "total_protocols": int(len(g)),
 340:                 "win_rate": float(g["v6_beats_fixed_qc"].mean()),
 341:                 "mean_delta": float(g["delta_v6_minus_v5_fixed_qc"].mean()),
 342:                 "median_delta": float(g["delta_v6_minus_v5_fixed_qc"].median()),
 343:                 "min_delta": float(g["delta_v6_minus_v5_fixed_qc"].min()),
 344:                 "max_delta": float(g["delta_v6_minus_v5_fixed_qc"].max()),
 345:             }
```

**`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 348–371 行**

```python
 348:     return pd.DataFrame(rows)
 349: 
 350: 
 351: def write_report(per_config: pd.DataFrame, delta: pd.DataFrame, summary: pd.DataFrame) -> None:
 352:     best = per_config[per_config["variant_id"] == "V6"].sort_values("auroc", ascending=False).reset_index(drop=True)
 353:     best["rank_by_v6_auroc"] = range(1, len(best) + 1)
 354: 
 355:     lines = []
 356:     lines += [
 357:         "# Stage 16-A3 Adaptive QCR-U",
 358:         "",
 359:         "## 1. Purpose",
 360:         "",
 361:         "Stage 16-A2 showed that candidate quality is stable, while fixed consistency is not universally beneficial.",
 362:         "",
 363:         "This stage tests an adaptive QCR-U score that uses quality-weighted crop scoring as the stable core and applies consistency only as a conservative reliability-gated bonus.",
 364:         "",
 365:         "## 2. Formula",
 366:         "",
 367:         "```text",
 368:         "D = detector anomaly score",
 369:         "M = crop VLM anomaly score",
 370:         "Q = candidate quality",
 371:         "K = high-high detector/VLM consistency",
```

**`experiments/stage16_qcru_ablation/run_stage16_a3_adaptive_qcru.py`，第 393–441 行**

```python
 393:         )
 394: 
 395:     lines += [
 396:         "",
 397:         "## 4. Adaptive QCR-U Protocol Ranking",
 398:         "",
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
 411:         "## 5. Protocol-level Delta Table",
 412:         "",
 413:         "| Backbone | Strategy | Eval Mode | V3 | V4 | V5 | V6 | V6-V3 | V6-V4 | V6-V5 |",
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
 426: 
 427:     lines += [
 428:         "",
 429:         "## 6. Decision Rule",
 430:         "",
 431:         "If adaptive QCR-U beats naive fusion consistently and avoids the full_all degradation of fixed Q+C, it can replace fixed Q+C as the next method candidate.",
 432:         "",
 433:         "If adaptive QCR-U still fails to beat quality-only, the method should be simplified to quality-weighted crop fusion and consistency should be moved to analysis rather than method.",
 434:         "",
 435:         "## 7. Outputs",
 436:         "",
 437:         f"- `{OUT_PER_CONFIG.relative_to(ROOT)}`",
 438:         f"- `{OUT_DELTA.relative_to(ROOT)}`",
 439:         f"- `{OUT_FAILURES.relative_to(ROOT)}`",
 440:         f"- `{OUT_DOC.relative_to(ROOT)}`",
 441:         "",
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_a0_ad2_qcr_inventory.py`，第 103–132 行**

```python
 103:                 "ad2_categories_found": ";".join(ad2_found),
 104:                 "ad2_coverage_count": len(ad2_found),
 105:                 "can_directly_run_ad2_qcr_ablation": can_direct,
 106:                 "notes": (
 107:                     "contains AD2 QCR-ready predictions"
 108:                     if can_direct
 109:                     else "does not contain full AD2 QCR-ready prediction columns/categories"
 110:                 ),
 111:             }
 112:         )
 113: 
 114:     out = pd.DataFrame(rows)
 115:     out.to_csv(OUT_CSV, index=False, lineterminator="\n")
 116: 
 117:     can_any = bool(out["can_directly_run_ad2_qcr_ablation"].any())
 118: 
 119:     lines = [
 120:         "# Stage 18-A0 AD2 QCR Inventory",
 121:         "",
 122:         "## Purpose",
 123:         "",
 124:         "Check whether existing QCR prediction/result files already contain AD2 four-category data for:",
 125:         "",
 126:         "```text",
 127:         "fruit_jelly",
 128:         "sheet_metal",
 129:         "vial",
 130:         "walnuts",
 131:         "```",
 132:         "",
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_a0_ad2_qcr_inventory.py`，第 154–172 行**

```python
 154:     ]
 155: 
 156:     if can_any:
 157:         lines += [
 158:             "Proceed to Stage 18-A1: run AD2 four-category QCR ablation directly from existing prediction file.",
 159:         ]
 160:     else:
 161:         lines += [
 162:             "Proceed to Stage 18-B: generate missing AD2 QCR prediction file first.",
 163:             "",
 164:             "This means the current QCR ablation evidence is not yet aligned with the AD2 four-category system-level baseline.",
 165:         ]
 166: 
 167:     OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
 168: 
 169:     print("[DONE]", OUT_CSV)
 170:     print("[DONE]", OUT_REPORT)
 171:     print()
 172:     print(out.to_string(index=False))
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b0_ad2_qcr_source_inventory.py`，第 332–353 行**

```python
 332:     medium = int((candidates["qcr_assembly_value"] == "medium").sum()) if not candidates.empty else 0
 333:     low = int((candidates["qcr_assembly_value"] == "low").sum()) if not candidates.empty else 0
 334: 
 335:     lines = [
 336:         "# Stage 18-B0 AD2 QCR Source Inventory",
 337:         "",
 338:         "## Purpose",
 339:         "",
 340:         "Scan existing result/run files to determine whether AD2 four-category QCR predictions can be assembled from existing per-image sources.",
 341:         "",
 342:         "## Summary",
 343:         "",
 344:         f"- scanned files: `{len(files)}`",
 345:         f"- AD2 high-value QCR-ready/near-ready files: `{high}`",
 346:         f"- AD2 medium-value partial per-image files: `{medium}`",
 347:         f"- AD2 low-value summary/category-level files: `{low}`",
 348:         "",
 349:         "## Candidate files",
 350:         "",
 351:         "| File | Coverage | Role | Value | Image ID | Label | Detector | VLM | Quality | Notes |",
 352:         "|---|---:|---|---|---|---|---|---|---|---|",
 353:     ]
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b0_ad2_qcr_source_inventory.py`，第 367–381 行**

```python
 367:     lines += [
 368:         "",
 369:         "## Decision rule",
 370:         "",
 371:         "- If high-value files exist, proceed to Stage 18-B1: assemble AD2 QCR predictions from existing sources.",
 372:         "- If only medium-value files exist, inspect whether detector/VLM/quality sources can be joined by image key.",
 373:         "- If no high/medium-value files exist, proceed to Stage 18-C: generate AD2 QCR predictions from scratch.",
 374:         "",
 375:     ]
 376: 
 377:     if high > 0:
 378:         decision = "proceed_to_stage18_b1_assemble_existing_sources"
 379:     elif medium > 0:
 380:         decision = "inspect_medium_sources_then_assemble_or_generate_missing_parts"
 381:     else:
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b1_ad2_qcr_schema_profile.py`，第 370–386 行**

```python
 370:     qcr_ready = profile[profile["qcr_readiness"] == "qcr_ready"]
 371:     partial = profile[profile["qcr_readiness"] == "partial_join_source"]
 372: 
 373:     lines = [
 374:         "# Stage 18-B1 AD2 QCR Source Schema Profile",
 375:         "",
 376:         "## Purpose",
 377:         "",
 378:         "Inspect Stage11/Stage13 AD2 source files to decide whether AD2 four-category QCR predictions can be assembled from existing files.",
 379:         "",
 380:         "## Summary",
 381:         "",
 382:         f"- qcr_ready files: `{len(qcr_ready)}`",
 383:         f"- partial_join_source files: `{len(partial)}`",
 384:         "",
 385:         "## File profile",
 386:         "",
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b1_ad2_qcr_schema_profile.py`，第 419–431 行**

```python
 419:         "## Decision rule",
 420:         "",
 421:         "- If a qcr_ready file exists, proceed to Stage 18-B2 directly.",
 422:         "- If partial files have strong joins and contain D/M/Q/label across files, assemble in Stage 18-B2.",
 423:         "- If VLM or quality is missing, proceed to Stage 18-C to generate missing AD2 QCR predictions.",
 424:         "",
 425:     ]
 426: 
 427:     if len(qcr_ready) > 0:
 428:         decision = "proceed_to_stage18_b2_direct_qcr_assembly"
 429:     elif len(partial) > 0:
 430:         decision = "inspect_columns_and_attempt_stage18_b2_partial_join"
 431:     else:
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b4_ad2_qcr_claim_safe_decision.py`，第 118–135 行**

```python
 118:     ).reset_index(drop=True)
 119: 
 120:     all_valid_ranked.to_csv(OUT_VALID, index=False, lineterminator="\n")
 121: 
 122:     b2_v3 = b2_summary[b2_summary["variant_id"] == "V3"].iloc[0]
 123:     b2_v4 = b2_summary[b2_summary["variant_id"] == "V4"].iloc[0]
 124:     b2_v6 = b2_summary[b2_summary["variant_id"] == "V6"].iloc[0]
 125: 
 126:     b2_v4_delta = b2_deltas[
 127:         b2_deltas["comparison"] == "Quality-Calibrated QCR vs naive fusion"
 128:     ].iloc[0]["delta_a_minus_b"]
 129: 
 130:     b2_v6_delta = b2_deltas[
 131:         b2_deltas["comparison"] == "Adaptive refinement vs naive fusion"
 132:     ].iloc[0]["delta_a_minus_b"]
 133: 
 134:     decision_rows = []
 135: 
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b4_ad2_qcr_claim_safe_decision.py`，第 148–160 行**

```python
 148:             "delta_V6_minus_V3": float(b2_v6_delta),
 149:             "wins_V4_over_V3": np.nan,
 150:             "worst_category": "",
 151:             "worst_category_delta_V4_minus_V3": np.nan,
 152:             "decision": "Do not use as main AD2 QCR evidence because V4 is below V3.",
 153:         }
 154:     )
 155: 
 156:     # 2. Invalid best source.
 157:     invalid_best = b3.iloc[0]
 158:     decision_rows.append(
 159:         {
 160:             "case_id": "B3_best_overall_invalid_as_Q",
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b4_ad2_qcr_claim_safe_decision.py`，第 247–259 行**

```python
 247:         final_status = "ad2_qcr_supporting_sensitivity_not_main_claim_yet"
 248:         recommended_q = f"{stable['q_source']} / {stable['q_direction']}"
 249: 
 250:     lines = [
 251:         "# Stage 18-B4 AD2 QCR Claim-safe Decision",
 252:         "",
 253:         "## Purpose",
 254:         "",
 255:         "Convert the AD2 Q-source sweep into a claim-safe decision table.",
 256:         "",
 257:         "## Key decision",
 258:         "",
 259:         f"- final_status: `{final_status}`",
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b4_ad2_qcr_claim_safe_decision.py`，第 264–276 行**

```python
 264:         "The top overall source in B3 is `full_image_score/direct`, but this is full-image VLM evidence, not candidate quality. It must not be used as Q in a candidate-quality calibration claim.",
 265:         "",
 266:         "## Claim-safe cases",
 267:         "",
 268:         "| Case | Status | Q source | Direction | V3 | V4 | V6 | V4-V3 | Wins | Worst category |",
 269:         "|---|---|---|---|---:|---:|---:|---:|---:|---|",
 270:     ]
 271: 
 272:     for _, r in decision.iterrows():
 273:         wins = "NA" if pd.isna(r["wins_V4_over_V3"]) else str(int(r["wins_V4_over_V3"]))
 274:         lines.append(
 275:             f"| {r['case_id']} | {r['paper_status']} | {r['q_source']} | {r['q_direction']} | "
 276:             f"{fmt(r['mean_auroc_V3_naive'])} | {fmt(r['mean_auroc_V4_quality'])} | "
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b4_ad2_qcr_claim_safe_decision.py`，第 281–298 行**

```python
 281:     lines += [
 282:         "",
 283:         "## Paper recommendation",
 284:         "",
 285:         "Use AD2 four-category QCR as a source-sensitivity/boundary-supporting result unless the new Q definition is formally locked and rerun consistently across the main VisA ablation.",
 286:         "",
 287:         "Recommended wording:",
 288:         "",
 289:         "```text",
 290:         "On the AD2 four-category setting, the default transferred Q source is not uniformly beneficial. A non-GT candidate-region score source recovers a positive mean gain over naive detector-crop fusion, but we report this as a candidate-quality source sensitivity rather than as the primary QCR claim.",
 291:         "```",
 292:         "",
 293:         "## Outputs",
 294:         "",
 295:         f"- `{OUT_VALID.relative_to(ROOT)}`",
 296:         f"- `{OUT_DECISION.relative_to(ROOT)}`",
 297:         f"- `{OUT_TABLE.relative_to(ROOT)}`",
 298:         "",
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b7_qcr_final_claim_update.py`，第 62–81 行**

```python
  62:     quality_mean = float(s["mean_test_quality_qcr"])
  63:     adaptive_mean = float(s["mean_test_adaptive_qcr"])
  64: 
  65:     if quality_mean >= adaptive_mean:
  66:         final_variant = "Quality-Calibrated QCR"
  67:         final_score = quality_mean
  68:         final_delta = float(s["mean_delta_quality_minus_V3"])
  69:         final_wins = int(s["wins_quality_over_V3"])
  70:         final_worst_delta = float(s["worst_quality_delta"])
  71:         final_note = "Quality-only calibration is selected because it is slightly stronger than adaptive refinement on AD2."
  72:     else:
  73:         final_variant = "Quality-Calibrated QCR + adaptive refinement"
  74:         final_score = adaptive_mean
  75:         final_delta = float(s["mean_delta_adaptive_minus_V3"])
  76:         final_wins = int(s["wins_adaptive_over_V3"])
  77:         final_worst_delta = float(s["worst_adaptive_delta"])
  78:         final_note = "Adaptive refinement is selected because it is stronger than quality-only calibration on AD2."
  79: 
  80:     final_table = pd.DataFrame(
  81:         [
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b7_qcr_final_claim_update.py`，第 88–108 行**

```python
  88:                 "paper_role": "baseline",
  89:             },
  90:             {
  91:                 "setting": "AD2 four-category LOCO policy",
  92:                 "method": "Quality-Calibrated QCR",
  93:                 "mean_image_auroc": quality_mean,
  94:                 "delta_vs_naive": float(s["mean_delta_quality_minus_V3"]),
  95:                 "wins_vs_naive": f"{int(s['wins_quality_over_V3'])}/4",
  96:                 "paper_role": "main_qcr_support",
  97:             },
  98:             {
  99:                 "setting": "AD2 four-category LOCO policy",
 100:                 "method": "Quality-Calibrated QCR + adaptive refinement",
 101:                 "mean_image_auroc": adaptive_mean,
 102:                 "delta_vs_naive": float(s["mean_delta_adaptive_minus_V3"]),
 103:                 "wins_vs_naive": f"{int(s['wins_adaptive_over_V3'])}/4",
 104:                 "paper_role": "auxiliary_refinement",
 105:             },
 106:         ]
 107:     )
 108: 
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b7_qcr_final_claim_update.py`，第 128–154 行**

```python
 128:     claim_rows = [
 129:         {
 130:             "claim_item": "main_innovation",
 131:             "decision": "keep",
 132:             "final_wording": "Quality-Calibrated QCR: candidate-quality calibration for localization-guided VLM anomaly evidence.",
 133:             "evidence": "VisA primary ablation plus AD2 four-category LOCO policy support.",
 134:         },
 135:         {
 136:             "claim_item": "adaptive_refinement",
 137:             "decision": "downgrade_to_auxiliary",
 138:             "final_wording": "Adaptive consistency refinement is an optional auxiliary refinement, not the primary source of the gain.",
 139:             "evidence": "On AD2, quality-only QCR slightly exceeds adaptive QCR.",
 140:         },
 141:         {
 142:             "claim_item": "ad2_qcr_support",
 143:             "decision": "supporting_cross_category_evidence",
 144:             "final_wording": (
 145:                 "With a fixed semantic candidate-quality source and LOCO-selected weights, "
 146:                 "QCR improves over naive detector-crop fusion on AD2 four-category evaluation."
 147:             ),
 148:             "evidence": f"{final_variant}: {fmt(final_score)} AUROC, {signed(final_delta)} over naive, wins {final_wins}/4.",
 149:         },
 150:         {
 151:             "claim_item": "overclaim_to_avoid",
 152:             "decision": "avoid",
 153:             "final_wording": "Do not claim that all Q sources or adaptive consistency are universally beneficial.",
 154:             "evidence": "Default transferred Q source and several robust selectors did not improve over naive on AD2.",
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b7_qcr_final_claim_update.py`，第 158–174 行**

```python
 158:     claim_update = pd.DataFrame(claim_rows)
 159:     claim_update.to_csv(OUT_CLAIM_UPDATE, index=False, lineterminator="\n")
 160: 
 161:     lines = [
 162:         "# Stage 18-B7 QCR Final Claim Update",
 163:         "",
 164:         "## Decision",
 165:         "",
 166:         "QCR remains viable as the main innovation, but the claim must focus on candidate-quality calibration rather than adaptive refinement.",
 167:         "",
 168:         "## Locked AD2 support setting",
 169:         "",
 170:         f"- locked selector: `{LOCKED_SELECTOR}`",
 171:         "- Q source: `candidate_score_max_mean`",
 172:         "- Q direction: `inverted`",
 173:         "- protocol: AD2 four-category leave-one-category-out policy selection",
 174:         "",
```

**`experiments/stage18_ad2_qcr_ablation/build_stage18_b7_qcr_final_claim_update.py`，第 187–222 行**

```python
 187:     lines += [
 188:         "",
 189:         "## Final method choice",
 190:         "",
 191:         f"- selected AD2-facing QCR variant: `{final_variant}`",
 192:         f"- selected AUROC: `{fmt(final_score)}`",
 193:         f"- selected delta vs naive: `{signed(final_delta)}`",
 194:         f"- selected wins vs naive: `{final_wins}/4`",
 195:         f"- worst category delta: `{signed(final_worst_delta)}`",
 196:         f"- note: {final_note}",
 197:         "",
 198:         "## Paper wording",
 199:         "",
 200:         "Use this wording:",
 201:         "",
 202:         "```text",
 203:         "The main contribution is Quality-Calibrated QCR, which calibrates localization-guided crop-level VLM evidence using candidate-region quality. On VisA, the controlled ablation shows consistent gains over naive detector-crop fusion. On the AD2 four-category setting, a fixed semantic candidate-quality source with leave-one-category-out policy selection improves over naive fusion, providing supporting cross-category evidence. Adaptive consistency refinement is retained only as an auxiliary analysis rather than the primary source of improvement.",
 204:         "```",
 205:         "",
 206:         "Avoid this wording:",
 207:         "",
 208:         "```text",
 209:         "Adaptive QCR is universally beneficial across all datasets and all candidate-quality sources.",
 210:         "```",
 211:         "",
 212:         "## Fold details",
 213:         "",
 214:         "| Held-out | Q source | Direction | eta | gamma | V3 | Quality QCR | Adaptive QCR | Quality-V3 | Adaptive-V3 |",
 215:         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
 216:     ]
 217: 
 218:     for _, r in fold_table.iterrows():
 219:         lines.append(
 220:             f"| {r['heldout_category']} | {r['selected_q_source']} | {r['selected_q_direction']} | "
 221:             f"{float(r['selected_eta']):.2f} | {float(r['selected_gamma']):.2f} | "
 222:             f"{fmt(r['test_V3'])} | {fmt(r['test_quality_qcr'])} | {fmt(r['test_adaptive_qcr'])} | "
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 192–212 行**

```python
 192:     df["V0_detector_only"] = df["D"]
 193:     df["V1_full_image_vlm"] = df["F"]
 194:     df["V2_crop_topk_vlm"] = df["M"]
 195:     df["V3_naive_detector_crop_fusion"] = 0.5 * df["D"] + 0.5 * df["M"]
 196:     df["V4_quality_calibrated_qcr"] = 0.5 * df["D"] + 0.5 * df["M"] * (0.5 + 0.5 * df["Q"])
 197:     df["V5_fixed_qc_diagnostic"] = 0.4 * df["D"] + 0.4 * df["M"] + 0.1 * df["Q"] + 0.1 * df["K"]
 198:     df["V6_adaptive_qcr_refinement"] = df["V4_quality_calibrated_qcr"] + 0.05 * df["adaptive_gate"]
 199: 
 200:     df["stage18_m_score_source"] = m_source
 201:     df["stage18_q_score_source"] = q_source
 202:     df["stage18_full_image_source"] = full_source
 203:     df["stage18_note"] = (
 204:         "AD2 QCR ablation assembled from Stage11 image-level VLM predictions and "
 205:         "candidate-level non-GT quality evidence."
 206:     )
 207: 
 208:     keep_cols = [
 209:         "category",
 210:         "image_path",
 211:         "gt_binary",
 212:         "D_raw_patchcore",
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 224–236 行**

```python
 224:         "V0_detector_only",
 225:         "V1_full_image_vlm",
 226:         "V2_crop_topk_vlm",
 227:         "V3_naive_detector_crop_fusion",
 228:         "V4_quality_calibrated_qcr",
 229:         "V5_fixed_qc_diagnostic",
 230:         "V6_adaptive_qcr_refinement",
 231:         "num_candidates",
 232:         "stage18_m_score_source",
 233:         "stage18_q_score_source",
 234:         "stage18_full_image_source",
 235:         "stage18_note",
 236:     ]
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 243–258 行**

```python
 243:     variants = [
 244:         ("V0", "Detector only", "V0_detector_only", "baseline_detector"),
 245:         ("V1", "Full-image VLM", "V1_full_image_vlm", "baseline_vlm"),
 246:         ("V2", "Crop top-k VLM", "V2_crop_topk_vlm", "baseline_crop_vlm"),
 247:         ("V3", "Naive detector-crop fusion", "V3_naive_detector_crop_fusion", "fusion_baseline"),
 248:         ("V4", "Quality-Calibrated QCR", "V4_quality_calibrated_qcr", "main_method_core"),
 249:         ("V5", "Fixed Q+C fusion", "V5_fixed_qc_diagnostic", "diagnostic_not_final"),
 250:         ("V6", "Quality-Calibrated QCR + adaptive refinement", "V6_adaptive_qcr_refinement", "final_refinement"),
 251:     ]
 252: 
 253:     rows = []
 254: 
 255:     for cat, sub in pred.groupby("category"):
 256:         y = pd.to_numeric(sub["gt_binary"], errors="coerce").astype(int)
 257: 
 258:         for vid, method, col, role in variants:
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 315–332 行**

```python
 315: 
 316:     summary = pd.DataFrame(summary_rows)
 317: 
 318:     delta_pairs = [
 319:         ("V4", "V3", "Quality-Calibrated QCR vs naive fusion"),
 320:         ("V6", "V4", "Adaptive refinement vs Quality-Calibrated QCR"),
 321:         ("V6", "V3", "Adaptive refinement vs naive fusion"),
 322:         ("V5", "V4", "Fixed Q+C diagnostic vs Quality-Calibrated QCR"),
 323:         ("V4", "V0", "Quality-Calibrated QCR vs detector only"),
 324:         ("V4", "V2", "Quality-Calibrated QCR vs crop top-k VLM"),
 325:     ]
 326: 
 327:     delta_rows = []
 328:     for a, b, name in delta_pairs:
 329:         ra = summary[summary["variant_id"] == a]
 330:         rb = summary[summary["variant_id"] == b]
 331:         if ra.empty or rb.empty:
 332:             continue
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 360–378 行**

```python
 360: 
 361: 
 362: def write_report(pred: pd.DataFrame, per_cat: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
 363:     lines = [
 364:         "# Stage 18-B2 AD2 Four-category QCR Ablation",
 365:         "",
 366:         "## Purpose",
 367:         "",
 368:         "Assemble AD2 four-category QCR ablation from existing Stage11 image-level VLM predictions and candidate-level quality evidence.",
 369:         "",
 370:         "This aligns the QCR ablation with the AD2 four-category system-level baseline setting.",
 371:         "",
 372:         "## Data",
 373:         "",
 374:         f"- input image-level predictions: `{IN_IMAGE.relative_to(ROOT)}`",
 375:         f"- input candidate scores: `{IN_CAND.relative_to(ROOT)}`",
 376:         f"- assembled images: `{len(pred)}`",
 377:         f"- categories: `{'; '.join(sorted(pred['category'].unique()))}`",
 378:         "- detector evidence `D`: normalized `patchcore_score`",
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b2_ad2_qcr_ablation.py`，第 423–437 行**

```python
 423:     lines += [
 424:         "",
 425:         "## Interpretation rules",
 426:         "",
 427:         "- If V4 improves over V3, AD2 supports candidate quality calibration.",
 428:         "- If V6 only slightly improves over V4, keep adaptive consistency as refinement.",
 429:         "- If V5 is strong but unstable or not selected, keep fixed Q+C as diagnostic.",
 430:         "- Do not use this table to claim pixel-level segmentation SOTA.",
 431:         "",
 432:         "## Outputs",
 433:         "",
 434:         f"- `{OUT_PRED.relative_to(ROOT)}`",
 435:         f"- `{OUT_PER_CATEGORY.relative_to(ROOT)}`",
 436:         f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
 437:         f"- `{OUT_DELTAS.relative_to(ROOT)}`",
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b3_ad2_q_source_sweep.py`，第 239–270 行**

```python
 239:         "# Stage 18-B3 AD2 Q Source Sweep",
 240:         "",
 241:         "## Purpose",
 242:         "",
 243:         "Diagnose whether the Stage 18-B2 AD2 QCR drop against naive fusion is caused by the selected candidate quality source.",
 244:         "",
 245:         "The sweep keeps the same QCR formulas as the current paper method and only changes the non-GT candidate quality source.",
 246:         "",
 247:         "## Best ranked source",
 248:         "",
 249:         f"- q_source: `{best['q_source']}`",
 250:         f"- q_direction: `{best['q_direction']}`",
 251:         f"- mean V3 AUROC: `{best['mean_auroc_V3_naive']:.4f}`",
 252:         f"- mean V4 AUROC: `{best['mean_auroc_V4_quality']:.4f}`",
 253:         f"- mean V6 AUROC: `{best['mean_auroc_V6_adaptive']:.4f}`",
 254:         f"- V4 minus V3: `{best['mean_delta_V4_minus_V3']:+.4f}`",
 255:         f"- V6 minus V3: `{best['mean_delta_V6_minus_V3']:+.4f}`",
 256:         f"- V4 wins over V3: `{int(best['wins_V4_over_V3'])}/4`",
 257:         f"- worst category: `{best['worst_category']}`",
 258:         f"- worst category delta V4-V3: `{best['worst_category_delta_V4_minus_V3']:+.4f}`",
 259:         "",
 260:         "## Top 10 sources",
 261:         "",
 262:         "| Rank | Q source | Direction | V3 | V4 | V6 | V4-V3 | V6-V3 | Wins V4/V3 | Worst category |",
 263:         "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
 264:     ]
 265: 
 266:     for i, (_, r) in enumerate(ranked.head(10).iterrows(), start=1):
 267:         lines.append(
 268:             f"| {i} | {r['q_source']} | {r['q_direction']} | "
 269:             f"{r['mean_auroc_V3_naive']:.4f} | {r['mean_auroc_V4_quality']:.4f} | "
 270:             f"{r['mean_auroc_V6_adaptive']:.4f} | {r['mean_delta_V4_minus_V3']:+.4f} | "
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b3_ad2_q_source_sweep.py`，第 275–288 行**

```python
 275:     lines += [
 276:         "",
 277:         "## Decision rule",
 278:         "",
 279:         "- If a non-GT Q source gives V4 > V3 on mean AUROC and wins at least 3/4 categories, AD2 QCR can be promoted to a stronger supporting ablation.",
 280:         "- If no Q source passes that threshold, AD2 QCR should be reported as a boundary/diagnostic result rather than a main claim.",
 281:         "- Do not select a Q source using ground-truth overlap, ground-truth mask quality, or label-derived information.",
 282:         "",
 283:         "## Outputs",
 284:         "",
 285:         f"- `{OUT_PER_CATEGORY.relative_to(ROOT)}`",
 286:         f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
 287:         f"- `{OUT_RANKED.relative_to(ROOT)}`",
 288:         "",
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b5_ad2_loco_qcr_policy_optimization.py`，第 169–181 行**

```python
 169:     work["agreement"] = 1.0 - (work["D"] - work["M"]).abs()
 170:     work["mutual_anomaly_evidence"] = np.minimum(work["D"], work["M"])
 171:     work["adaptive_gate"] = work["Q"] * work["K"] * work["agreement"] * work["mutual_anomaly_evidence"]
 172: 
 173:     # Generalized QCR policy.
 174:     work["S_quality"] = work["V3_naive"] - eta * work["M"] * (1.0 - work["Q"])
 175:     work["S_adaptive"] = work["S_quality"] + gamma * work["adaptive_gate"]
 176: 
 177:     rows = []
 178: 
 179:     for cat, sub in work.groupby("category"):
 180:         au_v3 = safe_auroc(sub["gt_binary"], sub["V3_naive"])
 181:         au_q = safe_auroc(sub["gt_binary"], sub["S_quality"])
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b5_ad2_loco_qcr_policy_optimization.py`，第 340–381 行**

```python
 340:     else:
 341:         final_status = "do_not_promote_ad2_qcr_main_claim"
 342: 
 343:     lines = [
 344:         "# Stage 18-B5 AD2 LOCO QCR Policy Optimization",
 345:         "",
 346:         "## Purpose",
 347:         "",
 348:         "Optimize QCR policy without using the held-out AD2 category labels for selection.",
 349:         "",
 350:         "Each fold selects Q source, Q direction, eta, and gamma on three AD2 categories, then evaluates on the held-out category.",
 351:         "",
 352:         "## Safe Q source candidates",
 353:         "",
 354:         "```text",
 355:         *q_cols,
 356:         "```",
 357:         "",
 358:         "## Summary",
 359:         "",
 360:         f"- final_status: `{final_status}`",
 361:         f"- mean test V3 naive: `{fmt(s['mean_test_V3'])}`",
 362:         f"- mean test quality QCR: `{fmt(s['mean_test_quality_qcr'])}`",
 363:         f"- mean test adaptive QCR: `{fmt(s['mean_test_adaptive_qcr'])}`",
 364:         f"- quality QCR minus V3: `{signed(s['mean_delta_quality_minus_V3'])}`",
 365:         f"- adaptive QCR minus V3: `{signed(s['mean_delta_adaptive_minus_V3'])}`",
 366:         f"- quality QCR wins over V3: `{int(s['wins_quality_over_V3'])}/4`",
 367:         f"- adaptive QCR wins over V3: `{int(s['wins_adaptive_over_V3'])}/4`",
 368:         f"- worst adaptive category: `{s['worst_adaptive_category']}`",
 369:         f"- worst adaptive delta: `{signed(s['worst_adaptive_delta'])}`",
 370:         "",
 371:         "## Selected folds",
 372:         "",
 373:         "| Held-out | Selected Q | Direction | eta | gamma | Test V3 | Test Quality | Test Adaptive | Adaptive-V3 |",
 374:         "|---|---|---|---:|---:|---:|---:|---:|---:|",
 375:     ]
 376: 
 377:     for _, r in selected.iterrows():
 378:         lines.append(
 379:             f"| {r['heldout_category']} | {r['selected_q_source']} | {r['selected_q_direction']} | "
 380:             f"{r['selected_eta']:.2f} | {r['selected_gamma']:.2f} | "
 381:             f"{fmt(r['test_V3'])} | {fmt(r['test_quality_qcr'])} | "
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b5_ad2_loco_qcr_policy_optimization.py`，第 385–399 行**

```python
 385:     lines += [
 386:         "",
 387:         "## Decision rule",
 388:         "",
 389:         "- If adaptive QCR has positive mean delta and wins at least 3/4 held-out categories, QCR can be promoted as cross-category calibrated AD2 support.",
 390:         "- If only mean delta is positive but wins fewer than 3/4, report AD2 as weak/boundary support.",
 391:         "- If mean delta is negative, keep AD2 QCR as source-sensitivity diagnostic and retain VisA as the main ablation.",
 392:         "",
 393:         "## Outputs",
 394:         "",
 395:         f"- `{OUT_ALL_CONFIGS.relative_to(ROOT)}`",
 396:         f"- `{OUT_SELECTED.relative_to(ROOT)}`",
 397:         f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
 398:         "",
 399:     ]
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b6_ad2_loco_robust_qcr_selector_sweep.py`，第 277–316 行**

```python
 277:         ascending=[False, False, False],
 278:     ).iloc[0]
 279: 
 280:     lines = [
 281:         "# Stage 18-B6 AD2 LOCO Robust QCR Selector Sweep",
 282:         "",
 283:         "## Purpose",
 284:         "",
 285:         "Test whether AD2 QCR can be rescued by a more robust train-category selector rather than the B5 selector that maximizes train adaptive AUROC.",
 286:         "",
 287:         "No held-out category labels are used for selecting Q source, direction, eta, or gamma.",
 288:         "",
 289:         "## Valid candidate-quality sources",
 290:         "",
 291:         "```text",
 292:         *valid_sources,
 293:         "```",
 294:         "",
 295:         "## Best selector",
 296:         "",
 297:         f"- selector: `{best['selector']}`",
 298:         f"- claim_status: `{best['claim_status']}`",
 299:         f"- mean test V3: `{fmt(best['mean_test_V3'])}`",
 300:         f"- mean test adaptive QCR: `{fmt(best['mean_test_adaptive_qcr'])}`",
 301:         f"- adaptive QCR minus V3: `{signed(best['mean_delta_adaptive_minus_V3'])}`",
 302:         f"- adaptive wins over V3: `{int(best['wins_adaptive_over_V3'])}/4`",
 303:         f"- worst adaptive category: `{best['worst_adaptive_category']}`",
 304:         f"- worst adaptive delta: `{signed(best['worst_adaptive_delta'])}`",
 305:         "",
 306:         "## Selector summary",
 307:         "",
 308:         "| Selector | Status | V3 | Adaptive | Delta | Wins | Worst category | Worst delta |",
 309:         "|---|---|---:|---:|---:|---:|---|---:|",
 310:     ]
 311: 
 312:     for _, r in summary.iterrows():
 313:         lines.append(
 314:             f"| {r['selector']} | {r['claim_status']} | "
 315:             f"{fmt(r['mean_test_V3'])} | {fmt(r['mean_test_adaptive_qcr'])} | "
 316:             f"{signed(r['mean_delta_adaptive_minus_V3'])} | "
```

**`experiments/stage18_ad2_qcr_ablation/run_stage18_b6_ad2_loco_robust_qcr_selector_sweep.py`，第 321–334 行**

```python
 321:     lines += [
 322:         "",
 323:         "## Decision rule",
 324:         "",
 325:         "- If at least one selector has positive mean adaptive delta and wins at least 3/4 held-out categories, AD2 QCR can be used as supporting cross-category evidence.",
 326:         "- If all selectors have negative mean delta, stop optimizing AD2 QCR and report AD2 as boundary/sensitivity evidence.",
 327:         "",
 328:         "## Outputs",
 329:         "",
 330:         f"- `{OUT_FOLDS.relative_to(ROOT)}`",
 331:         f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
 332:         "",
 333:     ]
 334: 
```

## 4. Stage 16 输入 CSV 路径

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 12 行附近

- 字符串形式 CSV：`stage16_a0_qcru_source_inventory.csv`, `stage16_a0_qcru_ablation_plan.csv`

```python
  10: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  11: 
  12: OUT_INVENTORY = OUT_DIR / "stage16_a0_qcru_source_inventory.csv"
  13: OUT_PLAN = OUT_DIR / "stage16_a0_qcru_ablation_plan.csv"
  14: OUT_DOC = DOC_DIR / "stage16_a0_qcru_inventory_and_ablation_plan.md"
  15: 
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 13 行附近

- 字符串形式 CSV：`stage16_a0_qcru_source_inventory.csv`, `stage16_a0_qcru_ablation_plan.csv`

```python
  11: 
  12: OUT_INVENTORY = OUT_DIR / "stage16_a0_qcru_source_inventory.csv"
  13: OUT_PLAN = OUT_DIR / "stage16_a0_qcru_ablation_plan.csv"
  14: OUT_DOC = DOC_DIR / "stage16_a0_qcru_inventory_and_ablation_plan.md"
  15: 
  16: SOURCE_PATHS = [
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 17 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a0_input_structure.csv`, `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv`, `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv`

```python
  15: 
  16: SOURCE_PATHS = [
  17:     "results/stage9_qcr_u/stage9_a0_input_structure.csv",
  18:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv",
  19:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv",
  20:     "results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 18 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a0_input_structure.csv`, `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv`, `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv`

```python
  16: SOURCE_PATHS = [
  17:     "results/stage9_qcr_u/stage9_a0_input_structure.csv",
  18:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv",
  19:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv",
  20:     "results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv",
  21:     "results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 19 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a0_input_structure.csv`, `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv`, `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv`

```python
  17:     "results/stage9_qcr_u/stage9_a0_input_structure.csv",
  18:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv",
  19:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv",
  20:     "results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv",
  21:     "results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv",
  22:     "results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 20 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv`, `results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv`

```python
  18:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv",
  19:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv",
  20:     "results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv",
  21:     "results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv",
  22:     "results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv",
  23:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 21 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv`

```python
  19:     "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv",
  20:     "results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv",
  21:     "results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv",
  22:     "results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv",
  23:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv",
  24:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 22 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv`

```python
  20:     "results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv",
  21:     "results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv",
  22:     "results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv",
  23:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv",
  24:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv",
  25:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 23 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv`, `results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv`

```python
  21:     "results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv",
  22:     "results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv",
  23:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv",
  24:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv",
  25:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv",
  26:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 24 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv`

```python
  22:     "results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv",
  23:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv",
  24:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv",
  25:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv",
  26:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv",
  27:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 25 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv`

```python
  23:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv",
  24:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv",
  25:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv",
  26:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv",
  27:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv",
  28:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 26 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv`, `results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv`, `results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv`

```python
  24:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv",
  25:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv",
  26:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv",
  27:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv",
  28:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv",
  29:     "results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 27 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv`, `results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv`, `results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv`

```python
  25:     "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv",
  26:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv",
  27:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv",
  28:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv",
  29:     "results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv",
  30:     "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 28 行附近

- 字符串形式 CSV：`results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv`, `results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv`, `results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv`, `results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv`

```python
  26:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv",
  27:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv",
  28:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv",
  29:     "results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv",
  30:     "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv",
  31:     "results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv",
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 29 行附近

- 字符串形式 CSV：`results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv`, `results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv`, `results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv`, `results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv`, `results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv`

```python
  27:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv",
  28:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv",
  29:     "results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv",
  30:     "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv",
  31:     "results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv",
  32: ]
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 30 行附近

- 字符串形式 CSV：`results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv`, `results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv`, `results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv`, `results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv`

```python
  28:     "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv",
  29:     "results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv",
  30:     "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv",
  31:     "results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv",
  32: ]
  33: 
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 31 行附近

- 字符串形式 CSV：`results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv`, `results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv`, `results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv`

```python
  29:     "results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv",
  30:     "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv",
  31:     "results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv",
  32: ]
  33: 
  34: 
```

### `experiments/stage16_qcru_ablation/build_stage16_a0_qcru_inventory_and_ablation_plan.py`，第 50 行附近

```python
  48: 
  49:     try:
  50:         df = pd.read_csv(path)
  51:         item["rows"] = len(df)
  52:         item["cols"] = len(df.columns)
  53:         item["columns"] = ";".join(map(str, df.columns))
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 9 行附近

- 字符串形式 CSV：`results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_config.csv`, `results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_category.csv`

```python
   7: ROOT = Path(".").resolve()
   8: 
   9: IN_PER_CONFIG = ROOT / "results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_config.csv"
  10: IN_PER_CATEGORY = ROOT / "results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_category.csv"
  11: 
  12: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 10 行附近

- 字符串形式 CSV：`results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_config.csv`, `results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_category.csv`

```python
   8: 
   9: IN_PER_CONFIG = ROOT / "results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_config.csv"
  10: IN_PER_CATEGORY = ROOT / "results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_category.csv"
  11: 
  12: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
  13: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 15 行附近

- 字符串形式 CSV：`stage16_a2_qcru_variant_delta_by_protocol.csv`, `stage16_a2_qcru_robustness_summary.csv`, `stage16_a2_qcru_failure_cases.csv`

```python
  13: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  14: 
  15: OUT_DELTA = OUT_DIR / "stage16_a2_qcru_variant_delta_by_protocol.csv"
  16: OUT_SUMMARY = OUT_DIR / "stage16_a2_qcru_robustness_summary.csv"
  17: OUT_FAILURES = OUT_DIR / "stage16_a2_qcru_failure_cases.csv"
  18: OUT_DOC = DOC_DIR / "stage16_a2_qcru_robustness_check_report.md"
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 16 行附近

- 字符串形式 CSV：`stage16_a2_qcru_variant_delta_by_protocol.csv`, `stage16_a2_qcru_robustness_summary.csv`, `stage16_a2_qcru_failure_cases.csv`

```python
  14: 
  15: OUT_DELTA = OUT_DIR / "stage16_a2_qcru_variant_delta_by_protocol.csv"
  16: OUT_SUMMARY = OUT_DIR / "stage16_a2_qcru_robustness_summary.csv"
  17: OUT_FAILURES = OUT_DIR / "stage16_a2_qcru_failure_cases.csv"
  18: OUT_DOC = DOC_DIR / "stage16_a2_qcru_robustness_check_report.md"
  19: 
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 17 行附近

- 字符串形式 CSV：`stage16_a2_qcru_variant_delta_by_protocol.csv`, `stage16_a2_qcru_robustness_summary.csv`, `stage16_a2_qcru_failure_cases.csv`

```python
  15: OUT_DELTA = OUT_DIR / "stage16_a2_qcru_variant_delta_by_protocol.csv"
  16: OUT_SUMMARY = OUT_DIR / "stage16_a2_qcru_robustness_summary.csv"
  17: OUT_FAILURES = OUT_DIR / "stage16_a2_qcru_failure_cases.csv"
  18: OUT_DOC = DOC_DIR / "stage16_a2_qcru_robustness_check_report.md"
  19: 
  20: 
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 30 行附近

```python
  28: 
  29: 
  30: def read_csv_robust(path: Path) -> pd.DataFrame:
  31:     df = pd.read_csv(path)
  32:     if len(df.columns) <= 1:
  33:         raise RuntimeError(f"{path} read as <=1 column. Fix CSV line breaks before running Stage 16-A2.")
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 31 行附近

```python
  29: 
  30: def read_csv_robust(path: Path) -> pd.DataFrame:
  31:     df = pd.read_csv(path)
  32:     if len(df.columns) <= 1:
  33:         raise RuntimeError(f"{path} read as <=1 column. Fix CSV line breaks before running Stage 16-A2.")
  34:     return df
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 257 行附近

```python
 255:     DOC_DIR.mkdir(parents=True, exist_ok=True)
 256: 
 257:     per_config = read_csv_robust(IN_PER_CONFIG)
 258:     _ = read_csv_robust(IN_PER_CATEGORY)
 259: 
 260:     delta = compute_protocol_deltas(per_config)
```

### `experiments/stage16_qcru_ablation/build_stage16_a2_qcru_robustness_check.py`，第 258 行附近

```python
 256: 
 257:     per_config = read_csv_robust(IN_PER_CONFIG)
 258:     _ = read_csv_robust(IN_PER_CATEGORY)
 259: 
 260:     delta = compute_protocol_deltas(per_config)
 261:     summary = make_summary(delta)
```

### `experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 10 行附近

- 字符串形式 CSV：`results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv`

```python
   8: ROOT = Path(".").resolve()
   9: 
  10: IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"
  11: 
  12: OUT_DIR = ROOT / "results/stage16_qcru_ablation"
  13: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
```

### `experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 15 行附近

- 字符串形式 CSV：`stage16_b_adaptive_qcru_all_variants_per_config.csv`, `stage16_b_adaptive_qcru_all_variants_per_category.csv`, `stage16_b_adaptive_qcru_primary_protocol_table.csv`, `stage16_b_adaptive_qcru_final_method_decision.csv`

```python
  13: DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
  14: 
  15: OUT_PER_CONFIG = OUT_DIR / "stage16_b_adaptive_qcru_all_variants_per_config.csv"
  16: OUT_PER_CATEGORY = OUT_DIR / "stage16_b_adaptive_qcru_all_variants_per_category.csv"
  17: OUT_PRIMARY = OUT_DIR / "stage16_b_adaptive_qcru_primary_protocol_table.csv"
  18: OUT_DECISION = OUT_DIR / "stage16_b_adaptive_qcru_final_method_decision.csv"
```

### `experiments/stage16_qcru_ablation/build_stage16_b_adaptive_qcru_paper_facing_comparison.py`，第 16 行附近

- 字符串形式 CSV：`stage16_b_adaptive_qcru_all_variants_per_config.csv`, `stage16_b_adaptive_qcru_all_variants_per_category.csv`, `stage16_b_adaptive_qcru_primary_protocol_table.csv`, `stage16_b_adaptive_qcru_final_method_decision.csv`

```python
  14: 
  15: OUT_PER_CONFIG = OUT_DIR / "stage16_b_adaptive_qcru_all_variants_per_config.csv"
  16: OUT_PER_CATEGORY = OUT_DIR / "stage16_b_adaptive_qcru_all_variants_per_category.csv"
  17: OUT_PRIMARY = OUT_DIR / "stage16_b_adaptive_qcru_primary_protocol_table.csv"
  18: OUT_DECISION = OUT_DIR / "stage16_b_adaptive_qcru_final_method_decision.csv"
  19: OUT_REPORT = DOC_DIR / "stage16_b_adaptive_qcru_paper_facing_comparison_report.md"
```

## 5. VisA 样本级缓存候选

| 排名 | 文件 | 行数 | 大小 MiB | 字段组 |
|---:|---|---:|---:|---|
| 1 | `results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv` | 80 | 0.038 | quality=delta_adaptive_minus_quality,delta_fixed_minus_quality,delta_quality_minus_naive,score_quality; label=is_anomaly_final,mutual_anomaly_evidence; category=category |
| 2 | `results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv` | 24 | 0.006 | quality=delta_v4_quality_minus_v3_naive,delta_v5_fixed_minus_v4_quality,delta_v6_adaptive_minus_v4_quality; label=boundary_label; category=category |
| 3 | `results/stage7_generalization/visa_binary_prompt_reasoning/visa_binary_prompt_predictions.csv` | 19458 | 5.107 | label=is_anomaly,label,vlm_anomaly_score; path=canonical_image_path,image_path; category=category |
| 4 | `results/stage7_generalization/visa_manifest/visa_image_manifest.csv` | 10821 | 2.428 | label=is_anomaly,label; path=image_path; category=category |
| 5 | `results/stage7_generalization/visa_anomalib_view/visa_anomalib_view_manifest.csv` | 10821 | 2.206 | label=label; path=source_image_path,view_image_path; category=category |
| 6 | `results/stage7_generalization/visa_multibackbone/fastflow_binary_prompt_reasoning/visa_binary_prompt_predictions.csv` | 6486 | 1.712 | label=is_anomaly,label,vlm_anomaly_score; path=canonical_image_path,image_path; category=category |
| 7 | `results/stage7_generalization/visa_patchcore/VisA/macaroni1/patchcore_image_predictions.csv` | 200 | 0.047 | label=is_anomaly,label,pred_is_anomaly; path=canonical_image_path,image_path; category=category |
| 8 | `results/stage7_generalization/visa_patchcore/VisA/macaroni2/patchcore_image_predictions.csv` | 200 | 0.047 | label=is_anomaly,label,pred_is_anomaly; path=canonical_image_path,image_path; category=category |
| 9 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/macaroni2/fastflow_image_predictions.csv` | 200 | 0.047 | label=is_anomaly,label,pred_is_anomaly; path=canonical_image_path,image_path; category=category |
| 10 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/macaroni1/fastflow_image_predictions.csv` | 200 | 0.047 | label=is_anomaly,label,pred_is_anomaly; path=canonical_image_path,image_path; category=category |
| 11 | `results/stage7_generalization/visa_multibackbone/fastflow_12cls/VisA/candle/fastflow_image_predictions.csv` | 200 | 0.044 | label=is_anomaly,label,pred_is_anomaly; path=canonical_image_path,image_path; category=category |
| 12 | `results/stage7_generalization/visa_multibackbone/fastflow_candle_full/VisA/candle/fastflow_image_predictions.csv` | 200 | 0.044 | label=is_anomaly,label,pred_is_anomaly; path=canonical_image_path,image_path; category=category |

### VisA 候选完整字段

#### 1. `results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv`

```text
case_type, selection_metric, selection_order, backbone, dataset, strategy, eval_mode, category, image_key, is_anomaly_final, D, M, Q, K, agreement, mutual_anomaly_evidence, adaptive_gate, score_naive, score_quality, score_fixed_qc, score_adaptive, delta_quality_minus_naive, delta_fixed_minus_quality, delta_adaptive_minus_quality, delta_adaptive_minus_fixed, detector_vlm_disagreement, has_candidate, num_candidates, fallback
```

#### 2. `results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv`

```text
backbone, dataset, strategy, eval_mode, category, V3, V4, V5, V6, delta_v4_quality_minus_v3_naive, delta_v6_adaptive_minus_v4_quality, delta_v5_fixed_minus_v4_quality, delta_v6_adaptive_minus_v5_fixed, boundary_label
```

#### 3. `results/stage7_generalization/visa_binary_prompt_reasoning/visa_binary_prompt_predictions.csv`

```text
dataset, category, strategy, eval_mode, used_mode, image_path, canonical_image_path, label, is_anomaly, vlm_anomaly_score, fallback, num_eval_images
```

#### 4. `results/stage7_generalization/visa_manifest/visa_image_manifest.csv`

```text
dataset, category, split, label, is_anomaly, image_rel_path, image_path, mask_rel_path, mask_path, image_exists, mask_exists, has_pixel_mask, source_split_csv
```

#### 5. `results/stage7_generalization/visa_anomalib_view/visa_anomalib_view_manifest.csv`

```text
dataset, category, split, label, source_image_path, view_image_path, source_mask_path, view_mask_path, image_status, mask_status
```

#### 6. `results/stage7_generalization/visa_multibackbone/fastflow_binary_prompt_reasoning/visa_binary_prompt_predictions.csv`

```text
dataset, category, strategy, eval_mode, used_mode, image_path, canonical_image_path, label, is_anomaly, vlm_anomaly_score, fallback, num_eval_images
```
