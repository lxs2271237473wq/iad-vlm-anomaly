from __future__ import annotations

from pathlib import Path
from io import StringIO
import re
import pandas as pd


ROOT = Path(".").resolve()

IN_CSV = ROOT / "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv"

OUT_CSV = ROOT / "results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv"
OUT_DOC = ROOT / "docs/stage15_modern_detector_baselines/stage15_f_baseline_decision_and_next_plan.md"

CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts", "MEAN"]


def read_stage15e(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8").strip()

    header = "category,method_group,method,image_auroc,image_ap,pixel_auroc,pixel_f1,protocol,fairness_tag"

    # Robust repair for GitHub/raw one-line CSV artifacts.
    if raw.startswith(header) and "\n" not in raw:
        body = raw[len(header):].strip()
        rows = re.split(
            r"\s+(?=(?:fruit_jelly|sheet_metal|vial|walnuts|MEAN),)",
            body,
        )
        rows = [r.strip() for r in rows if r.strip()]
        raw = header + "\n" + "\n".join(rows) + "\n"

    return pd.read_csv(StringIO(raw))


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def get_mean(df: pd.DataFrame, method: str) -> float | None:
    row = df[(df["category"] == "MEAN") & (df["method"] == method)]
    if row.empty:
        return None
    return safe_float(row.iloc[0]["image_auroc"])


def fmt(x):
    return "" if x is None else f"{x:.4f}"


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)

    df = read_stage15e(IN_CSV)

    for col in ["image_auroc", "image_ap", "pixel_auroc", "pixel_f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    mean_df = df[df["category"] == "MEAN"].copy()
    mean_df = mean_df.sort_values("image_auroc", ascending=False).reset_index(drop=True)
    mean_df["rank_by_image_auroc"] = range(1, len(mean_df) + 1)

    methods = {
        "same_set": "PatchCore + context VLM, same-set",
        "loco": "PatchCore + context VLM, LOCO",
        "patchcore": "PatchCore",
        "efficientad30": "EfficientAD-30 fixed-budget",
        "context_vlm": "context-aware VLM",
        "full_vlm": "full-image VLM",
        "winclip": "WinCLIP fixed protocol",
    }

    scores = {k: get_mean(df, v) for k, v in methods.items()}

    decision_rows = [
        {
            "decision_id": "D1",
            "decision": "Use LOCO fusion as the primary fair result",
            "evidence": f"PatchCore + context VLM, LOCO mean image AUROC is {fmt(scores['loco'])}.",
            "status": "locked",
        },
        {
            "decision_id": "D2",
            "decision": "Keep same-set fusion only as an upper-bound diagnostic",
            "evidence": f"Same-set fusion mean image AUROC is {fmt(scores['same_set'])}, but it is not the fair deployment protocol.",
            "status": "locked",
        },
        {
            "decision_id": "D3",
            "decision": "Keep EfficientAD-30 as a fixed-budget modern detector baseline",
            "evidence": f"EfficientAD-30 mean image AUROC is {fmt(scores['efficientad30'])}.",
            "status": "locked",
        },
        {
            "decision_id": "D4",
            "decision": "Do not immediately run four-category EfficientAD-100",
            "evidence": "EfficientAD-30 does not overturn the LOCO fusion conclusion; full four-category 100-epoch training is not the next bottleneck.",
            "status": "locked",
        },
        {
            "decision_id": "D5",
            "decision": "Add only a fruit_jelly EfficientAD-100 sensitivity check later",
            "evidence": "This is sufficient to test whether 30 epochs severely underestimates EfficientAD before spending time on four-category 100-epoch training.",
            "status": "planned",
        },
        {
            "decision_id": "D6",
            "decision": "Move the main line to QCR-U ablation",
            "evidence": "The current crop/fusion pipeline needs to become a method with quality, consistency, and optional unknown-aware reasoning.",
            "status": "next",
        },
    ]

    decision_df = pd.DataFrame(decision_rows)
    decision_df.to_csv(OUT_CSV, index=False, lineterminator="\n")

    loco_minus_ead = None
    ead_minus_patch = None
    ead_minus_ctx = None
    loco_minus_patch = None

    if scores["loco"] is not None and scores["efficientad30"] is not None:
        loco_minus_ead = scores["loco"] - scores["efficientad30"]
    if scores["efficientad30"] is not None and scores["patchcore"] is not None:
        ead_minus_patch = scores["efficientad30"] - scores["patchcore"]
    if scores["efficientad30"] is not None and scores["context_vlm"] is not None:
        ead_minus_ctx = scores["efficientad30"] - scores["context_vlm"]
    if scores["loco"] is not None and scores["patchcore"] is not None:
        loco_minus_patch = scores["loco"] - scores["patchcore"]

    lines = []
    lines += [
        "# Stage 15-F 强基线结论锁定与后续实验决策",
        "",
        "## 1. 本阶段目的",
        "",
        "Stage 15-E 已经把 WinCLIP、full-image VLM、context-aware VLM、PatchCore、EfficientAD-30、PatchCore+context VLM fusion 放进统一四类别对比表。",
        "",
        "Stage 15-F 的目的不是继续跑实验，而是把当前强基线结论和下一步实验优先级锁定下来。",
        "",
        "## 2. 当前平均 Image AUROC 排名",
        "",
        "| Rank | Method | Mean Image AUROC | Fairness Tag |",
        "|---:|---|---:|---|",
    ]

    for _, r in mean_df.iterrows():
        lines.append(
            f"| {int(r['rank_by_image_auroc'])} | {r['method']} | "
            f"{float(r['image_auroc']):.4f} | {r['fairness_tag']} |"
        )

    lines += [
        "",
        "## 3. 关键差值",
        "",
        f"- LOCO fusion minus EfficientAD-30: `{fmt(loco_minus_ead)}` mean image AUROC.",
        f"- EfficientAD-30 minus PatchCore: `{fmt(ead_minus_patch)}` mean image AUROC.",
        f"- EfficientAD-30 minus context-aware VLM: `{fmt(ead_minus_ctx)}` mean image AUROC.",
        f"- LOCO fusion minus PatchCore: `{fmt(loco_minus_patch)}` mean image AUROC.",
        "",
        "## 4. 当前可以安全使用的结论",
        "",
        "### 4.1 可以作为主结论",
        "",
        "`PatchCore + context VLM, LOCO` 是当前最重要的公平 fusion 结果。",
        "",
        "它比单独 PatchCore 和 EfficientAD-30 fixed-budget 都更高，因此当前 localization-guided VLM fusion 路线仍然站得住。",
        "",
        "### 4.2 只能作为 upper-bound / diagnostic",
        "",
        "`PatchCore + context VLM, same-set` 不能作为最终公平结论。它可以展示同类别调参或同集合融合的上界，但不能过度声称为真实泛化性能。",
        "",
        "### 4.3 EfficientAD 的定位",
        "",
        "`EfficientAD-30 fixed-budget` 是现代非 VLM detector baseline。它比 WinCLIP 和普通 VLM 分支更强，但没有超过 LOCO fusion。",
        "",
        "它不能被写成 EfficientAD official/full-budget baseline。正式论文中必须标注为 fixed-budget 结果。",
        "",
        "## 5. 是否现在跑 EfficientAD-100",
        "",
        "当前决策：**不立即跑四类别 EfficientAD-100**。",
        "",
        "理由：",
        "",
        "1. EfficientAD-30 没有推翻当前 LOCO fusion 结论。",
        "2. EfficientAD 在 Anomalib 下 `train_batch_size=1`，验证阶段还有 quantile/metric 开销，四类别 100 epoch 成本较高。",
        "3. 100 epoch 的价值主要是防守性质，即回答“30 epoch 是否低估 EfficientAD”。",
        "",
        "后续只需要先补一个：",
        "",
        "```text",
        "fruit_jelly EfficientAD-100 sensitivity",
        "```",
        "",
        "如果 fruit_jelly 上 100 epoch 明显高于 30 epoch，再考虑四类别 100 epoch。",
        "",
        "## 6. 下一阶段主线",
        "",
        "下一阶段不应该继续堆 detector baseline，而应该进入：",
        "",
        "```text",
        "Stage 16 / QCR-U ablation",
        "```",
        "",
        "目标是把当前 pipeline 从：",
        "",
        "```text",
        "detector map -> crop -> VLM score -> naive fusion",
        "```",
        "",
        "升级成：",
        "",
        "```text",
        "candidate quality + VLM abnormal margin + detector-VLM consistency + optional unknown-aware reasoning",
        "```",
        "",
        "也就是 QCR-U。",
        "",
        "## 7. 下一步实验优先级",
        "",
        "| Priority | Task | Why |",
        "|---:|---|---|",
        "| 1 | QCR-U fixed-protocol ablation | 这是论文方法核心，不是 baseline 补丁 |",
        "| 2 | fruit_jelly EfficientAD-100 sensitivity | 防守 30 epoch 是否低估 EfficientAD 的质疑 |",
        "| 3 | AnomalyCLIP feasibility check | 补更强 VLM anomaly baseline，但复现成本可能高 |",
        "| 4 | failure case analysis | 支撑论文边界，避免过度声称 |",
        "",
        "## 8. 论文写作边界",
        "",
        "后续论文不能写：",
        "",
        "- 本文解决完整 industrial anomaly understanding。",
        "- 本文能解释 manufacturing cause。",
        "- 本文达到 pixel-level segmentation SOTA。",
        "- EfficientAD 已经被 full-budget 完整击败。",
        "",
        "后续论文应该写：",
        "",
        "- 本文研究如何将 anomaly localization evidence 转化为 reliable visual-language evidence。",
        "- 本文提出 quality-consistency guided candidate reasoning/fusion。",
        "- 本文在多类别和多个 baseline 下证明 localization-guided VLM branch 与传统 detector 具有互补性。",
        "",
        "## 9. 本阶段输出",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_DOC)
    print()
    print(decision_df.to_string(index=False))


if __name__ == "__main__":
    main()
