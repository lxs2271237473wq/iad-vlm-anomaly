from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

OUT_CSV = ROOT / "results" / "stage10_dataset_expansion" / "stage10_dataset_availability.csv"
OUT_DOC = ROOT / "docs" / "stage10_dataset_expansion" / "stage10_dataset_selection_plan.md"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CANDIDATES = [
    {
        "dataset": "MVTec AD 2",
        "priority": 1,
        "recommended_role": "main_next_dataset",
        "expected_dirs": [
            "datasets/MVTecAD2",
            "datasets/MVTec_AD_2",
            "datasets/mvtec_ad_2",
            "MVTecAD2",
            "MVTec_AD_2",
        ],
        "reason": "Official newer 2D industrial anomaly benchmark with advanced scenarios, lighting shifts, transparent/overlapping objects, high normal variance, and tiny defects.",
        "risk": "Dataset structure differs from MVTec AD. Need adapter before PatchCore/FastFlow/crop reasoning.",
    },
    {
        "dataset": "Real-IAD",
        "priority": 2,
        "recommended_role": "large_scale_generalization",
        "expected_dirs": [
            "datasets/Real-IAD",
            "datasets/RealIAD",
            "datasets/realiad",
            "Real-IAD",
            "RealIAD",
        ],
        "reason": "Large-scale real-world multi-view IAD dataset; useful when MVTec AD and VisA are saturated.",
        "risk": "Large download and storage cost. Multi-view structure needs careful subset selection.",
    },
    {
        "dataset": "MVTec LOCO AD",
        "priority": 3,
        "recommended_role": "logical_anomaly_reasoning_supplement",
        "expected_dirs": [
            "datasets/MVTec_LOCO_AD",
            "datasets/MVTecLOCO",
            "datasets/mvtec_loco_ad",
            "MVTec_LOCO_AD",
            "MVTecLOCO",
        ],
        "reason": "Contains both structural and logical anomalies; useful for VLM reasoning discussion.",
        "risk": "Logical anomalies may not align with detector-localization crop assumption.",
    },
    {
        "dataset": "MPDD",
        "priority": 4,
        "recommended_role": "small_real_metal_part_sanity_check",
        "expected_dirs": [
            "datasets/MPDD",
            "datasets/mpdd",
            "MPDD",
            "mpdd",
        ],
        "reason": "Small real metal-part defect benchmark with pixel-level masks.",
        "risk": "Too small to become the main paper dataset.",
    },
]


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            count += 1
    return count


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def inspect_candidate(item: dict) -> dict:
    found_paths = []
    total_images = 0
    total_files = 0

    for rel in item["expected_dirs"]:
        path = ROOT / rel
        if path.exists():
            found_paths.append(str(path.relative_to(ROOT)))
            total_images += count_images(path)
            total_files += count_files(path)

    status = "available" if found_paths and total_images > 0 else "not_found"

    return {
        "priority": item["priority"],
        "dataset": item["dataset"],
        "status": status,
        "found_paths": ";".join(found_paths),
        "image_count": total_images,
        "file_count": total_files,
        "recommended_role": item["recommended_role"],
        "reason": item["reason"],
        "risk": item["risk"],
    }


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)

    rows = [inspect_candidate(item) for item in CANDIDATES]
    df = pd.DataFrame(rows).sort_values("priority")
    df.to_csv(OUT_CSV, index=False)

    available = df[df["status"] == "available"].copy()

    lines = []
    lines.append("# Stage 10-A 新数据集扩展计划")
    lines.append("")
    lines.append("## 1. 当前判断")
    lines.append("")
    lines.append("MVTec AD 和 VisA 已经完成主要验证。继续在这两个数据集上挤小模块，论文收益有限。")
    lines.append("Stage 10 建议引入更具挑战性的数据集，用于验证 localization-guided VLM reasoning 是否仍然成立。")
    lines.append("")
    lines.append("## 2. 推荐优先级")
    lines.append("")
    lines.append("| Priority | Dataset | Status | Image Count | Role |")
    lines.append("|---:|---|---|---:|---|")

    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['priority'])} | {r['dataset']} | {r['status']} | "
            f"{int(r['image_count'])} | {r['recommended_role']} |"
        )

    lines.append("")
    lines.append("## 3. 数据集选择原则")
    lines.append("")
    lines.append("### 3.1 首选 MVTec AD 2")
    lines.append("")
    lines.append("原因：它仍然是 2D industrial anomaly detection，和当前 PatchCore / FastFlow / crop reasoning pipeline 最接近；")
    lines.append("同时它比 MVTec AD 和 VisA 更难，更适合说明方法在新挑战场景下仍有价值。")
    lines.append("")
    lines.append("### 3.2 第二选择 Real-IAD")
    lines.append("")
    lines.append("原因：它规模更大，真实产线、多视角，适合做大规模泛化实验。")
    lines.append("风险：下载和整理成本高，不建议在没有先跑通 MVTec AD 2 adapter 前直接全量上 Real-IAD。")
    lines.append("")
    lines.append("### 3.3 MVTec LOCO AD 作为 reasoning supplement")
    lines.append("")
    lines.append("原因：logical anomaly 更适合 VLM 解释能力。")
    lines.append("风险：logical anomaly 不一定表现为局部 anomaly map 高响应，和当前 crop assumption 可能冲突。")
    lines.append("")
    lines.append("### 3.4 MPDD 作为小型 sanity check")
    lines.append("")
    lines.append("原因：数据小、真实金属件缺陷，适合快速验证。")
    lines.append("风险：规模不够，不建议作为主实验数据集。")
    lines.append("")
    lines.append("## 4. 本地可用性检查结果")
    lines.append("")
    lines.append(f"结果表：`{OUT_CSV.relative_to(ROOT)}`")
    lines.append("")

    if available.empty:
        lines.append("当前没有检测到可直接使用的新数据集。下一步需要先下载并整理 MVTec AD 2。")
    else:
        lines.append("检测到以下可用数据集：")
        lines.append("")
        for _, r in available.iterrows():
            lines.append(f"- {r['dataset']}: {r['found_paths']}，images={int(r['image_count'])}")
        lines.append("")
        first = available.sort_values("priority").iloc[0]
        lines.append(f"建议下一步优先适配：**{first['dataset']}**。")

    lines.append("")
    lines.append("## 5. 下一步")
    lines.append("")
    lines.append("Stage 10-B 应实现所选数据集的 manifest builder。")
    lines.append("manifest 至少包含：")
    lines.append("")
    lines.append("```text")
    lines.append("dataset, category, split, image_path, mask_path, is_anomaly, anomaly_type")
    lines.append("```")
    lines.append("")
    lines.append("之后再复用当前 pipeline：")
    lines.append("")
    lines.append("```text")
    lines.append("detector prediction -> candidate crop -> VLM full/crop reasoning -> table/report")
    lines.append("```")

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8")

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_DOC)
    print(df[["priority", "dataset", "status", "image_count", "found_paths"]].to_string(index=False))


if __name__ == "__main__":
    main()
