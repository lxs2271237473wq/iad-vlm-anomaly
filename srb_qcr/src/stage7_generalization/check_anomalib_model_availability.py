from pathlib import Path
import csv
import inspect

import anomalib.models as models


TARGETS = [
    "Patchcore",
    "EfficientAd",
    "Fastflow",
    "ReverseDistillation",
    "Stfpm",
    "Padim",
]


def main():
    out_dir = Path("results/stage7_generalization/multibackbone_availability")
    out_dir.mkdir(parents=True, exist_ok=True)

    available_names = dir(models)
    rows = []

    for target in TARGETS:
        matches = [name for name in available_names if name.lower() == target.lower()]

        if not matches:
            rows.append({
                "target": target,
                "matched_name": "",
                "available": 0,
                "signature": "",
                "error": "not found in anomalib.models",
            })
            continue

        matched_name = matches[0]

        try:
            cls = getattr(models, matched_name)
            signature = str(inspect.signature(cls.__init__))
            rows.append({
                "target": target,
                "matched_name": matched_name,
                "available": 1,
                "signature": signature,
                "error": "",
            })
        except Exception as exc:
            rows.append({
                "target": target,
                "matched_name": matched_name,
                "available": 0,
                "signature": "",
                "error": repr(exc),
            })

    out_path = out_dir / "anomalib_model_availability.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["target", "matched_name", "available", "signature", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Wrote: {out_path}")
    for row in rows:
        status = "OK" if row["available"] else "MISSING"
        print(f"{status:8s} {row['target']:22s} -> {row['matched_name']}")


if __name__ == "__main__":
    main()
