import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from experiments.analysis.conservative_residual_anomaly_calibration import train_category


def build_configs():
    return [
        {
            "config_name": "cfg01_baseline",
            "max_delta": 0.10,
            "lr": 5e-4,
            "identity_weight": 2.0,
            "residual_weight": 1.0,
            "normal_fp_weight": 1.0,
            "dice_weight": 1.0,
        },
        {
            "config_name": "cfg02_smaller_delta",
            "max_delta": 0.05,
            "lr": 5e-4,
            "identity_weight": 2.0,
            "residual_weight": 1.0,
            "normal_fp_weight": 1.0,
            "dice_weight": 1.0,
        },
        {
            "config_name": "cfg03_larger_delta",
            "max_delta": 0.15,
            "lr": 5e-4,
            "identity_weight": 2.0,
            "residual_weight": 1.0,
            "normal_fp_weight": 1.0,
            "dice_weight": 1.0,
        },
        {
            "config_name": "cfg04_stronger_identity",
            "max_delta": 0.10,
            "lr": 5e-4,
            "identity_weight": 5.0,
            "residual_weight": 2.0,
            "normal_fp_weight": 1.0,
            "dice_weight": 1.0,
        },
        {
            "config_name": "cfg05_weaker_identity",
            "max_delta": 0.10,
            "lr": 5e-4,
            "identity_weight": 1.0,
            "residual_weight": 0.5,
            "normal_fp_weight": 1.0,
            "dice_weight": 1.0,
        },
        {
            "config_name": "cfg06_stronger_normal_fp",
            "max_delta": 0.10,
            "lr": 5e-4,
            "identity_weight": 2.0,
            "residual_weight": 1.0,
            "normal_fp_weight": 3.0,
            "dice_weight": 1.0,
        },
        {
            "config_name": "cfg07_lower_lr",
            "max_delta": 0.10,
            "lr": 2e-4,
            "identity_weight": 2.0,
            "residual_weight": 1.0,
            "normal_fp_weight": 1.0,
            "dice_weight": 1.0,
        },
        {
            "config_name": "cfg08_weaker_dice",
            "max_delta": 0.10,
            "lr": 5e-4,
            "identity_weight": 2.0,
            "residual_weight": 1.0,
            "normal_fp_weight": 1.0,
            "dice_weight": 0.5,
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--output_root", type=str, default="results/analysis/conservative_residual_calibration_sweep")
    parser.add_argument("--work_root", type=str, default="runs/analysis/conservative_residual_calibration_sweep")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--threshold_normal_penalty", type=float, default=0.25)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_configs", type=int, default=0)
    args = parser.parse_args()

    all_configs = build_configs()
    if args.max_configs > 0:
        all_configs = all_configs[: args.max_configs]

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    summary_rows = []

    for cfg in all_configs:
        config_name = cfg["config_name"]
        print(f"\n========== Running {config_name} ==========")

        config_rows = []

        for category in args.categories:
            run_args = SimpleNamespace(
                data_root=args.data_root,
                categories=[category],
                output_root=str(Path(args.output_root) / config_name),
                work_root=str(Path(args.work_root) / config_name),
                epochs=args.epochs,
                lr=cfg["lr"],
                weight_decay=args.weight_decay,
                max_delta=cfg["max_delta"],
                bce_weight=args.bce_weight,
                dice_weight=cfg["dice_weight"],
                identity_weight=cfg["identity_weight"],
                residual_weight=cfg["residual_weight"],
                normal_fp_weight=cfg["normal_fp_weight"],
                threshold_normal_penalty=args.threshold_normal_penalty,
                log_interval=args.log_interval,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )

            result = train_category(run_args, category)

            row = copy.deepcopy(result)
            row.update(cfg)
            detail_rows.append(row)
            config_rows.append(row)

        cfg_df = pd.DataFrame(config_rows)

        summary_row = {
            "config_name": config_name,
            "num_categories": len(config_rows),
            "mean_raw_abnormal_f1": cfg_df["eval_raw_abnormal_f1"].mean(),
            "mean_calibrated_abnormal_f1": cfg_df["eval_calibrated_abnormal_f1"].mean(),
            "mean_delta_abnormal_f1": cfg_df["eval_delta_abnormal_f1"].mean(),
            "mean_raw_normal_fp_ratio": cfg_df["eval_raw_normal_fp_ratio"].mean(),
            "mean_calibrated_normal_fp_ratio": cfg_df["eval_calibrated_normal_fp_ratio"].mean(),
            "mean_delta_normal_fp_ratio": cfg_df["eval_delta_normal_fp_ratio"].mean(),
        }
        summary_row.update(cfg)
        summary_rows.append(summary_row)

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    detail_csv = out_root / "conservative_residual_calibration_sweep_detail.csv"
    summary_csv = out_root / "conservative_residual_calibration_sweep_summary.csv"

    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n========== Sweep Summary ==========")
    print(
        summary_df[
            [
                "config_name",
                "mean_raw_abnormal_f1",
                "mean_calibrated_abnormal_f1",
                "mean_delta_abnormal_f1",
                "mean_delta_normal_fp_ratio",
                "max_delta",
                "lr",
                "identity_weight",
                "residual_weight",
                "normal_fp_weight",
                "dice_weight",
            ]
        ].to_string(index=False)
    )

    best = summary_df.sort_values("mean_delta_abnormal_f1", ascending=False).iloc[0]
    print("\n========== Best Config by Delta F1 ==========")
    print(best.to_string())

    print(f"\n[DONE] Detail saved to: {detail_csv}")
    print(f"[DONE] Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
