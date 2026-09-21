# Reproduction

Set the repository root and run from the repository checkout:

```bash
export IAD_REPO_ROOT=/root/private_data/iad-vlm-anomaly
cd "$IAD_REPO_ROOT"
```

Before the first run, apply the integration patch described in
`patches/README.md` to the SuperAD checkout.

## AD2 normal calibration and matched evaluation

```bash
bash cross_resolution_fusion/scripts/run_a67_normal_validation_dual_resolution.sh
python cross_resolution_fusion/scripts/a68_normal_only_resolution_router.py \
  --maps-root ad2_model_zoo/results/a67_normal_validation_dual_resolution_v1 \
  --output-dir orbitad/results/a68_normal_only_resolution_router_v1
bash cross_resolution_fusion/scripts/run_a74_ad2_matched_dual_resolution.sh
```

The A74 runner invokes its evaluator after all eight categories pass component
map count checks.

## MVTec AD external validation

```bash
bash cross_resolution_fusion/scripts/run_a73_mvtec14_dual_resolution.sh
python cross_resolution_fusion/scripts/a73_evaluate_mvtec14_unified.py
```

## Integrity checks

Every generation runner skips a category only when both global and tiled map
counts equal the expected image count. Completion JSON files record dataset,
split, and resolution provenance. Confirm the completion markers before using
the aggregate CSV/JSON reports.
