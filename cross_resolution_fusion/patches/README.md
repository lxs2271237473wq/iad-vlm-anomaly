# SuperAD integration patch

`superad_cross_resolution.patch` contains the three upstream-file changes
required by this experiment. It was produced against the commit recorded in
`SUPERAD_BASE_COMMIT.txt` and adds:

- local DINOv2 code/weight loading and register-token handling;
- runtime selection of categories and global resolution;
- paired global/tiled inference, component-map export, configurable split,
  and optional suppression of SuperAD's bundled post-evaluation.

Apply it in a clean SuperAD checkout:

```bash
cd "$IAD_REPO_ROOT/ad2_model_zoo/repos/SuperAD"
test "$(git rev-parse HEAD)" = "$(cat "$IAD_REPO_ROOT/cross_resolution_fusion/patches/SUPERAD_BASE_COMMIT.txt")"
git apply --check --ignore-whitespace \
  "$IAD_REPO_ROOT/cross_resolution_fusion/patches/superad_cross_resolution.patch"
git apply --ignore-whitespace \
  "$IAD_REPO_ROOT/cross_resolution_fusion/patches/superad_cross_resolution.patch"
```

The patch does not include upstream SuperAD or DINOv2 source, datasets, or
weights. Obtain those from their respective official distributions.
