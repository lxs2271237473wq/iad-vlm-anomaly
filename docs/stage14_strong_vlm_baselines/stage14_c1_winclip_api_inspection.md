# Stage 14-C1 WinCLIP API Inspection

## 1. Purpose

This report inspects the current Anomalib WinClip, Engine, and Folder APIs before running a WinCLIP pilot.
It does not train or evaluate models.

## 2. Import and Signature Check

| Module | Attribute | Import | Signature / Error |
|---|---|---:|---|
| `anomalib` | `` | True | `2.5.0` |
| `anomalib.models` | `WinClip` | True | `(class_name: str | None = None, k_shot: int = 0, scales: tuple = (2, 3), few_shot_source: pathlib.Path | str | None = None, pre_processor: torch.nn.modules.module.Module | bool = True, post_processor: torch.nn.modules.module.Module | bool = True, evaluator: anomalib.metrics.evaluator.Evaluator | bool = True, visualizer: anomalib.visualization.base.Visualizer | bool = True) -> None` |
| `anomalib.engine` | `Engine` | True | `(callbacks: list[lightning.pytorch.callbacks.callback.Callback] | None = None, logger: lightning.pytorch.loggers.logger.Logger | collections.abc.Iterable[lightning.pytorch.loggers.logger.Logger] | bool | None = None, default_root_dir: str | pathlib.Path = 'results', **kwargs) -> None` |
| `anomalib.data` | `Folder` | True | `(name: str, normal_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path], root: str | pathlib.Path | None = None, abnormal_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path] | None = None, normal_test_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path] | None = None, mask_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path] | None = None, normal_split_ratio: float = 0.2, extensions: tuple[str] | None = None, train_batch_size: int = 32, eval_batch_size: int = 32, num_workers: int = 8, train_augmentations: torchvision.transforms.v2._transform.Transform | None = None, val_augmentations: torchvision.transforms.v2._transform.Transform | None = None, test_augmentations: torchvision.transforms.v2._transform.Transform | None = None, augmentations: torchvision.transforms.v2._transform.Transform | None = None, test_split_mode: anomalib.data.utils.split.TestSplitMode | str = <TestSplitMode.FROM_DIR: 'from_dir'>, test_split_ratio: float = 0.2, val_split_mode: anomalib.data.utils.split.ValSplitMode | str = <ValSplitMode.FROM_TEST: 'from_test'>, val_split_ratio: float = 0.5, seed: int | None = None) -> None` |
| `anomalib.data.datamodules.image.folder` | `Folder` | True | `(name: str, normal_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path], root: str | pathlib.Path | None = None, abnormal_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path] | None = None, normal_test_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path] | None = None, mask_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path] | None = None, normal_split_ratio: float = 0.2, extensions: tuple[str] | None = None, train_batch_size: int = 32, eval_batch_size: int = 32, num_workers: int = 8, train_augmentations: torchvision.transforms.v2._transform.Transform | None = None, val_augmentations: torchvision.transforms.v2._transform.Transform | None = None, test_augmentations: torchvision.transforms.v2._transform.Transform | None = None, augmentations: torchvision.transforms.v2._transform.Transform | None = None, test_split_mode: anomalib.data.utils.split.TestSplitMode | str = <TestSplitMode.FROM_DIR: 'from_dir'>, test_split_ratio: float = 0.2, val_split_mode: anomalib.data.utils.split.ValSplitMode | str = <ValSplitMode.FROM_TEST: 'from_test'>, val_split_ratio: float = 0.5, seed: int | None = None) -> None` |

## 3. Dataset Path Check

| Dataset root | Exists | train/good | test/good | test/bad | ground_truth/bad | #train good | #test good | #test bad | #gt bad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder` | True | True | True | True | True | 300 | 20 | 60 | 60 |
| `datasets/MVTec_AD_2_anomalib_all/vial_folder` | True | True | True | True | True | 332 | 35 | 105 | 105 |
| `datasets/MVTec_AD_2_anomalib_all/walnuts_folder` | True | True | True | True | True | 480 | 60 | 90 | 90 |
| `datasets/MVTec_AD_2_anomalib_all/sheet_metal_folder` | True | True | True | True | True | 156 | 24 | 90 | 90 |

## 4. Minimal Instantiation Check

- WinClip instantiation: success, type = `<class 'anomalib.models.image.winclip.lightning_model.WinClip'>`
- Engine instantiation: success, type = `<class 'anomalib.engine.engine.Engine'>`

## 5. Next Decision

If Folder API is available and AD2 folder roots are valid, Stage 14-C2 should create a one-category WinCLIP pilot on fruit_jelly.
If Folder API parameters are incompatible, Stage 14-C2 should first build a small adapter script based on the inspected signature.