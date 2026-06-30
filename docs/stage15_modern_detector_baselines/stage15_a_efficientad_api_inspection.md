# Stage 15-A EfficientAD API Inspection

## 1. Purpose

This stage checks whether EfficientAD is available in the current Anomalib 2.5.0 environment.

It does not train or evaluate any model.

## 2. Import and Signature Results

| Module | Attribute | Import | Signature / Error |
|---|---|---:|---|
| `anomalib.models` | `EfficientAd` | True | `(imagenet_dir: pathlib.Path | str = './datasets/imagenette', teacher_out_channels: int = 384, model_size: anomalib.models.image.efficient_ad.torch_model.EfficientAdModelSize | str = <EfficientAdModelSize.S: 'small'>, lr: float = 0.0001, weight_decay: float = 1e-05, padding: bool = False, pad_maps: bool = True, pre_processor: anomalib.pre_processing.pre_processor.PreProcessor | bool = True, post_processor: anomalib.post_processing.post_processor.PostProcessor | bool = True, evaluator: anomalib.metrics.evaluator.Evaluator | bool = True, visualizer: anomalib.visualization.base.Visualizer | bool = True) -> None` |
| `anomalib.models` | `EfficientAD` | False | `AttributeError("module 'anomalib.models' has no attribute 'EfficientAD'")` |
| `anomalib.models.image.efficient_ad` | `EfficientAd` | True | `(imagenet_dir: pathlib.Path | str = './datasets/imagenette', teacher_out_channels: int = 384, model_size: anomalib.models.image.efficient_ad.torch_model.EfficientAdModelSize | str = <EfficientAdModelSize.S: 'small'>, lr: float = 0.0001, weight_decay: float = 1e-05, padding: bool = False, pad_maps: bool = True, pre_processor: anomalib.pre_processing.pre_processor.PreProcessor | bool = True, post_processor: anomalib.post_processing.post_processor.PostProcessor | bool = True, evaluator: anomalib.metrics.evaluator.Evaluator | bool = True, visualizer: anomalib.visualization.base.Visualizer | bool = True) -> None` |
| `anomalib.models.image.efficient_ad` | `EfficientAD` | False | `AttributeError("module 'anomalib.models.image.efficient_ad' has no attribute 'EfficientAD'")` |
| `anomalib.models.image.efficientad` | `EfficientAd` | False | `ModuleNotFoundError("No module named 'anomalib.models.image.efficientad'")` |
| `anomalib.models.image.efficientad` | `EfficientAD` | False | `ModuleNotFoundError("No module named 'anomalib.models.image.efficientad'")` |
| `anomalib.engine` | `Engine` | True | `(callbacks: list[lightning.pytorch.callbacks.callback.Callback] | None = None, logger: lightning.pytorch.loggers.logger.Logger | collections.abc.Iterable[lightning.pytorch.loggers.logger.Logger] | bool | None = None, default_root_dir: str | pathlib.Path = 'results', **kwargs) -> None` |
| `anomalib.data` | `Folder` | True | `(name: str, normal_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path], root: str | pathlib.Path | None = None, abnormal_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path] | None = None, normal_test_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path] | None = None, mask_dir: str | pathlib.Path | collections.abc.Sequence[str | pathlib.Path] | None = None, normal_split_ratio: float = 0.2, extensions: tuple[str] | None = None, train_batch_size: int = 32, eval_batch_size: int = 32, num_workers: int = 8, train_augmentations: torchvision.transforms.v2._transform.Transform | None = None, val_augmentations: torchvision.transforms.v2._transform.Transform | None = None, test_augmentations: torchvision.transforms.v2._transform.Transform | None = None, augmentations: torchvision.transforms.v2._transform.Transform | None = None, test_split_mode: anomalib.data.utils.split.TestSplitMode | str = <TestSplitMode.FROM_DIR: 'from_dir'>, test_split_ratio: float = 0.2, val_split_mode: anomalib.data.utils.split.ValSplitMode | str = <ValSplitMode.FROM_TEST: 'from_test'>, val_split_ratio: float = 0.5, seed: int | None = None) -> None` |

## 3. Dataset Path Check

| Dataset root | Exists | train/good | test/good | test/bad | ground_truth/bad | #train good | #test good | #test bad | #gt bad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder` | True | True | True | True | True | 300 | 20 | 60 | 60 |
| `datasets/MVTec_AD_2_anomalib_all/sheet_metal_folder` | True | True | True | True | True | 156 | 24 | 90 | 90 |
| `datasets/MVTec_AD_2_anomalib_all/vial_folder` | True | True | True | True | True | 332 | 35 | 105 | 105 |
| `datasets/MVTec_AD_2_anomalib_all/walnuts_folder` | True | True | True | True | True | 480 | 60 | 90 | 90 |

## 4. Minimal Instantiation Check

- EfficientAD instantiation from `anomalib.models.EfficientAd`: success, type = `<class 'anomalib.models.image.efficient_ad.lightning_model.EfficientAd'>`

## 5. Decision

If EfficientAD can be imported and instantiated, Stage 15-B should run a one-category pilot on fruit_jelly or vial.

If EfficientAD requires extra constructor parameters, Stage 15-B should use the inspected signature to create a valid pilot script.