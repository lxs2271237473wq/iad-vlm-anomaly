# Stage 14-B WinCLIP Availability Report

## 1. Purpose

This report checks whether the current environment can directly import and instantiate WinCLIP from Anomalib.

## 2. Import Results

| Module | Attribute | Import | Instantiate | Error |
|---|---|---:|---:|---|
| `anomalib.models` | `WinClip` | True | True | `` |
| `anomalib.models` | `WinCLIP` | False | False | `AttributeError("module 'anomalib.models' has no attribute 'WinCLIP'")` |
| `anomalib.models.image.winclip` | `WinClip` | True | True | `` |
| `anomalib.models.image.winclip` | `WinCLIP` | False | False | `AttributeError("module 'anomalib.models.image.winclip' has no attribute 'WinCLIP'")` |
| `anomalib.models.image.winclip.torch_model` | `WinClipModel` | True | not_attempted | `` |

## 3. Decision

WinCLIP-related classes are available in the current environment.

Next step: create a small Stage 14-C pilot script for one AD2 primary category.