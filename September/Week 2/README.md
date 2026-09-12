# September — Week 2


## Overview

| # | Activity | Output | Hours |
|---|---|---|---|
| 1 | Dealing with class imbalance | [ICIFAR10.ipynb](ICIFAR10.ipynb) | 8 h |

---

### Dataset

[Imbalanced CIFAR-10](https://www.kaggle.com/datasets/akhiltheerthala/imbalanced-cifar-10) (Kaggle), with the original CIFAR-10 test set used for evaluation.

| Split | Images | Balance |
|---|---|---|
| Train | 29,009 | imbalanced |
| Validation | 3,222 | imbalanced |
| Test | 10,000 | balanced |

`Normalize` transform was applied with CIFAR-10's statistics (`mean`,`std`).

### Architecture

**ResNet-18 trained from scratch** (`pretrained=False`), adapted to 32 × 32 inputs:

- `conv1` → 3 × 3, stride 1, padding 1 (instead of 7 × 7, stride 2)
- `maxpool` → `Identity()`
- `fc` → 10 outputs

### Parameters

| | Baseline | SMOTE |
|---|---|---|
| Training loss | `CrossEntropyLoss` | `CrossEntropyLoss` |
| Training data | imbalanced, 29,009 imgs | oversampled to 4,500/class, 45,000 imgs |
| Optimizer | Adam, `lr = 1e-3` | Adam, `lr = 1e-3` |
| Batch size | 64 | 64 |
| Max epochs | 50 | 50 |
| Early stopping | patience 15 | patience 15 |

Seeds (`random`, `numpy`, `torch`, CUDA) are fixed at 42 and re-set before each dataloader and each model, and the `DataLoader` shuffles receive their own seeded generators (the two runs differ only in the training data).

### Training / Validation

Standard loop with the best checkpoint by validation loss saved to disk, and early stopping on the same criterion.

Validation uses a weighted loss `CrossEntropyLoss(weight=1/validation_sizes, reduction='sum')`.

### Test

Evaluation was performed on the **balanced** CIFAR-10 test set.

### Results

| | Baseline | SMOTE |
|---|---|---|
| Best val. loss | **0.902** | 1.246 |
| Test accuracy | 70.95% | **71.51%** |
| Macro F₁ | 0.69 | **0.70** |
| `dog` — precision / recall / F₁ | **0.94** / 0.06 / 0.11 | 0.76 / **0.17** / **0.27** |

- **The baseline abandons the minority class.** `dog` recall is 0.06 with precision 0.94: the model only predicts `dog` when it is nearly certain.
- **SMOTE.** `dog` recall goes 0.06 → 0.17 and its F₁ 0.11 → 0.27, due to `SMOTE` oversampling technique.

### AI use

**Claude** was used in the SMOTE cells (flattening the batches into the array `SMOTE` expects, reshaping the result back to image tensors, and wrapping it in a `Dataset`).

### Extra: WeightedRandomSampler

`WeightedRandomSampler` with per-sample weights `1/class_count`, `replacement=True`.

The metrics were worse than the baseline because of the sampling itself: with replacement, the whole dataset is not processed in an epoch, so the model overfits minority classes instead of learning, and loses information from other classes.
