# August — Week 3


## Overview

| # | Activity | Output | Hours |
|---|---|---|---|
| 1 | Paper reading — Welch et al. (2024), the BMT cross-validated ThinPrep cervical cytology dataset | [Paper-BMT.md](Paper-BMT.md) | 3 h |
| 2 | PyTorch warmup — reviewing PyTorch syntax on MNIST | [MNIST_basic.ipynb](MNIST_basic.ipynb) | 4 h |

---

### 1. Paper reading — Welch et al. (2024), BMT — 3 h

The dataset is composed of 600 multicellular fields of view — 200 NILM, 200 LSIL, 200 HSIL — captured from 180 archival ThinPrep® slides and released on Synapse under CC BY 4.0.

Two further points worth carrying forward:

- **The labels are unusually trustworthy, by construction.** Each image required **100% consensus among three board-certified pathologists**, and disagreements were *discarded rather than adjudicated* — under 10% of captured images were dropped. The stated cost is a selection bias toward unambiguous cases.
- **The weak baselines ship with the dataset** — SVC and Random Forest at 55%, VGG19 at 64.16%, **ResNet50 at 74.16%** — so they do not have to be rebuilt; a fine-tuned CNN of our own can be positioned directly against published numbers.

### 2. PyTorch warmup — MNIST — 4 h

A warmup to review PyTorch syntax.

List of some commands/modules:

- `transforms.v2.Compose`
- `datasets.MNIST`
- `random_split`
- `DataLoader`
- `nn.Module`
- `nn.Conv2d`
- `nn.MaxPool2d`
- `nn.Flatten`
- `nn.Sequential`
- `nn.CrossEntropyLoss`
- `optim.Adam`
- `model.train()` / `model.eval()`
- `torch.no_grad()`
