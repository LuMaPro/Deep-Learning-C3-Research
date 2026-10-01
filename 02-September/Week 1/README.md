# September — Week 1


## Overview

| # | Activity | Output | Hours |
|---|---|---|---|
| 1 | First cytology baseline — full fine-tuning of a ResNet-18 on SipakMed | [SipakMed.ipynb](SipakMed.ipynb) | 6 h |
| 2 | Paper reading — Zhang et al. (2025), the HMCHH-TCT-CellDet annotated cervical cytology detection dataset | [Paper-HMCHH.md](Paper-HMCHH.md) | 2 h |

---

### 1. Full fine-tuning on SipakMed — 6 h

**Data:** `CROPPED` single-cell images of SipakMed [Dyskeratotic (813), Koilocytotic (825), Metaplastic (793), Parabasal (787), Superficial-Intermediate (831)]. </br>
A `Dataset` subclass is written per class and the images are resized to 224 × 224 to match the ImageNet input geometry.

**Architecture:** `ResNet-18` with the final `fc` replaced by a 5-output linear layer.

**Model and training:** The whole network is trained (no freezing), and the setup was: Cross-entropy, SGD at `lr = 1e-3`, batch size 64, up to 50 epochs, with early stopping after 5 epochs without improvement.

**Result:** Best validation loss at epoch 25 (`0.194`), with the model reaching **95.10% accuracy** on the test split (see more results on `SipakMed.ipynb`).

### 2. Paper reading — Zhang et al. (2025), HMCHH-TCT-CellDet — 2 h

*HMCHH-TCT-CellDet: A Large Annotated Cervical Cytology Images Dataset for AI Models to Aid Cervical Cancer Screening* (Scientific Data, 12, 23). 15,761 bounding boxes over 8,037 patches of 2048 × 2048 px, cut from 129 digitized ThinPrep slides at Heilongjiang Maternal and Child Health Hospital, released on figshare with Pascal-VOC XML annotations.

Reasons why the dataset does not fit our main project:

- **It supports detection, not classification:** There is a single annotated class `abnormal` (no `negative` label to learn from).
- **The sampling per slide is thin:** ~63 patches are released per WSI, against the several thousand a WSI can yield.
