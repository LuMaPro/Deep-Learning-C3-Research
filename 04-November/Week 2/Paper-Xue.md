# Paper Notes — Deep learning enabled liquid-based cytology model for cervical precancer and cancer detection

> Xue, P.; Dang, L.; Kong, L.-H.; ... ; Qiao, Y.-L.; Zhu, L.; Zhao, F.-H.
> *Nature Communications* **2025**, 16. Published April 13, 2025. DOI: [10.1038/s41467-025-58883-3](https://doi.org/10.1038/s41467-025-58883-3)
> Chinese Academy of Medical Sciences & Peking Union Medical College · Peking Union Medical College Hospital
> Code: <https://github.com/LuZWCHA/LBC_WSI_Classification>



## 1. Data

| Item | Value |
|---|---|
| Training set | **17,397 slides** (one slide per woman), 2 pathology archives (Chongqing, Guangzhou) |
| Slide labels (training) | 10,601 negative · 2,314 ASC-US · 1,833 LSIL · 1,657 ASC-H · 693 HSIL + SCC · 299 AGC/AIS/ADC |
| Cell annotations | **1,011,322 expert-annotated cell tiles**: 225,970 ASC-US · 121,404 LSIL · 318,241 ASC-H · 232,199 HSIL + SCC · 113,508 AGC/AIS/ADC |
| Preparations | ThinPrep, SurePath and LBP|
| Scanners | 3 (Bingli IBL-300, Shengqiang SQ-600, Motic EasyScan NFC-60) |
| Test sets | A: 5,803 slides from **9 other hospitals** · B: 550 (from A) · C: 3,001 (community screening) · D: 1,472 (4 hospitals) |

Decisions that matter for the model:

- **Positive = ASC-US or worse.** The model output is binary (negative / positive), even though the slide labels have 6 Bethesda categories
- **Training labels are cytological, test labels are histological** (CIN2+ by biopsy, or by HPV + cytology + colposcopy when there is no biopsy)
- **No overlap** between training and test sets, and the test hospitals are different from the training archives (external validation)
- Mixing preparations and scanners in training was intentional, to generalize across settings

---

## 2. Pipeline

**Model development**

```text
Training set (17,397 women) → Specimen collection → Specimen processing → LBC slides
→ Scan → Whole digital slides → Annotations → Negative and positive cells (squamous and glandular)
→ Train → DL model
```

**Stepwise external validation**

```text
Stage 1 (multi-institutional):  test set A (5,803 women)  → DL model  vs  9 cytopathologists (3 juniors, 6 seniors) from 9 hospitals

Stage 2 (multi-reader multi-case):  test set B (550 women)  → 28 cytopathologists (16 juniors, 12 seniors)
                                    1st read: manual  →  2nd read (4 weeks later): with DL assistance

Stage 3.1 (community screening):  slides → scan → cloud → DL model (automatic)  vs  3 seniors (manual)   — test set C (3,001 women)
Stage 3.2 (hospital screening):   slides → scan → local service → DL model → 4 juniors (1st manual, 2nd with DL assistance) — test set D (1,472 women)
```

---

## 3. Architecture

The paper only mentions cell-level detection, then slide-level classification with multiple instance learning, so the details below come from the github repository of the project.

```text
WSI → grid patches (1280 × 1280) → Cell detector (RetinaNet) → abnormal cell boxes + class + score
→ top-k boxes (k = 24) → crops (640 × 384) → Crop classifier (ResNet-50 + FPN + ImageClsHead) → 24 × 512-d
→ Slide classifier (MIL Transformer) → negative / positive
```

#### 3.1 Cell-level detector (RetinaNet)

Detectron2 RetinaNet:

- `backbone` → ResNet-50 + FPN (levels P3–P7, 256 channels), FrozenBN, stem and first stage frozen
- `cls_subnet` → 4 × [`Conv2d(256, 256, 3×3)` + GroupNorm + ReLU] → `cls_score` (9 anchors × classes)
- `bbox_subnet` → 4 × [`Conv2d(256, 256, 3×3)` + GroupNorm + ReLU] → `bbox_pred` (9 anchors × 4)
- Loss → sigmoid **focal loss** (α = 0.25, γ = 2) for classes + **GIoU** loss for the boxes
- Inference → score > 0.05, NMS (IoU 0.3), up to 100 detections per patch

The WSI is cut into **1280 × 1280** patches and each of them goes through the detector, which includes classes ASC-US, LSIL, HSIL (+ SCC), AGC and non-neoplastic findings (trichomonas, fungi, clue cells, etc.).

#### 3.2 Top-k selection

- Only detections of the abnormal classes (**ASC-US, LSIL, HSIL, AGC**) are kept
- They are sorted by detector score and the **top-k = 24** are kept
- For each one, a **640 × 384** patch at is cropped around the box center

The instances of the MIL are these 24 crops, not all the tiles of the slide.

#### 3.3 Slide-level classifier (MIL Transformer)

Each crop goes through the same ResNet-50 + FPN backbone with an image classification head (`ImageClsHead`, 256 channels, GroupNorm). It gives one **512-d** vector per crop , then `MILClassifier`:

- `cls_token` → learnable `[1, 1, 512]` vector, concatenated before the 24 instances → `[1, 25, 512]`
- `attn_blocks` → **1 Transformer block**: LayerNorm → multi-head self-attention (**8 heads**, dropout 0.3) → residual → LayerNorm → MLP (`512 → 2048 → 512`, GELU) → residual
- `norm` → LayerNorm, keep only the **CLS token** → `[1, 512]`
- `fc` → `Linear(512, 1024)` + ReLU
- `classifier` → `Linear(1024, 3)`

| Output | Classes |
|---|---|
| 0 | ASC-US, LSIL |
| 1 | HSIL, ASC-H |
| 2 | AGC, SCC |

At inference the slide score is the **max of the 3 outputs**, and the slide is positive above a threshold.

#### 3.4 Training (slide-level)

- 24 crops per slide, random crop `384 × 640` + augmentation (CLAHE, Gaussian/ISO noise, color jitter, blur)
- SGD, lr `0.0025`, momentum 0.9, weight decay `1e-4`, warmup + steps at 400/600/800 iterations, 4,000 iterations, mixed precision
- Initialized from the detector checkpoint

---

## 4. Results (DL alone)

| Setting | Sensitivity | Specificity | Other |
|---|---|---|---|
| Stage 1, 9 hospitals (CIN2+) | 0.861 | 0.829 | accuracy 0.832, AUC **0.814–0.868** per hospital |
| Stage 3.1, community screening | 0.878 (seniors 0.854) | 0.831 (seniors 0.901) | — |
| Stage 3.2, juniors without → with DL | 0.657 → 0.857 | 0.737 → 0.840 | — |

- Multi-reader study: DL assistance raised the cytopathologists' sensitivity by 9% and cut the reading time from 218 s to 30 s
- Most errors involve **ASC-US**