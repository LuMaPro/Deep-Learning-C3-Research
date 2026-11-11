# Paper Notes — Domain-specific foundation model allows weakly supervised whole-slide cervical cytology classification

> Martins, L.V.d.C.; Toledo, M.; Valieris, R.; Defelicibus, A.; Bueno, A.P.; Saieg, M.; Tojal da Silva, I.
> *Surgical and Experimental Pathology* **2026**, 9, 28. Published June 1, 2026. DOI: [10.1186/s42047-026-00241-8](https://doi.org/10.1186/s42047-026-00241-8)
> CIPE & Department of Anatomic Pathology, A.C.Camargo Cancer Center (São Paulo) · Jewish General Hospital, McGill University
> Code: not released · Data: available from the lead contact upon request

## 1. Data

| Item | Value |
|---|---|
| Cohort | **1,235 patients**, routine screening at A.C.Camargo Cancer Center (2019–2023), retrospective |
| Preparation | Liquid-based cytology (the protocol is not specified) |
| Scanner | Leica Aperio AT2, **40×** |
| Labels | Slide-level only, taken from the **pathology reports** (Bethesda) |
| Classes | NEG 530 (43%) · ASC 362 (29%, ASC-US + ASC-H) · POS 343 (28%, LSIL + HSIL) |
| Split | **15% of patients** held out for test (patient-wise), 5-fold cross-validation on the remaining 85% |

Three classification targets were trained on the same cohort:

- **NEG × POS:** binary, the ASC slides are left out.
- **NEG × OTHERS:** binary, OTHERS = ASC + POS (ASC-US, ASC-H, LSIL, HSIL).
- **NEG × ASC × POS:** three classes.

Glandular atypias and carcinoma were excluded because they are too rare in the cohort.

## 2. Pipeline

```text
WSI (40×) → tiles 224 × 224 at 0.5 µm/px → Otsu background filter (~20,000 tiles per slide)
→ frozen feature extractor (8 options) → [N × D] per slide
→ MIL (CLAM or TransMIL), 8,000 tiles sampled per slide → slide class
```

All 3 targets × 2 MIL methods × 8 feature extractors × 5 folds add up to **450 trained models**.

## 3. Architecture

#### 3.1 Feature extractors

| Model | Release | Parameters | Embedding |
|---|---|---|---|
| ResNet50-ImageNet | 2015 | 26 M | 2048 |
| ResNet50-RetCCL | Jan 2023 | 26 M | 2048 |
| UNI | Aug 2023 | 300 M | 1024 |
| Hibou-B | Jun 2024 | 86 M | 768 |
| H-Optimus | Jul 2024 | **1.1 B** | 1536 |
| Virchow2 | Aug 2024 | 632 M | 3584 |
| UNI2 | Jan 2025 | 681 M | 1536 |
| **DINOv3 finetune** (proposed) | — | 86 M | 768 |

All of them are histology (or ImageNet) models, except the proposed one, which is adapted to cytology.

#### 3.2 DINOv3 finetune (proposed)

Self-supervised finetune of a **DINOv3 ViT-B** on the cohort itself, without labels.

#### 3.3 MIL aggregators

**CLAM**

- One attention branch **per class**: each one gives an attention-weighted sum of the tiles
- One linear head per class (the class with the highest score wins)

**TransMIL**

- Transformer over all the tiles

#### 3.4 Training

- 150 epochs, AdamW, lr `5e-4` (fixed), early stopping after 20 epochs without improvement in validation loss
- **8,000 tiles** sampled per slide, batch of 1 slide, gradient accumulation of 10

## 4. Results

**Proposed extractor + CLAM:**

| Target | AUC (median) |
|---|---|
| NEG × POS | ≈ 0.83 |
| NEG × OTHERS | ≈ 0.79 |
| NEG × ASC × POS | ≈ 0.76 |

**Feature extractor comparison:**

| Model | NEG × POS | NEG × OTHERS | NEG × ASC × POS |
|---|---|---|---|
| H-Optimus | ≈ 0.86 | ≈ 0.84 | ≈ 0.82 |
| **Proposed** | ≈ 0.83 | ≈ 0.80 | ≈ 0.76 |
| Virchow2 | ≈ 0.80 | ≈ 0.78 | ≈ 0.76 |
| Hibou-B | ≈ 0.79 | ≈ 0.78 | ≈ 0.75 |
| ImageNet | ≈ 0.78 | ≈ 0.78 | ≈ 0.75 |
| UNI2 | ≈ 0.79 | ≈ 0.75 | ≈ 0.75 |
| UNI | ≈ 0.78 | ≈ 0.76 | ≈ 0.75 |
| RetCCL | ≈ 0.72 | ≈ 0.75 | ≈ 0.73 |