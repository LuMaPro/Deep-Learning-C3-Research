# August — Week 2


## Overview

With the problem specified in Week 1, this week was dedicated to **surveying the two viable answers to the annotation bottleneck** identified in the clinical meeting: either *learn the representation from our own unlabeled slides* through self-supervised pre-training, or *inherit it* from a foundation model already pre-trained on the same kind of material. Each of the two papers read this week is the strongest published instance of one of those options, and both operate on liquid-based cervical cytology digitized as WSI — the exact data setting of this project.

---

## Activities

| # | Activity | Output | Hours |
|---|---|---|---|
| 1 | Paper reading — Stegmüller et al. (2024), self-supervised learning for cervical cytology in a low-data regime | [Paper-Stegmuller.md](Paper-Stegmuller.md) | 4 h |
| 2 | Paper reading — Jiang et al. (2026), UniCAS foundation model for cervical cytology screening | [Paper-UniCAS.md](Paper-UniCAS.md) | 3 h |

---

### 1. Paper reading — Stegmüller et al. (2024) — 4 h

*Self-supervised learning-based cervical cytology for the triage of HPV-positive women in resource-limited settings and low-data regime* (Computers in Biology and Medicine, 169, 107809). SurePath™ LBC slides from HPV-positive women in Cameroon, scanned with a portable low-cost scanner, with a label budget of 1,228 annotated cells against ~1.5M unlabeled tiles.

The paper is the methodological reference for the **in-domain SSL** route. Three points are directly transferable to our setting:

- **DINO pre-training on our own unlabeled tiles is worth more than ImageNet supervision** when the target domain is the same material: +14.8 to +21.6 weighted F₁ at tile level, with the backbone frozen and only a k-NN on top. On external single-cell datasets (Herlev, SIPaKMeD), however, the gain nearly vanishes — the benefit is a function of domain alignment, not of SSL as such.
- **C³P (Cervical Cell Copy-Pasting)**, the proposed augmentation, transfers supervision from public single-cell datasets into the tile domain by pasting annotated cells onto unlabeled tiles. Without it, a classifier fitted on isolated cells scores **0.0 F₁ on the positive class** when tested on real tiles — a concrete measurement of the cells→tiles domain gap that constrains how public datasets can be used before our own material arrives.
- **The tile→slide gap is the unresolved part**: ~96 F₁ at tile level against a best slide-level AUC of 77.5, with off-the-shelf MIL barely above chance until top-k selection and a tile-level loss are added.

Two design decisions were also noted for later reuse: the **frozen-backbone + k-NN probing protocol** as an evaluation that measures the representation rather than the head, and the removal of DINO's local crops on the grounds that cytology tiles are not object-centric.

### 2. Paper reading — Jiang et al. (2026), UniCAS — 3 h

*UniCAS: A foundation model for cervical cytology screening* (Cell Reports Medicine, 7(1)). A ViT encoder pre-trained with DINOv2 on **80.3M patches from 48,532 WSIs**, and the counterpart to the previous paper: instead of running SSL ourselves, use a representation someone else already paid for.

What makes it relevant is that **the pre-training domain is our domain** — ThinPrep, Papanicolaou, 20×, multi-center and multi-scanner — rather than a histopathology model borrowed for cytology. It is used almost everywhere as a **frozen** encoder with a lightweight head, which matches the compute available to us, and its region-level abnormality grading task (NILM / ASC-US / LSIL / ASC-H / HSIL) is the closest published match to the objective of this project.

It also sets realistic expectations: binary abnormality classification reaches 91.47 F₁, but **fine-grained grading drops to 67.53% accuracy on ASC-US** — the Bethesda categories that encode cytopathologist uncertainty remain the hard part even for a model of this scale.

`open`: the paper provides a code repository but makes no explicit statement that the pre-trained weights are downloadable, and mentions no license. This has to be verified in the repository before any experiment is planned around UniCAS.
