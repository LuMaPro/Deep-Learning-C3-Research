# August — Week 1


## Overview

The opening week of the project was dedicated to **defining the problem before touching any model**: mapping out the space of decisions (datasets, label granularity, candidate architectures, evaluation criteria) and then deciding what the input data actually is, and what the label actually means. The second one was settled in a technical meeting with the clinical collaborator of the project.

---

## Activities

| # | Activity | Output | Hours |
|---|---|---|---|
| 1 | Problem formulation and scoping | [Problem's scratch.png](Problem's%20scratch.png) | 3 h |
| 2 | Interdisciplinary technical meeting — problem specification | [meeting-clinical-alignment.md](meeting-clinical-alignment.md) | 2 h |
| 3 | Paper reading — Kanavati et al. (2022), deep learning for cervical cancer screening on LBC specimens in WSI | [paper-kanavati-2022-cervical-lbc-wsi.md](paper-kanavati-2022-cervical-lbc-wsi.md) | 3 h |

---

### 1. Problem formulation and scoping — 3 h

A diagram mapping the decision space of the project:

- **Candidate datasets** — SipakMed, Herlev, BMT (ThinPrep), TCGA-CESC, in-house dataset.
- **Label granularity** — isolated cell vs. WSI tile vs. whole-slide decision.
- **Candidate architectures** — convolutional baselines (ResNet) vs. foundation models (UniCAS, DINOv2).

This corresponds to step 1 of the research plan ("become familiar with the application and the type of data involved") and is what determines the datasets and baselines used in the following months.

### 2. Interdisciplinary technical meeting — 2 h

Meeting with Prof. Dr. Michelle Discacciati (Faculty of Pharmaceutical Sciences / University Hospital — USP) and Prof. Dr. Nina S. T. Hirata, to align the computational and clinical sides of the problem. Covered the Bethesda System categories, the morphological criteria used to grade cellular atypia, and the ThinPrep liquid-based preparation protocol; in exchange, the possibilities and limitations of deep learning on this task were presented from our side.


### 3. Paper reading — Kanavati et al. (2022) — 3 h

*A Deep Learning Model for Cervical Cancer Screening on Liquid-Based Cytology Specimens in Whole Slide Images* (Cancers, 14(5), 1159). The published work closest to the exact setting of this project: ThinPrep liquid-based cytology, digitized as WSI, screened for neoplastic vs. NILM at the slide level.
