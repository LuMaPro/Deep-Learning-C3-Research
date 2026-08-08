# Interdisciplinary Technical Meeting — Problem Specification

| | |
|---|---|
| **Date** | August 2026 (Week 1) |
| **Participants** | Prof. Dr. Michelle Discacciati (Faculty of Pharmaceutical Sciences / University Hospital — USP), Prof. Dr. Nina S. T. Hirata (IME-USP), Lucas Martins Próspero |
| **Duration** | 2 hours |
| **Type** | Interdisciplinary technical meeting (problem specification), not an individual advising session |

---

## 1. Purpose

The goal of this meeting was to align the computational side and the clinical side of the project before any modeling decision was made. Two things had to be settled: **what exactly the input data is going to be**, and **what exactly the label is going to mean**. Neither can be decided from the machine learning side alone, and both constrain every experiment that follows.

The exchange went in both directions. From our side, we presented what deep learning can and cannot realistically deliver on this kind of problem. From her side, we received the cytopathological grounding: how a cervical cytology slide is prepared, how a cytopathologist actually reads it, and what the diagnostic categories mean.

---

## 2. What we presented (machine learning side)

- **The general shape of a supervised image classification pipeline**, and the fact that its performance is bounded by the quality and consistency of the annotation, not only by the architecture.

- **The cost structure of annotation.** Cell-level labels are the most informative for us and the most expensive for her team; slide-level labels are cheap but give a much weaker training signal. This trade-off is the reason self-supervised pre-training is on the table as a strategy.

- **Why the granularity of the label matters.** A model trained on isolated cell crops, a model trained on tiles extracted from a Whole Slide Image, and a model that outputs a single decision per slide are three different problems with three different evaluation protocols. We needed to know which one is clinically useful before committing to one.

- **The practical constraints of Whole Slide Imaging**: gigapixel files, multi-resolution pyramids, memory and I/O limits, and the fact that a model never sees a whole slide at once, only patches of it.

- **Class imbalance as a modeling problem.** The abnormal categories are rare, and they are precisely the ones that matter, so global accuracy would be a misleading metric and the loss function has to be chosen accordingly.

---

## 3. What we received (cytopathology side)

### 3.1 The Bethesda System

Cervical cytology reporting follows the Bethesda System, which is a categorical scheme rather than a binary normal/abnormal call. The squamous categories discussed were:

| Category | Meaning |
|---|---|
| `NILM` | Negative for Intraepithelial Lesion or Malignancy |
| `ASC-US` | Atypical Squamous Cells of Undetermined Significance |
| `LSIL` | Low-grade Squamous Intraepithelial Lesion |
| `ASC-H` | Atypical Squamous Cells, cannot exclude HSIL |
| `HSIL` | High-grade Squamous Intraepithelial Lesion |
| `SCC` | Squamous Cell Carcinoma |

Glandular abnormalities (AGC and related) form a separate branch of the same system and are relevant for completeness, though the project focus is on the squamous lesions.

Two points here are directly relevant to the modeling:

1. This is an **ordinal scale of severity**, not a set of unrelated classes. Confusing NILM with HSIL is a much more serious error than confusing ASC-US with LSIL.
2. The **ASC categories exist precisely because the cytopathologist is uncertain**, which means the label itself encodes ambiguity rather than a clean ground truth.

### 3.2 Morphological criteria for atypia

The visual features a cytopathologist relies on to grade a cell were explained, mainly at the nuclear level: nuclear enlargement, increased nuclear-to-cytoplasmic ratio, hyperchromasia, irregular nuclear contour, and coarsening of the chromatin pattern. Severity in the Bethesda scale correlates strongly with how far the nucleus deviates from normal and with how little cytoplasm remains around it. HPV cytopathic effect (koilocytosis) is characteristic of the low-grade end.

This matters to us for two reasons:

- It tells us **what a correct model should be looking at**, which is exactly what will be checked later with Grad-CAM.
- It tells us that **resolution and focus quality are not cosmetic details**: the discriminative signal lives in fine nuclear texture, so aggressive downsampling of patches would destroy the information the decision depends on.

### 3.3 Slide preparation — ThinPrep / liquid-based cytology

The material for this project will be **liquid-based cytology prepared under the ThinPrep protocol**, not conventional smears. The collected cells are suspended in a preservative solution and transferred onto the slide as a thin, roughly uniform layer within a small circular deposition area, then stained with Papanicolaou stain.

Practical consequences for the computational side:

- Cells are distributed in a **monolayer**, with far less overlapping and clumping than in a conventional smear. Individual cell segmentation and cell-level classification are therefore much more tractable.
- **Obscuring material** (blood, mucus, inflammatory background) is largely removed, so the background is cleaner and less likely to dominate what the model learns.
- The cellular material is confined to a **small circle** on the slide, so a large part of the scanned area is empty glass and must be discarded by tissue/cell detection before tiling.
- Because the preparation protocol is standardized, appearance is **more consistent across slides** than in conventional cytology — but this also means a model trained on ThinPrep material should not be assumed to transfer to conventionally prepared slides, and vice versa.

---

## 4. Decisions made

- The project data will be **ThinPrep (liquid-based) cervical cytology slides, digitized as Whole Slide Images**.
- Prof. Discacciati and her team will start the acquisition pipeline on their side: **slide collection, digitization to WSI, and annotation of the lesion regions**.
- Estimated delivery of the collected, digitized and annotated material: **about one month from this meeting** (early September 2026).
- Until that material is available, work on the computational side proceeds on **public datasets**, so that no part of the schedule is blocked by the acquisition.

---

## 5. Why this meeting determines the rest of the project

This was not a status update; it fixed the specification of the data and of the label, which is what the whole experimental design is built on top of.

- **Fixing ThinPrep / liquid-based cytology as the target domain** is what makes BMT a strong candidate as the main public evaluation dataset: it is one of the few public sets prepared under the same protocol and with cross-observer validation. Datasets of isolated cells such as SipakMed and Herlev remain useful for the initial transfer-learning experiments, but they are not the target domain.

- **Fixing WSI as the input format** is what justifies building a whole-slide processing pipeline (tissue detection, tiling, patch filtering, slide-level aggregation) on a public WSI dataset in September, instead of waiting for the real slides to arrive.

- **Fixing the Bethesda categories as the label space** defines the evaluation protocol: per-class sensitivity on the rare high-grade categories, macro-averaged metrics, and a confusion matrix read as an ordinal scale rather than as a flat set of classes.

- **The fact that region-level annotation is expensive and depends on a specialist** is the concrete justification for the self-supervised learning track of the project, rather than a purely methodological preference.

- **The morphological criteria described above** give the qualitative analysis a real criterion: a Grad-CAM map is not just "interpretable output", it can be checked against what a cytopathologist actually looks at.

---

## 6. Open points to settle with the clinical team

- [ ] Annotation granularity of the delivered material: are the lesion regions marked as bounding boxes, polygons, or point annotations on individual cells?
- [ ] Whether each slide carries a single overall Bethesda category in addition to the region-level annotations.
- [ ] Number of slides expected and the approximate distribution across categories, in order to anticipate how severe the class imbalance will be.
- [ ] Whether any slide is read by more than one cytopathologist, which would allow measuring inter-observer agreement and give a realistic upper bound on the accuracy any model can be expected to reach.
- [ ] Scanner model, magnification and file format of the digitized slides, which determines the reading library and the working resolution of the patches.

---

*Estimated time: 2 hours*
