# Paper Notes — UniCAS: A foundation model for cervical cytology screening

> Jiang, H.; Cai, J.; Shen, Z.; Xu, M.; Fei, M.; Huang, H.; Wang, X.; Bi, R.; Shen, D.; Zhang, L.; Wang, Q.
> *Cell Reports Medicine* **2026**, 7(1). Published January 20, 2026.
> <https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(25)00643-3> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC12866161/>


## 1. Summary

### 1.1 Motivation

Manual examination of cervical cytology slides is labor-intensive and error-prone. The authors' specific complaint about the existing deep learning literature is **fragmentation**: current approaches require a *specialized model for each diagnostic task*, producing disconnected workflows. UniCAS is proposed as **"a unified encoder for comprehensive analysis"** spanning multiple diagnostic scales — from pixels to whole slides — so that one pretrained representation serves screening, grading, detection, segmentation and image enhancement alike.

### 1.2 What UniCAS is

A **ViT-style encoder pretrained with DINOv2** on cervical cytology patches, meant to be used as a frozen (or lightly adapted) feature extractor for downstream tasks. It is a **patch-level** foundation model: patches are encoded independently, and there is **no separate slide-level pretraining stage** — slide-level behaviour comes from aggregators trained on top of the frozen patch features.

### 1.3 Pretraining data and setup

| Item | Value |
|---|---|
| WSIs | **48,532** (one slide per patient) |
| Patches | **80,314,640**, size **256 × 256 px at 20×**, non-overlapping |
| Preparation | **ThinPrep (TCT) liquid-based cytology, Papanicolaou stained** |
| Sources | Multiple third-party pathology diagnostic centers; **11 scanning devices** with varying staining protocols |
| Patients | Age 15–90, majority 30–45 |
| Label distribution (of the pretraining cohort) | 34,103 NILM · 14,429 abnormal — ASC-US 10,299, LSIL 2,895, ASC-H 641, AGC 419, HSIL 175, candidiasis 816, clue cells 1,980 |
| SSL framework | **DINOv2** (student–teacher self-distillation) |
| Views | two 224 × 224 global crops, eight 96 × 96 local crops, plus masked crops |
| Compute | 8 × NVIDIA A100, batch size 768, **1M steps**, lr 3 × 10⁻⁴ with cosine schedule |
| Augmentation | Colour augmentations to simulate staining variability |

**The single most relevant fact for us: the pretraining domain is exactly our domain.** ThinPrep, Papanicolaou, 20×, multi-center, multi-scanner. This is not a histopathology model being borrowed for cytology.

`not reported`: the ViT variant (S/B/L/g), the internal patch size of the transformer, and the embedding dimension of the encoder (the linear probing layer is described as 768-dimensional, but the backbone feature width is not stated).

### 1.4 Availability

> "Code and demo data for UniCAS are available at https://github.com/peter-fei/UniCAS."

The pretraining data is **not** public (patient privacy); region-level annotated data is available on reasonable request from the lead contact. The paper contains **no explicit statement that the pretrained weights are downloadable**, and mentions no license or HuggingFace entry — this has to be checked directly in the repository before planning anything around it.

---

## 2. What the model was shown to do

Five task families, each adapted differently on top of the same encoder. Headline numbers only.

| Scale | Task | Adaptation used | Headline result |
|---|---|---|---|
| **Slide** | Cancer screening (pos/neg) | Frozen encoder + MIL aggregator | **AUC 92.60%** internal · **90.10%** external |
| **Slide** | Candidiasis testing | idem | AUC 92.58% · 89.92% external |
| **Slide** | Clue cell diagnosis | idem | AUC 98.39% · 98.11% external |
| **Slide** | 5-class grading (**BMT dataset**) | Multi-task aggregator | **AUC 99.19% NILM · 98.54% LSIL · 99.79% HSIL** |
| **Region** | Abnormality classification | Linear probe (2-layer) | F1 **91.47%** |
| **Region** | Candidiasis / clue cells | Linear probe | F1 96.30% / 98.44% |
| **Region** | **Abnormality grading (NILM, ASC-US, LSIL, ASC-H, HSIL)** | Linear probe | Accuracy **67.53% (ASC-US) – 81.40% (ASC-H)** |
| **Region** | Abnormal cell detection | ViTDet + Mask R-CNN | AP **29.99** |
| **Region** | Clue cell detection | idem | AP 30.06 |
| **Region** | Nucleus / cytoplasm segmentation | ViTDet + Mask R-CNN | Dice **91.23% / 92.99%** (ISBI 2014), 90.69% (CNSeg), 94.38% (Cx22-Multi) |
| **Pixel** | Out-of-focus deblurring | DA-CLIP, **fine-tuned end-to-end** | SSIM 74.45%, PSNR 26.92 dB |

**Datasets involved** include large internal cohorts (28,110 WSIs for screening; 74,250 regions for abnormality classification) plus public ones: **Herlev** (917), **SIPaKMeD** (4,049), **ComparisonDetector** (7,410 images / 48,587 boxes), **ISBI 2014**, **CNSeg**, **Cx22-Multi**, and **BMT**.

**Comparisons.** UniCAS is compared against other foundation models — **Gigapath, H-optimus, UNI, Hibou-B, CONCH** — and the paper states it "consistently outperforms existing foundation models". It is **not** compared against an ImageNet ResNet baseline.

**Efficiency.** The multi-task aggregator processes all three slide-level tasks in a single forward pass: **9.75 s per WSI**, described as a 70% reduction / **3× faster** than running the tasks separately.

**The proposed aggregator**, briefly: a **Task-specific Attention** where each task token selects the **top-m** most significant patch tokens (m = 256 by default), followed by a **Task Mixture-of-Experts** — a soft MoE with s = 8 slots, each slot a weighted combination of input tokens routed to a dedicated expert and recombined by row-wise softmax. Ablations exist for m (Table S19) and s (Table S20).

**No ablation** is reported on pretraining data scale, nor on DINOv2 versus alternative SSL frameworks.

---

## 3. How we could use UniCAS in our project

Our task — classification of cellular atypias in liquid-based cervical cytology — sits squarely inside what this encoder was pretrained for. Below are the usable entry points, ordered roughly by increasing cost.

### 3.1 As a frozen feature extractor, with no training at all

The cheapest possible first experiment: run the encoder over patches and inspect the embedding space directly — k-NN probing, t-SNE projections, retrieval of visually similar cells. This measures **representation quality on our data** before committing to any training, and gives an immediate answer to "does a model pretrained on Chinese multi-center ThinPrep transfer to our material?".

*(The paper does not use k-NN probing — this is our own proposed protocol, borrowed from the Stegmüller evaluation methodology.)*

### 3.2 Frozen encoder + linear probe — region/patch-level atypia classification

This is the paper's own **region-level abnormality grading** task, and it is the closest match to our stated objective: a two-layer linear network on frozen features, classifying **NILM / ASC-US / LSIL / ASC-H / HSIL**. Trained for 30 epochs, batch 32, lr 1 × 10⁻⁴.

Two things make this the natural starting point for us:

- It is cheap — the encoder never receives gradients, so a single GPU suffices.
- It is the honest way to measure the *representation*, since a linear head has almost no capacity to compensate for a bad backbone.

Worth noting the reported numbers: abnormality classification reaches F1 91.47%, but **fine-grained grading is much weaker — 67.53% accuracy on ASC-US**. The hard part of our problem remains hard even for this model, which is useful to know before setting expectations.

### 3.3 Transfer learning + fine-tuning

The paper mostly keeps the encoder **frozen**, adapting only the head — linear probes for classification, ViTDet/Mask R-CNN heads for detection and segmentation. The one place it fine-tunes **end-to-end** is the pixel-level enhancement task.

That asymmetry is a useful signal for us: with a well-matched pretraining domain, the authors judged that frozen features plus a task head were sufficient for the diagnostic tasks. A sensible plan is therefore to treat full fine-tuning as a **second step**, only if the linear probe underperforms — and to consider intermediate options (unfreezing the last blocks, or partial fine-tuning of normalization parameters, as Kanavati did) before paying for a full fine-tune.

---

[← Back to Week 2](README.md)
