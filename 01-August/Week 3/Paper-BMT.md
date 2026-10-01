# Paper Notes — BMT: A Cross-Validated ThinPrep Pap Cervical Cytology Dataset for Machine Learning Model Training and Validation

> Welch, E.C.; Lu, C.; Sung, C.J.; Zhang, C.; Tripathi, A.; Ou, J.
> *Scientific Data* **2024**, 11, 1444. Published December 28, 2024. DOI: [10.1038/s41597-024-04328-3](https://doi.org/10.1038/s41597-024-04328-3) · PMID: 39732723
> <https://www.nature.com/articles/s41597-024-04328-3> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC11682344/>

## 1. Summary

This is a **data descriptor**, not a modelling paper. The contribution is the dataset itself and the protocol used to label it.

### 1.1 The gap it fills

The authors argue that the publicly available cervical cytology datasets are unusable as proxies for real ThinPrep workflows, for three reasons:

- **Wrong preparation.** Existing datasets use conventional (non-liquid) Pap preparations, SurePath™, or cytospin — all *visually distinct* from the widely adopted ThinPrep® protocol.
- **Wrong granularity.** Many contain **pre-segmented single cells** rather than multicellular fields, which is not what a screener actually looks at.
- **Wrong balance.** Significant class imbalance skews them toward benign cases.

The evidence they give for why this matters is the strongest single sentence in the paper for our purposes: **"accuracies as low as 60.29% were obtained when the SurePath™ trained model was tested on ThinPrep® images"** — even with domain adaptation. Preparation protocol is not a cosmetic detail; it is a domain shift severe enough to destroy a model.

BMT is presented as **"the first publicly available multicellular ThinPrep® dataset"**.

### 1.2 What the dataset is

| Attribute | Value |
|---|---|
| Source slides | **180 deidentified archival ThinPrep® slides**, from **180 individual patients** |
| Institution | Women and Infants Hospital of RI (WIHRI), Providence, RI |
| Images | **600 multicellular images** — **200 each for NILM, LSIL, HSIL** (perfectly balanced) |
| Fields of view | "At least 1 and a maximum of 5 fields of view images were captured for each of the 180 unique patient slides" |
| Image size / format | **1920 × 1080 px, JPEG** |
| Optics | **Olympus BX43, 40× objective, 0.5× C-mount adaptor**, Excelis HD colour camera |
| Preparation | **Hologic ThinPrep® 5000** (automated), PreservCyt solution, broom/brush collection |
| Staining | **Papanicolaou**, automated Hologic protocol |

**Capture criteria** — each field had to contain at least one diagnostic cell; fields mixing LSIL and HSIL cells were excluded; and confounding features (overlapping cells, debris, mucus, blood) were avoided when possible.

**Classes are NILM / LSIL / HSIL only.** ASC-US and SCC were deliberately excluded — ASC-US because of high interobserver variability, SCC because of its rarity in routine screening.

### 1.3 The labelling protocol — why "cross-validated"

This is what distinguishes BMT from other public datasets, and it is worth reading carefully:

| Step | Who | What |
|---|---|---|
| Initial diagnosis | **2 board-certified cytopathologists** (third party) | Diagnosed the slides as part of clinical routine, blinded to everything downstream |
| Capture + preliminary label | **1 board-certified gynecologic pathologist** | Selected the multicellular fields and assigned a preliminary class |
| Independent validation | **2 further board-certified gynecologic pathologists**, blinded to the original diagnosis | Independently classified each image by Bethesda criteria |
| Inclusion rule | all three | **100% class consensus required**; images without unanimous agreement were dropped |

**Fewer than 10% of the initially captured images were excluded** for lack of consensus.

Note the design decision: disagreements are **not adjudicated, they are discarded**. That produces labels of unusually high confidence, at the cost of a dataset that systematically over-represents unambiguous cases (the authors acknowledge this as a selection bias).

`not reported`: any inter-observer agreement statistic (no kappa values), despite "cross-validated" being in the title; and the years of experience of the pathologists beyond "board-certified".

### 1.4 Technical validation (baselines provided by the authors)

Split **60:20:20** (360 train / 120 val / 120 test), with the training set augmented to 720 images via TensorFlow `RandomFlip` and `RandomRotation`:

| Model | Test accuracy |
|---|---|
| Support Vector Classifier | 55% |
| Random Forest | 55% |
| VGG19 | 64.16% |
| **ResNet50** | **74.16%** |

ROC curves, confusion matrices and classification reports are provided for each. A **saliency analysis** on ResNet50 shows the model attending to **nuclear morphology**.

The authors also mention — as forthcoming and **unpublished** — a hierarchical CNN reaching **>90%** accuracy on this dataset.

### 1.5 Availability

- **Data:** hosted on **Synapse** (Sage Bionetworks), DOI `10.7303/syn55259257`, **CC BY 4.0**, declared "permanently publicly accessible".
- **Structure:** images in original JPEG, split into three folders by validated true class, plus a separate sub-folder of **annotated images** — "These images clearly depict important cellular diagnostic classifications that are relevant in classification of each image" — with an accompanying table giving classification context.
- **Code:** <https://github.com/celwelch/BMTcode/> (Python scripts for data analysis and comparison).

---

## 2. How BMT can be useful for our project

### 2.1 As an inference-only validator of UniCAS as a feature extractor

This is the primary use, and it is cheap. The pipeline requires **no training of the encoder at all**:

```text
600 BMT images → UniCAS (frozen) → embeddings → light probe → NILM/LSIL/HSIL metrics
```

The "light probe" can be a k-NN or a linear head — both are minutes of compute. What we get out is a direct answer to *does the UniCAS representation separate atypia grades on ThinPrep material it has never seen?*, measured on labels of unusually high confidence.

Two properties make BMT nearly ideal for this specific job:

- **Perfectly balanced classes** (200/200/200), so accuracy and macro-F1 are interpretable without imbalance corrections and no class weighting is needed to get a first reading.
- **Public and permanently accessible under CC BY 4.0**, so the experiment is fully reproducible and publishable without any data agreement.

### 2.2 The baseline comparison

BMT is where the comparison our advisor suggested can actually be run: **UniCAS as published**, against a conventionally pretrained CNN fine-tuned for the same task, under an identical evaluation protocol.

The dataset conveniently ships with its own reference points — SVC and Random Forest at 55%, VGG19 at 64.16%, **ResNet50 at 74.16%** — so we do not have to build the weak baselines ourselves; we can position against published numbers and add our own fine-tuned ResNet as the serious contender.

There is also a published UniCAS number to compare against: the UniCAS paper reports AUCs of **99.19% / 98.54% / 99.79%** (NILM/LSIL/HSIL) on BMT. But that was obtained with **their multi-task aggregator**, not a plain probe on frozen features, so our numbers will not be directly comparable unless we replicate their setup. Treating that gap explicitly — *frozen features + simple head* versus *their full adaptation* — is itself a worthwhile experiment, since it separates the value of the representation from the value of their aggregator.

### 2.3 Domain match, and the cross-dataset experiment it enables

BMT is **ThinPrep**, which is exactly what the clinical team will deliver for our project, and exactly what UniCAS was pretrained on. That triple alignment is rare and is the main reason to prefer BMT over SIPaKMeD or Herlev as our principal public evaluation set.

It also sets up a concrete generalization study for later in the schedule: train or probe on BMT and test on our own ThinPrep slides (same preparation, different laboratory, different scanner, different population), which isolates **acquisition shift** from **preparation shift**. The authors' own 60.29% figure gives us the expected magnitude of the latter, and their usage note states the rule directly: *"machine learning models must be trained on images that have been collected from slides prepared with the same preparation format as the intended test slides."*

---

## 3. Caveats — what BMT cannot validate

Being explicit about the limits matters, because BMT is small and curated and it would be easy to over-read a good result.
- **These are not WSIs.** They are individually captured fields of view, so BMT cannot exercise any slide-level pipeline — no tiling, no tissue detection, no MIL aggregation. Whatever we validate here is *region-level* only. (Worth noting that the UniCAS paper describes its BMT evaluation as "slide-level grading" over 600 images from 180 slides; with 1–5 images per slide, what "slide-level" means there is ambiguous and would need checking against their code.)
- **The test set is tiny.** The authors' own 60:20:20 split leaves **120 test images**, so a single misclassification moves accuracy by ~0.8 points and their reported differences between models sit well within noise. Any evaluation we run should use cross-validation over the full 600 rather than a single split.
- **Three classes only, and the hard one is missing.** ASC-US — the category with the worst interobserver agreement and arguably the most clinically consequential ambiguity — is deliberately excluded. Good performance on BMT therefore says nothing about the hardest part of the Bethesda scale. This connects directly to UniCAS's own weakest reported result (67.53% accuracy on ASC-US in fine-grained grading).

---

[← Back to Week 2](README.md)
