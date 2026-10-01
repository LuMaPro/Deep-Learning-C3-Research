# Paper Notes — HMCHH-TCT-CellDet: A Large Annotated Cervical Cytology Images Dataset for AI Models to Aid Cervical Cancer Screening

> Zhang, X.; Ji, J.; Zhang, Q.; Zheng, X.; Ge, K.; Hua, M.; Cao, L.; Wang, L.
> *Scientific Data* **2025**, 12, 23. Received August 28, 2024; accepted January 1, 2025; published online January 7, 2025. DOI: [10.1038/s41597-025-04374-5](https://doi.org/10.1038/s41597-025-04374-5) · PMID: 39774182
> <https://www.nature.com/articles/s41597-025-04374-5>

## 1. Summary

**15,761 bounding boxes** over **8,037 patches** cut from **129 digitized ThinPrep slides** (the task it supports is **detection**, not classification).

### 1.1 The gap it fills

- **Scarcity:** "The main reason is the scarcity of publicly available image datasets containing annotated abnormal cervical cells"
- **Generalization:** "Performance may be inconsistent when dealing with images derived from different centres, instruments, or staining techniques"

### 1.2 What the dataset is

| Attribute | Value |
|---|---|
| Source slides | **129 TCT (ThinPrep cytologic test) slides**, all "**reported as abnormal levels**" |
| Institution | **Heilongjiang Maternal and Child Health Hospital (HMCHH)**, Harbin, China — collected **Oct 2018 – May 2019** |
| Digitization | Each slide → **WSI** → sliding window → **333 non-overlapping patches** per slide |
| Images released | **8,037 patches** (after removing "image patches with low information content, such as those covered by background or blurred patches") |
| Image size / format | **2048 × 2048 px, PNG** |
| Optics | **Olympus BX53** optical microscope, **20× objective** |
| Annotations | **15,761 bounding boxes**, average of **~2 abnormal cells per image**; Pascal-VOC-style **XML**, one file per image |
| Classes | **One class: `abnormal`** |
| Preparation | **ThinPrep (TCT)** — liquid-based |
| Staining | `not reported` |
| Ethics | Approval **2022ZFYJ295-01**; informed-consent waiver, irreversibly anonymized, Declaration of Helsinki |

### 1.3 The annotation protocol

| Step | Who | What |
|---|---|---|
| Initial labelling | **Reader B or C** (~10 years' experience each), randomly assigned | Draws boxes around abnormal cells in each patch using **Colabeler** |
| Verification | **the other of B / C** | Reviews the image and the annotation |
| Final check + export | **Reader A** (~**33 years** reading cervical cytology) | Checks and exports all annotation files |

### 1.4 Technical validation

20% of *patients* held out as the test subset, while the remaining patients were split into **five folds** for cross-validation.

| Method | AP₅₀:₉₅ | AP₅₀ | AP₇₅ | AR₅₀:₉₅ | F1-score |
|---|---|---|---|---|---|
| SSD | 10.8 | 24.1 | 7.2 | 14.3 | 40.1 |
| RetinaNet | 25.7 | 54.8 | 20.3 | 34.4 | 66.5 |
| FCOS | 27.6 | 61.6 | 21.2 | 37.7 | **68.9** |
| **Faster R-CNN** | **31.5** | **67.4** | 25.0 | **45.2** | 58.9 |
| Cascade R-CNN | 29.1 | 57.7 | **27.0** | 39.2 | 66.4 |
| Sparse R-CNN | 23.2 | 50.1 | 19.0 | 32.0 | 65.8 |
| YOLOv3 | 9.1 | 28.3 | 2.8 | 17.6 | 46.3 |
| YOLOv7 | 13.3 | 37.3 | 5.4 | 22.6 | 55.3 |
| DETR | 19.4 | 47.1 | 12.2 | 36.5 | 43.7 |


### 1.5 Availability

- **Data:** **figshare**, DOI `10.6084/m9.figshare.27901206`
- **Structure:** `JPEGImages/` (the PNG patches) and `Annotations/` (the XML files), same count in each, filenames `patient number_image number`
- **Code:** <https://github.com/zx333445/TCT_data>


## 2. Caveats 

- Only one class (`abnormal`): we can't use this for classification task (`negative`, `positive`)
- Few patches for each WSI (`~63`): a WSI can provide `~5000` patches, and we need this high volume for our project