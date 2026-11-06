# Paper Notes — Deep Learning for Cervical Cytology Screening: Methods, Algorithms, Challenges and Prospects

> Jiang, Y.; Zheng, N.
> *arXiv* preprint **2611.01730** (cs.CV), submitted November 1, 2026. Link: <https://arxiv.org/html/2611.01730>
> Qiushi Academy for Advanced Studies & College of Computer Science and Technology, Zhejiang University

## 1. Taxonomy

The review organizes the field along 3 axes: **data scale** (cell → patch → WSI), **function** (segmentation, feature extraction, classification, detection) and **supervision** (full, weak, semi, self-supervised, few-shot).

| Scale | Input | Main tasks |
|---|---|---|
| Cell | single-cell crops, ~64 × 64 to 128 × 128 px | nucleus/cytoplasm segmentation, features, Bethesda classification |
| Patch | multi-cell regions, ~256 × 256 to 1024 × 1024 px | instance segmentation, abnormal cell detection |
| WSI | gigapixel slide, ≥ 15,000 × 15,000 px | slide-level classification, aggregation, lesion localization |

Two clinical roadmaps for WSI screening:

- **Roadmap 1 — WSI confidence aggregation:** patches/ROIs are scored, top-K most suspicious are aggregated (mean, voting, learned fusion / MIL) into a slide label (needs only slide or patch labels).
- **Roadmap 2 — Cell detection and classification:** a detector localizes and classifies each abnormal cell with its Bethesda category (needs dense box + class annotations).

Four "generations" of methods: handcrafted features + classical classifiers → deep networks replacing pipeline stages → end-to-end detection (no explicit segmentation) → foundation models and label-efficient learning.

## 2. Datasets

Public datasets are small, mostly single-center and with heterogeneous labels. Herlev and SIPaKMeD (< 5k cells) are still the most used, and there is **no public multi-center WSI benchmark with fixed splits and clinical outcomes** (like CAMELYON / TCGA).

- **Herlev / SIPaKMeD:** single-cell benchmarks, saturated (> 99% accuracy), useful only as a sanity check.
- **ISBI14/15, CERVIX93:** the reference for **overlapping-cell** segmentation
- **Tianchi:** one of the first public WSI-level sets, but with coarse ROI labels
- **Cx22:** full instance annotations, enables comparing instance segmentation methods
- **HiCervix:** largest annotated single-cell TCT set
- **APACC / RIVA:** multi-annotator, allows inter-observer analysis, but Pap smear
- **CCS-127K:** largest WSI collection (used by Smart-CCS), not public
- **CCBench:** benchmark for vision-language models, not for training

## 3. Cell-level methods

#### 3.1 Segmentation

Delineates nucleus and cytoplasm.

- **U-Net variants:** PGU-net+ (progressive U-Net with shape residuals, DSC, URestNet (two-stage residual U-Net), VGG16 + HSDC decoder, D-MEM (dense deformable U-Net ensemble)
- **FCN:** FCN-8 + FCN-16 ensemble
- **Mask R-CNN:** contour segmentation
- **Hybrid:** CNN + classical refinement (superpixels/SLIC + Graph-Cut, level sets, CRF, Random Walk, shape priors)
- **Recent:** attention U-Nets, Transformers and **VMamba**

#### 3.2 Feature extraction

Three families: handcrafted (shape, color, texture), deep features and hybrid fusion. The **N/C ratio** is the most diagnostic morphological feature in Bethesda.

- **Handcrafted:** shape: (area, perimeter, eccentricity, N/C ratio), color: (brightness, color moments), texture: (LBP, GLCM, Haralick, HoG)
- **Deep features:** ImageNet-pretrained VGG, ResNet, Inception, Xception used as frozen extractors
- **Fusion:** concatenation of several CNNs (VGG16 + VGG19 + ResNet50 + Xception) or deep + handcrafted (GLCM + Gabor + CNN)
- **Main limitation:** domain shift: ImageNet features do not capture Papanicolaou texture/color well, which motivates **self-supervised pretraining on cytology**

#### 3.3 Classification

Assigns cells to Bethesda classes. Single-cell accuracy on Herlev/SIPaKMeD has **plateaued above 99%**, and the frontier moved to multi-cell classification, rare classes (AGC, AIS) and generalization across centers.

- **VGG16/19:** ~94–95% (5 and 7 classes on Herlev)
- **AlexNet** (DeepPap): ~93% accuracy, 7 classes, efficient baseline
- **ResNet:** best single models, consistently > 95%
- **Stacked autoencoders:** 97.8% on Herlev
- **Ensembles:** consistently better than single models (Inception + Xception + DenseNet, EfficientNet + MobileNet + ...)
- **Multi-cell with attention:** classifies patches with several cells
- **HiCervix** (hierarchical): much lower scores than binary benchmarks, closer to the real difficulty

## 4. Patch-level methods

End-to-end models that skip the explicit segmentation step.

#### 4.1 Instance segmentation

- **Mask R-CNN:**  classification + localization + mask
- **Mask R-CNN + NAS-FPN:**  searched feature pyramid + soft-NMS for overlapping cells
- **MMT-PSM:**  semi-supervised knowledge distillation with two Mask R-CNNs (teacher/student)
- **Recent:**  PointRend, Mask2Former, SAM and CellViT adaptations. Little cervical-specific evidence so far

#### 4.2 Object detection

Two-stage detectors keep a precision advantage, one-stage ones are faster.

- **One-stage:**  improved **SSD** with stronger multi-scale connections + center loss, **YOLOv3** with an image-level classification branch to suppress false positives or with a cascaded **InceptionV2** classifier for hard classes
- **Two-stage:**  **Faster R-CNN** baselines, SPFNet (modified FPN + SE attention), GPRB (compares proposals to class reference samples), global-context Faster R-CNN
- **Cascade R-CNN + HRNet:**  pairwise sampling to reduce false positives
- **Semi-supervised:**  teacher-student Faster R-CNN
- **Recent:**  Transformer (DETR-like) detectors, graph/hypergraph modules for inter-cell context, active learning

## 5. WSI-level methods

Gets a slide diagnosis from the gigapixel image. Almost every pipeline first selects suspicious regions/cells, then aggregates. However, since 2024, the field moved to **MIL and graph networks**, but almost all of it is on **private cohorts**.

```text
WSI → patches → patch/cell model (CNN or detector) → scores / features → top-K selection
→ aggregation (mean, SVM, RNN, attention, GNN, MIL) → slide label
```

- **3-stage cascaded CNN** (Gupta): background filter → artifact/noise filter → abnormality classifier
- **CNN + RNN** (Kanavati): patch CNN, RNN aggregation
- **RetinaNet + GAT** (Zhang): top-K and bottom-K patches as graph nodes + supervised contrastive learning
- **Detection-based** (Roadmap 2): **AttFPN** (channel + spatial attention), **YOLOv3 + Xception + U-Net**
- **Twin-MIL:** refines a cell detector using only slide-level labels (weakly semi-supervised)

## 6. Learning paradigms and foundation models

- **MIL:** the slide is a bag of patches with only the slide label.
- **Semi-supervised:** small labeled set + large unlabeled set (SCAC, teacher-student)
- **Self-supervised:** pretraining on unlabeled cytology (contrastive learning approach)
- **Foundation models:** **UniCAS** (48,532 TCT WSIs, TransMIL aggregation)
- **Few-shot / prototypical:** proposed for rare classes (AGC, AIS), clinical value not yet shown
- **Domain adaptation:** test-time adaptation (Smart-CCS), color normalization, stain augmentation, domain-adversarial training (scanner/stain/center shift is the main barrier to deployment)

## 7. Evaluation

- **Classification:** accuracy, precision, recall, F1, sensitivity, specificity (accuracy alone is misleading because prevalence of abnormal cases is low)
- **Segmentation:** DSC, IoU and **AJI** (for overlapping cells)
- **Detection:** mAP and AP50
- **Instance segmentation:** Panoptic Quality (PQ)
- **WSI:** AUC, sensitivity at fixed specificity, calibration