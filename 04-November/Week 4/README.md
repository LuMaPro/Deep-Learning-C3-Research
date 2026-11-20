# November — Week 4


## Overview

| # | Activity | Output | Hours |
|---|---|---|---|
| 1 | UniCAS slide-level (multi-task) inference on the in-house slides | [UniCAS_multitask.ipynb](UniCAS_multitask.ipynb) | 20 h |

---

### 1. UniCAS multi-task inference — 20 h

#### 1.1 Setting up UniCAS slide-level inference

Code taken from the [UniCAS repository](https://github.com/peter-fei/UniCAS) (`multitask_agg.py` and `slide_multitask.py`), which is the UniCAS **multi-task pipeline** used for WSI inference.

The only change is in `TaskAttention`, with one line added after the top-k attention is computed, so the attention of the inference is stored and the top-k tiles can be traced:

```python
atten_topk.scatter_(-1, top_k_indice, top_k_value_softmax)
self.last_attn = atten_topk.detach()     # [NEW]
```

#### 1.2 Attention (How to get top-k tiles for cancer?)

This section shows how to retrieve the top-k tiles of the inference, using a random input `torch.randn(1, 5, 1024)` (5 tiles):

- `last_attn` has shape `[1, 12, 3, M]`: **12 attention heads**, each with **3 vectors** (cancer / candidiasis / clue cell)
- Each vector has size `M`, the **closest perfect square** to the number of tiles (e.g. 5 tiles → `M = 9 = 3²`)
- Only the first `N` scores matter (`N` = number of tiles).
- The cancer score of each tile is the sum of the cancer vector over the 12 heads (last block), and the **top-k tiles** are the `k` highest scores

#### 1.3 Sanity check on UniCAS's own examples

Check that the model and the weights (`multi-task.pth`) were loaded and used correctly for inference, using the [example embeddings](https://github.com/peter-fei/UniCAS/tree/main/slide-level/Pathology_UniCAS_patch) of the UniCAS repository.

#### 1.4 UniCAS inference on in-house slides

Inference of the UniCAS multi-task model on the slides labeled by Prof. Michelle ([Week 3](../Week%203/README.md)), using the UniCAS embeddings of all tiles of each slide.

#### Results

| Slide | Label | Cancer (prediction) |
|---|---|---|
| 2024-10-14_09_45_11CYT | Normal | **Positive** |
| 2024-10-14_08_18_15CYT | Normal | **Positive** |
| 2024-10-14_11_07_25CYT | Normal | **Positive** |
| 2024-10-14_09_39_28CYT | Normal | **Positive** |
| 2024-10-14_09_56_46CYT | Normal | **Positive** |
| 2024-10-14_09_30_47CYT | Normal | **Positive** |
| 2024-10-14_10_22_39CYT | LSIL | **Positive** |
| 2024-10-14_11_13_09CYT | HSIL | **Positive** |
| 2024-10-14_08_34_52CYT | HSIL | **Positive** |
| 2024-10-14_11_18_59CYT | HSIL | **Positive** |

<details>
<summary><b>2024-10-14_09_45_11CYT</b> — Normal</summary>

| Task | Logit 0 (negative) | Logit 1 (positive) | Prediction |
|---|---|---|---|
| Cancer | -3.332 | 3.005 | Positive |
| Candidiasis | 0.750 | -1.430 | Negative |
| Clue cell | -2.443 | 2.475 | Positive |

</details>

<details>
<summary><b>2024-10-14_08_18_15CYT</b> — Normal</summary>

| Task | Logit 0 (negative) | Logit 1 (positive) | Prediction |
|---|---|---|---|
| Cancer | -2.874 | 2.672 | Positive |
| Candidiasis | -1.975 | 1.396 | Positive |
| Clue cell | -2.688 | 2.369 | Positive |

</details>

<details>
<summary><b>2024-10-14_11_07_25CYT</b> — Normal</summary>

| Task | Logit 0 (negative) | Logit 1 (positive) | Prediction |
|---|---|---|---|
| Cancer | -3.411 | 3.089 | Positive |
| Candidiasis | 0.224 | -0.718 | Negative |
| Clue cell | -1.555 | 1.692 | Positive |

</details>

<details>
<summary><b>2024-10-14_09_39_28CYT</b> — Normal</summary>

| Task | Logit 0 (negative) | Logit 1 (positive) | Prediction |
|---|---|---|---|
| Cancer | -3.394 | 3.077 | Positive |
| Candidiasis | -0.654 | 0.131 | Positive |
| Clue cell | -1.022 | 0.927 | Positive |

</details>

<details>
<summary><b>2024-10-14_09_56_46CYT</b> — Normal</summary>

| Task | Logit 0 (negative) | Logit 1 (positive) | Prediction |
|---|---|---|---|
| Cancer | -3.187 | 2.898 | Positive |
| Candidiasis | 0.138 | -0.724 | Negative |
| Clue cell | 0.677 | -0.388 | Negative |

</details>

<details>
<summary><b>2024-10-14_09_30_47CYT</b> — Normal</summary>

| Task | Logit 0 (negative) | Logit 1 (positive) | Prediction |
|---|---|---|---|
| Cancer | -3.315 | 3.018 | Positive |
| Candidiasis | 0.754 | -1.215 | Negative |
| Clue cell | 1.958 | -1.471 | Negative |

</details>

<details>
<summary><b>2024-10-14_10_22_39CYT</b> — LSIL</summary>

| Task | Logit 0 (negative) | Logit 1 (positive) | Prediction |
|---|---|---|---|
| Cancer | -3.259 | 2.966 | Positive |
| Candidiasis | 1.447 | -1.962 | Negative |
| Clue cell | -3.020 | 2.992 | Positive |

</details>

<details>
<summary><b>2024-10-14_11_13_09CYT</b> — HSIL</summary>

| Task | Logit 0 (negative) | Logit 1 (positive) | Prediction |
|---|---|---|---|
| Cancer | -3.445 | 3.106 | Positive |
| Candidiasis | 0.682 | -1.236 | Negative |
| Clue cell | -2.675 | 2.818 | Positive |

</details>

<details>
<summary><b>2024-10-14_08_34_52CYT</b> — HSIL</summary>

| Task | Logit 0 (negative) | Logit 1 (positive) | Prediction |
|---|---|---|---|
| Cancer | -3.230 | 2.933 | Positive |
| Candidiasis | 0.197 | -0.851 | Negative |
| Clue cell | -1.725 | 1.807 | Positive |

</details>

<details>
<summary><b>2024-10-14_11_18_59CYT</b> — HSIL</summary>

| Task | Logit 0 (negative) | Logit 1 (positive) | Prediction |
|---|---|---|---|
| Cancer | -3.348 | 3.049 | Positive |
| Candidiasis | -0.388 | -0.104 | Positive |
| Clue cell | -1.801 | 1.841 | Positive |

</details>

#### Next steps

The cancer logits of all slides (e.g. `-3.33 / 3.01`, `P(cancer) ≈ 0.998`) are almost the same as the ones of the random tensor of section 1.2 (`-3.40 / 3.03`, `P(cancer) ≈ 0.998`), so the model classifies every slide as cancer, including the Normal ones. This is probably caused by **domain shift**: Prof. Michelle's slides may differ from the UniCAS training data in scanner, preparation and other aspects. Therefore, for better conclusions, enough slides are needed to train and repeat the pipeline.

While Prof. Michelle's slides are being prepared, one possible approach is to obtain the slides of the A.C. Camargo Cancer Center paper ([Martins et al.](../Week%203/Paper-Martins.md)), and then repeat the multi-task pipeline with training, as well as [kfb_inference.ipynb](../Week%202/kfb_inference.ipynb) with the ABMIL aggregator trained on these slides.
