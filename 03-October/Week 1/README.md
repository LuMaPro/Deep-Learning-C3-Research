# October — Week 1


## Overview

| # | Activity | Output | Hours |
|---|---|---|---|
| 1 | t-SNE visualization of UniCAS embeddings from an in-house slide | [TSNE_embeddings.ipynb](TSNE_embeddings.ipynb) | 5 h |
| 2 | Evaluation on the BMT dataset — ABMIL over UniCAS and ResNet embeddings | [UniCAS_BMT.ipynb](UniCAS_BMT.ipynb) · [ResNet_BMT.ipynb](ResNet_BMT.ipynb) | 12 h |

---

### 1. t-SNE of UniCAS embeddings — 5 h

Graphical visualization of the embeddings of a sample slide provided by Prof. Michelle (`.kfb`).

**Tile extraction:** the slide goes through the extraction pipeline from [September — Week 4](../../02-September/Week%204/README.md) (`extract_tiles`), yielding **2,006 tiles** of `224 × 224` at 20×.

**Backbone:** **UniCAS** (ViT-L/16, frozen, ImageNet normalization) extracts one **1024-d embedding** per tile.

**t-SNE:** `tsnecuda` reduces the `2006 × 1024` matrix to 2-D (`perplexity = 30`, `learning_rate = 500`) for the scatter plot.

**Result:**

```text
Cosine similarity between pairs: mean 0.266 ± 0.089 (min -0.038, max 0.803)
```

- The t-SNE shows a **single diffuse cloud**, with no clear clusters
- The low mean similarity shows that UniCAS separates the tiles from each other

### 2. Evaluation on BMT — UniCAS vs. ResNet — 12 h

#### Dataset

[BMT](../../01-August/Week%203/Paper-BMT.md): **600 ThinPrep images**, **200 per class** (`NIL`, `LSIL`, `HSIL`).

| Split | Images per class | Total |
|---|---|---|
| Train | 120 | 360 |
| Validation | 40 | 120 |
| Test | 40 | 120 |

#### Why tiles instead of resizing

Resizing `1920 × 1080 → 224 × 224` implies:

- **~41× fewer pixels** (`2,073,600 → 50,176`): each cell shrinks to a few pixels, and the **nuclear detail** (size, chromatin, N:C ratio) that separates LSIL from HSIL is lost
- **Aspect ratio distortion** (`16:9 → 1:1`) deforms the cells

So each image is cut into `224 × 224` **tiles** at native resolution.

#### ABMIL

**Attention-based Multiple Instance Learning** (Ilse et al., 2018): the image is a **bag** of tiles with **only one label for the bag**. An attention module learns a weight for each tile (softmax over the tiles of the bag), and the bag embedding is the **weighted sum** of the tile embeddings, followed by a classifier.

#### Pipeline

```text
image → N tiles (224 × 224) → frozen backbone → embeddings [N × d] (saved to disk) → ABMIL → NIL / LSIL / HSIL
```

#### Backbones

| | UniCAS | ResNet |
|---|---|---|
| Model | ViT-L/16, cytology foundation model | ResNet-18, ImageNet pretrained |
| Embedding dim (`d`) | 1024 | 512 |
| Weights | frozen | frozen |

#### Architecture (ABMIL)

Gated attention:

- `proj` → `Linear(d, 256)` + ReLU + Dropout(0.25)
- `att_V` → `Linear(256, 128)` + Tanh
- `att_U` → `Linear(256, 128)` + Sigmoid
- `att_w` → `Linear(128, 1)`, softmax over the tiles
- `classifier` → `Linear(256, 3)`

#### Parameters

| | |
|---|---|
| Training loss | `CrossEntropyLoss` |
| Optimizer | Adam, `lr = 1e-3` |
| Batch size | 1 bag |
| Max epochs | 50 |
| Early stopping | patience 10 (validation loss), best checkpoint saved |

#### Results

| Test AUC | UniCAS | ResNet-18 |
|---|---|---|
| NIL | **0.998** | 0.984 |
| LSIL | **0.992** | 0.938 |
| HSIL | **0.994** | 0.978 |
| Best val. loss | **0.098** (epoch 5) | 0.300 (epoch 15) |

- **UniCAS is better in all three classes**, and converges faster
- The largest gap is in **LSIL** (`0.992` vs `0.938`), the intermediate class
