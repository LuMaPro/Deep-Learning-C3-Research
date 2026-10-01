# September — Week 4


## Overview

| # | Activity | Output | Hours |
|---|---|---|---|
| 1 | Reading — DINOv2, self-supervised general-purpose visual features | [DINOv2.md](DINOv2.md) | 3 h |
| 2 | Contrastive learning on synthetic 2-D data | [CLR.ipynb](CLR.ipynb) | 5 h |
| 3 | Tile extraction on `.kfb` slides — `extract_tiles` function | [TileExtraction.ipynb](TileExtraction.ipynb) | 8 h |

---

### 1. Reading — DINOv2 — 3 h

*DINOv2: Learning Robust Visual Features without Supervision* (Oquab et al., Meta AI, 2023).

Main points:

- The goal is **features that work frozen** — a linear classifier or k-NN on top of the backbone, with **no finetuning**
- **LVD-142M**: an automatic curation pipeline (embedding, deduplication, retrieval by k-NN against curated sources)
- Training is **student–teacher** (teacher = EMA of the student): **DINO** (class token) and **iBOT** (patch level)
- A ViT-g/14 with 1B parameters is trained and then **distilled** into smaller models

### 2. Contrastive learning on synthetic data — 5 h

**Data:** three Gaussian clusters of 500 points each in 2-D (centres `(-2.5,-2.5)`, `(0,0)`, `(2.5,2.5)`)

**Encoder:** `Linear(2,32) → ReLU → Linear(32,2)`

**Training:** for each batch of 64, two **views** are made by adding Gaussian noise (`0.4 · randn`) to the same points. Both are encoded, `L2`-normalised, and the similarity matrix `Z₁ Z₂ᵀ / τ` (τ = 1) is fed to a `cross_entropy` (targets are the **diagonal indices**)

**Result:** before training, the three groups are mixed in the embedding space. After training, they stretch along an arc, with each cluster occupying its own region and the overlap reduced

### 3. Tile extraction on `.kfb` slides — 8 h

The only syntactic change from previous tile excraction (week 3) is **how the slide is opened**:

```python
import openslide, kfbslide

s = kfbslide.OpenSlide(Path(r"...CYT.kfb"))
```

#### 3.1 Why 20% of coverage

The tile is now `224` instead of `320`, so the same *absolute* amount of cells fills a smaller field. However, more importantly, in cytology the material is sparse, and a tile with a handful of well-spread cells is exactly what we want to keep

#### 3.2 Discarding empty regions and dirty tiles

Two filters run before a tile is saved:

- **Outside the scanned area:** `read_region` returns RGBA, and regions that fall outside what the scanner captured come back with transparent pixels, so any tile with `alpha.min() == 0` is dropped
- **Dirty tiles:** Some tiles pass the coverage rule with **black/gray pixels**, which usually show up along one of the slide's borders and are most likely dirt. The filter therefore looks at the **colour** of the pixels the mask selected: it takes their **median hue**, and drops those dirty tiles

#### 3.3 The `extract_tiles` function

Everything above is wrapped into `extract_tiles(slide_path, save_path, tile_size=224)`:

- Opens the slide and **rejects it** if it is not a 40× scan with a `2×` level 1
- Computes the **Otsu threshold** of the whole slide on the thumbnail
- Sweeps the grid at **level 1 (20×)**, reading `224 × 224` tiles with a stride of `448` in level-0 coordinates
- Skips the tile if it is **outside the scanned area**, below **20% coverage**, or **dirty**
- Saves each accepted tile as a `.png` file

`extract_tiles` is the first piece of the project's pipeline (it will be used exactly as written).
