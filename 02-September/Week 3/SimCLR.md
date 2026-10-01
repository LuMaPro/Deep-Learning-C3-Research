# Paper Notes — SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

> Review — SimCLR: A Simple Framework for Contrastive Learning of Visual Representations (Sik-Ho Tsang) https://sh-tsang.medium.com/review-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-5de42ba0bc66

## 1. Summary

A **self-supervised** method that learns representations **without labels** by maximizing agreement between two augmented views of the same image.

Main findings:

- **Composition of augmentations** is important
- A **nonlinear projection head** between the representation and the loss substantially improves quality
- Contrastive learning benefits from **larger batch sizes** and **longer training**

### 1.1 Framework

```text
x ─┬─ t(x) → x̃ᵢ → f(·) → hᵢ → g(·) → zᵢ  ─┐
   │                                      ├─ maximize agreement (NT-Xent)
   └─ t'(x) → x̃ⱼ → f(·) → hⱼ → g(·) → zⱼ ─┘
```

| Component | What it is |
|---|---|
| Data augmentation `t` | Random crop + resize, random color distortion, random Gaussian blur |
| Base encoder `f(·)` | **ResNet**; `h` = output after average pooling (2048-d for ResNet-50) |
| Projection head `g(·)` | MLP with one hidden layer: `z = W₂ σ(W₁ h)`, σ = ReLU |
| Contrastive loss | **NT-Xent** on `z` |

After pretraining, **`g(·)` is discarded** and `h` is used as the representation for downstream tasks.

### 1.2 Contrastive loss (NT-Xent)

A minibatch of **N** images produces **2N** augmented views. For each positive pair, the other **2N−2** views in the batch are the **negatives**:

$$
\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\mathrm{sim}(z_i, z_k)/\tau)}
$$

- `sim` = **cosine similarity**, `τ` = **temperature**
- Final loss averaged over all positive pairs, both `(i, j)` and `(j, i)`

### 1.3 Ablations

| Study | Finding |
|---|---|
| Augmentations | **random crop + color distortion** performed better (no single transformation suffices) |
| Projection head | Loss on `z` (nonlinear head) is better than on `h` (experimental) |
| Batch size / epochs | Larger batches and longer training help (bigger batch = more negatives) |

### 1.4 Results

| Evaluation | Result |
|---|---|
| Linear evaluation on ImageNet | **ResNet-50 (4×)** with SimCLR **matches a supervised ResNet-50** |
| Few labels (1% and 10% of ImageNet) | Significantly outperforms previous state of the art |
| Transfer learning (fine-tuned, 12 datasets) | Better than supervised baseline on **5**, worse on **2** (Pets, Flowers), statistically tied on **5** |

---