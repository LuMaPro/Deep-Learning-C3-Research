# August — Week 4


## Overview

| # | Activity | Output | Hours |
|---|---|---|---|
| 1 | Dimensionality reduction — t-SNE on MNIST CNN embeddings with tsne-cuda | [MNIST_TSNE.ipynb](MNIST_TSNE.ipynb) | 5 h |
| 2 | Interpretability — Grad-CAM on a pre-trained ResNet-50 | [GradCAM.ipynb](GradCAM.ipynb) | 4 h |

---

### 1. t-SNE on MNIST embeddings — 5 h

Projecting the embeddings of the CNN trained in Week 3 to 2D with [tsne-cuda](https://github.com/CannyLab/tsne-cuda), the CUDA implementation from CannyLab, and plotting the 60,000 training samples coloured by digit.

### 2. Grad-CAM on a pre-trained ResNet-50 — 4 h

Producing a class activation heatmap over the last convolutional block of an ImageNet-pretrained ResNet-50, validated on a single test image correctly classified as ImageNet class 155 (Shih-Tzu).

A second module replicates the ResNet forward pass up to `layer4`, so the activations are read before the last non-linearity. The heatmap weights each activation channel by the mean gradient of that block, averages over channels, then upsamples and blends the result over the original image.