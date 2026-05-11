# Zero-Shot Learning Report:

## Reproducibility Instructions

### Environment Requirements
- **CUDA 13.0** used
- Python dependencies (install via `requirements.txt`)

```bash
pip install -r requirements.txt
```

### Run Command
```bash
python validate.py --data_dir ./data --batch_size 16 --n_batches 512 --output results.json
```

## Final Solution Description

### What Modified

1. `zo_optimizer.py`
2. `head_init.py`

### Final Approach

1. **Hyperparameters**: `lr = 3e-4`, `eps = 3e-4`
2. **Budget**: 16 × 512 (best configuration found)

In `zo_optimizer.py`, I replaced the 2-point central-difference gradient estimator, which perturbs each parameter individually, with the SPSA algorithm, which adds random noise to all weights at once and provides an asymptotically unbiased gradient estimate. SPSA works by evaluating the loss at two perturbed points per step (w + c·Δ and w − c·Δ) and estimating the gradient from their difference, requiring only two loss evaluations regardless of parameter count. This is critical for models with a large number of parameters, as the 2-point method scales linearly with the number of weights. Other optimizer modifications did not yield improvements.

In `head_init.py`, I implemented a mapping from ImageNet classes to CIFAR-100 and initialized the head as the average of the corresponding classes with added random noise. This provided the largest improvement in metrics. Noise is added to give the optimizer room to search for a local minimum.

---

## Experiments and Failed Attempts

### Gradient Estimate

First, I replaced the gradient estimator.

| Configuration | Metric |
|---------------|--------|
| Fine-tune model (no improvements) | 1.36 |
| + SPSA | 1.48 |

---

### Head Initialization Experiments

| Initialization Strategy | Checkpoint 2 | Checkpoint 3 |
|------------------------|----------------|---------------|
| Kaiming Uniform + zeros bias | 1.36 | 1.48 |
| Xavier Uniform + zeros bias | 1.36 | 1.46 |
| Orthogonal + zeros bias | 1.45 | 1.30 |
| Kaiming Uniform + constant bias (0.001) | 1.36 | 1.41 |
| Uniform + constant bias (0.01) | 1.36 | 1.26 |
| Label mapping (mean only) | 22.24 | 23.13 |
| **Label mapping + random noise** | **22.59** | **23.14** |

Label mapping completely transforms performance, jumping from ~1.3 to ~23.0. Adding noise gives small but consistent improvement.

---

NOTE! All subsequent metrics were obtained by running only Checkpoint 3 (Fine-tuned) without running Checkpoint 2 and Checkpoint 1, therefore the final metric value differs from the best one in the experiments.

### Hyperparameter Search

| lr | eps | Metric |
|----|-----|--------|
| 1e-4 | 3e-4 | 0.2487 |
| 1e-4 | 1e-3 | 0.2481 |
| 1e-4 | 3e-3 | 0.2482 |
| **3e-4** | **3e-4** | **0.2675** |
| 3e-4 | 1e-3 | 0.2673 |
| 3e-4 | 3e-3 | 0.2661 |
| 1e-3 | 3e-4 | 0.2268 |
| 1e-3 | 1e-3 | 0.2267 |
| 1e-3 | 3e-3 | 0.2268 |


---

### Optimizer Comparison

All experiments with `lr=1e-4, eps=1e-4`

| Optimizer | Metric |
|-----------|--------|
| SGD (baseline) | **0.2675** |
| SGD + Momentum (0.9) | 0.0631 |
| Adam (β1=0.95 β2=0.999) | 0.2382 |

---

### Layer Selection

| Trainable Layers | Metric |
|-----------------|--------|
| `fc.weight`, `fc.bias` (full head) | **0.2675** |
| `fc.weight` only | 0.2495 |
| `layer4.1.conv2.weight`, `fc.weight`, `fc.bias` | 0.1515 |

---

### Budget Split

| batch_size | n_batches | Total Samples | Metric |
|------------|-----------|---------------|--------|
| 64 | 128 | 8192 | 0.2529 |
| 32 | 256 | 8192 | 0.2675 |
| **16** | **512** | **8192** | **0.2788** |
| 8 | 1024 | 8192 | 0.2262 |


---

### Subsample

Next, I decided to train the model on a balanced subsample. The table shows how the metrics changed depending on the number of samples taken from each class. Unfortunately, this did not bring any improvement.Apparently, random subsamples better reflect the data distribution.

| Steps | Metric |
|-------|--------|
| 1 | 0.2555 |
| 5 | 0.2611 |
| 8 | 0.2741 |
| 10 | 0.2750 |
| 20 | 0.2783 |
| 40 | 0.2773 |
| 80 | 0.2782 |

---

### Data Augmentation

I tried applying augmentations, but this also did not bring any improvement. Augmentations usually help when the model is overfitting, but in our case the model is heavily underfitting.

| Augmentation | Metric |
|--------------|--------|
| baseline | 0.2788 |
| baseline + RandomAffine + ColorJitter | 0.2413 |
