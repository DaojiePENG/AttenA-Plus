# AttenA+: Velocity Field Action Attention for Enhancing Action-Centric Robotic Foundation Models

<div align="center">

<!-- Paper badges — replace placeholders once published -->
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-blue)](https://arxiv.org)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

</div>

---

## Overview

**AttenA+** is a paradigm-agnostic, plug-in training framework that enhances action-centric robotic foundation models by prioritizing **slow, precision-demanding action steps** during training.

Existing frameworks treat all action timesteps equally during training, ignoring the inherent physical hierarchy of robotic manipulation:

> _Slow movements → critical, precision-demanding steps (grasping, precise placement)_
> _Fast movements → transitional, error-tolerant steps (repositioning, coarse approach)_

AttenA+ constructs a **velocity field** over the action sequence and assigns higher learning weights to low-velocity timesteps. This single modification can be seamlessly plugged into both discriminative (VLA) and generative (flow-matching, WAM) models **without any architectural changes**.

```
Action chunk:  [a₀,  a₁,  a₂,  a₃,  a₄,  a₅,  a₆,  a₇]
Speed:         [0.8, 0.9, 0.1, 0.05, 0.7, 0.6, 0.1, 0.05]
AttenA+ weight:[1.3, 1.1, 2.0, 2.0,  1.4, 1.7, 2.0, 2.0 ]
                              ^^^^ slow → high weight ^^^^
```

### Supported Base Models

| Variant | Base Model | Paradigm | Benchmark |
|---|---|---|---|
| **AttenA+OFT** | [OpenVLA-OFT](third_party/openvla-oft) (`attena+oft`) | Discriminative VLA | LIBERO, Franka |
| **AttenA+WAM** | [FastWAM](third_party/FastWAM) (`attena+wam`) | World-Action Model | RoboTwin 2.0 |
| **AttenA+π₀ / AttenA+π₀.₅** | [OpenPI](third_party/openpi) (`attena+pi`) | Generative (Flow Matching) | LIBERO |

---

## Results

### LIBERO Benchmark

Results compared with state-of-the-art methods. SR (%): average success rate across 4 task suites.

| Method | Spatial | Object | Goal | Long | **SR** | ER |
|---|---|---|---|---|---|---|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 | 2.9 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.15 | 5.85 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.23 | 4.78 |
| VLA-ADP | 99.0 | 98.2 | 96.8 | 91.2 | 96.3 | 3.7 |
| **AttenA+OFT (Ours)** | **99.0** | **100.0** | **98.8** | **96.6** | **98.6** | **1.4** |
| **AttenA+π₀ (Ours)** | **98.1** | **99.4** | **96.9** | **86.7** | **95.28** | **4.72** |

AttenA+OFT improves over OpenVLA-OFT by **+1.5% SR** and reduces error rate by **-1.5% ER**, with the largest gains on long-horizon tasks (+2.1%).

### RoboTwin 2.0 Benchmark

| Method | Embodied PT. | Clean | Rand. | **SR** | ER |
|---|---|---|---|---|---|
| π₀ | ✓ | 65.92 | 58.40 | 62.2 | 37.8 |
| π₀.₅ | ✓ | 82.74 | 76.76 | 79.75 | 20.25 |
| Motus | ✓ | 88.66 | 87.02 | 87.8 | 12.2 |
| LingBot-VA | ✓ | 92.90 | 91.50 | 92.2 | 7.8 |
| Fast-WAM | ✗ | 91.88 | 91.78 | 91.8 | 8.2 |
| **AttenA+WAM (Ours)** | **✗** | **93.3** | **92.9** | **93.1** | **6.9** |

AttenA+WAM achieves SOTA without embodied pre-training, outperforming LingBot-VA by +0.9%.

### Real-World Franka Robot Experiments

| Model | Close Draw | Put Cube | Multi-object | Long | **SR** | ER |
|---|---|---|---|---|---|---|
| OpenVLA-OFT | 100 | 96 | 90 | 84 | 92.5 | 7.5 |
| **AttenA+OFT (Ours)** | **100** | **100** | **98** | **90** | **97.0** | **3.0** |

Each task evaluated over 50 trials on a Franka manipulator. AttenA+ shows the largest gains on complex multi-object (+8%) and long-horizon (+6%) tasks.

---

## Core Algorithm

Given an action chunk $\mathbf{A} = \{a_t\}_{t=1}^T$ where $a_t \in \mathbb{R}^D$ (D=7 for LIBERO: 6 joint velocities + 1 gripper):

**Step 1 — Velocity magnitude:**
$$v_t = \|a_t^{(1:6)}\|_2$$

**Step 2 — Attention weight** (monotonically decreasing in $v_t$, clipped to $[1/c_{\max}, c_{\max}]$):
$$w_t = f(v_t), \quad w_t \in [1/c_{\max},\ c_{\max}]$$

**Step 3 — Weighted loss:**
- Discriminative (AttenA+Disc): $\theta^* = \arg\min_\theta \mathbb{E}\left[\frac{1}{T \cdot D}\sum_{t=1}^T\sum_{d=1}^D w_t \cdot |a_{t,d}^{\text{pred}} - a_{t,d}^{\text{gt}}|\right]$
- Generative / Flow Matching (AttenA+FM): $\phi^* = \arg\min_\phi \mathbb{E}\left[\frac{1}{T \cdot D}\sum_{t=1}^T\sum_{d=1}^D w_t \cdot \|u_t(\epsilon;\mathcal{I},L) - (a_{t,d}^{\text{gt}} - \epsilon_d)\|_2^2\right]$

### Weighting Strategies

| Strategy | Formula | Notes |
|---|---|---|
| `inverse` | $w = 1/v$ | Baseline, mild emphasis |
| `inverse_squared` | $w = 1/v^2$ | **Paper default** — strongly amplifies slow/fast contrast |
| `exp_decay` | $w = e^{-\alpha v}$, $\alpha=5.0$ | Fast suppression of high-speed actions |
| `log` | $w = 1/\log(1+v)$ | Gentle, noise-tolerant scaling |

---

## Installation

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/DaojiePENG/AttenA-Plus.git
cd AttenA-Plus
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Install the AttenA+ core package

```bash
pip install -e .
```

### 3. Install the target base model

**AttenA+OFT (OpenVLA-OFT):**
```bash
cd third_party/openvla-oft
conda env create -f environment.yml && conda activate openvla
pip install flash-attn --no-build-isolation && pip install -e .
```

**AttenA+WAM (FastWAM):**
```bash
cd third_party/FastWAM
pip install -e .
```

**AttenA+π₀ / π₀.₅ (OpenPI):**
```bash
cd third_party/openpi
pip install "jax[cuda12]" && pip install -e ".[dev]"
```

---

## Quick Start

```python
from attena import VelocityAttention

# Paper default: inverse_squared with clip_max=2.0
attena = VelocityAttention(
    weight_strategy="inverse_squared",  # paper default
    clip_max_weight=2.0,
    epsilon=1e-3,
    normalize_weights=True,
)

# Discriminative model training loop (OpenVLA-OFT style):
# ground_truth / predicted: (B, T, D)
loss = attena.weighted_loss(
    ground_truth=actions,
    predicted=pred_actions,
    loss_type="l1",
)

# Flow-matching model training loop (OpenPI / FastWAM style):
# target = noise - actions  (flow-matching velocity target)
loss = attena.weighted_loss(
    ground_truth=actions,    # used for speed/weight computation
    predicted=v_t,           # model's predicted velocity field
    target=u_t,              # flow-matching target (noise - actions)
    loss_type="mse",
    action_is_pad=pad_mask,  # optional (B, T) padding mask
    reduction="none",        # returns per-sample (B,) for flexible aggregation
)
```

---

## Pretrained Models

All released checkpoints are hosted on Hugging Face.

### AttenA+OFT — LIBERO (OpenVLA-OFT based)

| Task Suite | Model |
|---|---|
| LIBERO-Spatial | [attena-oft-7b-finetuned-libero-spatial](https://huggingface.co/DaojiePENG/attena-oft-7b-finetuned-libero-spatial) |
| LIBERO-Object | [attena-oft-7b-finetuned-libero-object](https://huggingface.co/DaojiePENG/attena-oft-7b-finetuned-libero-object) |
| LIBERO-Goal | [attena-oft-7b-finetuned-libero-goal](https://huggingface.co/DaojiePENG/attena-oft-7b-finetuned-libero-goal) |
| LIBERO-Long (10) | [attena-oft-7b-finetuned-libero-10](https://huggingface.co/DaojiePENG/attena-oft-7b-finetuned-libero-10) |

### AttenA+WAM — RoboTwin 2.0 (FastWAM based)

| Benchmark | Model |
|---|---|
| RoboTwin 2.0 (all 50 tasks) | [attena-wam-finetuned-head-robotwin2-all](https://huggingface.co/DaojiePENG/attena-wam-finetuned-head-robotwin2-all) |

---



- [AttenA+OFT — OpenVLA-OFT](docs/openvla_integration.md)
- [AttenA+WAM — FastWAM](docs/fastwam_integration.md)
- [AttenA+π₀ / π₀.₅ — OpenPI](docs/openpi_integration.md)

---

## Repository Structure

```
AttenA-Plus/
├── attena/                         # Core AttenA+ implementation
│   ├── __init__.py
│   └── velocity_attention.py       # VelocityAttention class
├── configs/                        # Reference training configs
│   ├── attena_openvla_libero.yaml
│   ├── attena_fastwam_libero_finetune.yaml
│   ├── attena_fastwam_robotwin_finetune.yaml
│   ├── attena_openpi_libero.yaml
│   └── attena_openpi_pi05_libero.yaml
├── docs/                           # Integration guides + paper PDF
│   ├── openvla_integration.md
│   ├── fastwam_integration.md
│   ├── openpi_integration.md
│   └── NeurIPS_2026_AttenA+_...pdf
├── third_party/                    # Base model submodules
│   ├── openvla-oft/                # branch: attena+oft
│   ├── FastWAM/                    # branch: attena+wam
│   └── openpi/                     # branch: attena+pi
├── setup.py
└── README.md
```

---

## Citation

If you find AttenA+ useful, please cite our paper:

```bibtex
@inproceedings{peng2026attenaplus,
  title     = {AttenA+: Velocity Field Action Attention for Enhancing Action-Centric Robotic Foundation Models},
  author    = {Peng, Daojie and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```

---

## Acknowledgements

AttenA+ builds on the following excellent open-source projects:

- [OpenVLA-OFT](https://github.com/openvla/openvla-oft) — Vision-Language-Action model fine-tuning
- [FastWAM](https://github.com/FastWAM/FastWAM) — Fast World Action Models
- [OpenPI](https://github.com/Physical-Intelligence/openpi) — π₀ / π₀.₅ models from Physical Intelligence

---

## License

This project is released under the [Apache 2.0 License](LICENSE).
