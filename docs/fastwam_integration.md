# FastWAM + AttenA+ Integration Guide

This guide explains how AttenA+ is integrated into the [FastWAM](../third_party/FastWAM) submodule (branch `attena+wam`).

---

## What Changed

AttenA+ replaces FastWAM's standard action loss with a **velocity-weighted action loss** in `src/fastwam/losses/action_loss.py`. The Hydra config system exposes all AttenA+ parameters directly in YAML task configs.

```
src/fastwam/losses/action_loss.py   ← compute_weighted_action_loss()
src/fastwam/trainer.py              ← loss routing
configs/task/                       ← AttenA+ parameters in YAML
```

---

## Setup

```bash
cd third_party/FastWAM
pip install -e .
```

FastWAM requires PyTorch ≥ 2.7.1 and DeepSpeed for multi-GPU training.

```bash
pip install deepspeed accelerate
```

See [third_party/FastWAM/README.md](../third_party/FastWAM/README.md) for the full environment setup.

---

## Training with AttenA+

AttenA+ hyperparameters are controlled via Hydra YAML configs. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `use_weighted_action_loss` | `true` | Enable AttenA+ |
| `weight_strategy` | `"inverse"` | `inverse` / `inverse_squared` / `exp_decay` / `log` |
| `clip_max_weight` | `2.0` | Maximum weight clipping |
| `epsilon` | `1e-3` | Speed floor |
| `alpha` | `2.0` | Decay rate (for `exp_decay`) |
| `normalize_weights` | `true` | Rescale weights |

### LIBERO Training Example

```bash
cd third_party/FastWAM

# Pre-compute text embeddings first
python scripts/precompute_text_embeds.py \
  data=libero_2cam224 \
  data.data_path=/path/to/libero

# Launch training with DeepSpeed ZeRO-2
bash scripts/train_zero2.sh \
  task=libero_uncond_2cam224_1e-4 \
  use_weighted_action_loss=true \
  weight_strategy=inverse_squared \
  clip_max_weight=2.0
```

### RoboTwin Training Example

```bash
bash scripts/train_zero2.sh \
  task=robotwin_uncond_3cam224_1e-4 \
  use_weighted_action_loss=true \
  weight_strategy=inverse_squared \
  clip_max_weight=2.0
```

### RoboTwin Fine-tuning (AttenA+WAM, Paper Protocol)

The AttenA+WAM paper results are obtained by freezing the vision encoders and WAM backbone and **fine-tuning only the action head** with velocity attention. Training runs for ~1 epoch on RoboTwin 2.0 using 2× H800 GPUs (~4 days).

The reference config is at [`configs/attena_fastwam_robotwin_finetune.yaml`](../configs/attena_fastwam_robotwin_finetune.yaml), which mirrors [`third_party/FastWAM/configs/task/robotwin_finetune_action_head.yaml`](../third_party/FastWAM/configs/task/robotwin_finetune_action_head.yaml).

```bash
cd third_party/FastWAM

# Requires a pre-trained FastWAM checkpoint
bash scripts/train_zero2.sh \
  task=robotwin_finetune_action_head \
  resume=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

Key fine-tuning parameters (from the config):

| Parameter | Value | Notes |
|---|---|---|
| `freeze_strategy` | `action_head_only` | Freeze vision + WAM backbone |
| `learning_rate` | `5e-6` | Smaller LR for fine-tuning |
| `batch_size` | `8` | With `gradient_accumulation_steps=2` |
| `num_epochs` | `2` | Save checkpoints every 1,000 steps |
| `weight_strategy` | `inverse_squared` | Paper default |
| `clip_max_weight` | `2.0` | Paper experiments use 2.0 and 5.0 |

---

## How It Works

**Loss computation** (`src/fastwam/losses/action_loss.py`):

```python
def compute_weighted_action_loss(
    ground_truth_actions,   # (B, T, D)
    predicted_actions,      # (B, T, D)
    weight_strategy="inverse",
    clip_max_weight=2.0,
    epsilon=1e-3,
    alpha=2.0,
    normalize_weights=True,
    action_is_pad=None,     # (B, T) optional padding mask
):
    # 1. Speed from joint dims (first 6, excluding gripper)
    joint_gt = ground_truth_actions[..., :6]
    speed = torch.norm(joint_gt, dim=-1, keepdim=True).clamp(min=epsilon)

    # 2. Weight strategy
    weights = 1.0 / speed                           # "inverse"

    # 3. Clip and optionally normalize
    weights = weights.clamp(max=clip_max_weight)
    weights = weights.clamp(min=1.0 / clip_max_weight)
    if normalize_weights:
        weights = weights / clip_max_weight * 2.0

    # 4. Weighted L1 errors, respecting padding
    errors = torch.abs(predicted_actions - ground_truth_actions)
    weighted_errors = errors * weights              # (B, T, D)
    loss_per_timestep = weighted_errors.mean(dim=2) # (B, T)

    if action_is_pad is not None:
        valid = (~action_is_pad).float()
        loss_per_sample = (loss_per_timestep * valid).sum(1) / valid.sum(1).clamp(1)
    else:
        loss_per_sample = loss_per_timestep.mean(1)

    return loss_per_sample  # (B,) — Trainer reduces to scalar
```

---

## Fine-tuning a Pretrained FastWAM

### LIBERO Fine-tuning (AttenA+WAM)

The reference config is at [`configs/attena_fastwam_libero_finetune.yaml`](../configs/attena_fastwam_libero_finetune.yaml), which mirrors [`third_party/FastWAM/configs/task/libero_finetune_action_head.yaml`](../third_party/FastWAM/configs/task/libero_finetune_action_head.yaml).

```bash
cd third_party/FastWAM

bash scripts/train_zero2.sh \
  task=libero_finetune_action_head \
  resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

| Parameter | Value | Notes |
|---|---|---|
| `freeze_strategy` | `action_head_only` | Freeze vision + WAM backbone |
| `learning_rate` | `1e-5` | Smaller LR for fine-tuning |
| `batch_size` | `30` | |
| `num_epochs` | `3` | Save checkpoints every 500 steps |
| `weight_strategy` | `inverse_squared` | Paper default |
| `clip_max_weight` | `2.0` | Paper experiments use 2.0 and 5.0 |

### RoboTwin 2.0 Fine-tuning (AttenA+WAM, Paper Protocol)

The AttenA+WAM paper results are obtained by freezing the vision encoders and WAM backbone and **fine-tuning only the action head** with velocity attention. Training runs for ~1 epoch on RoboTwin 2.0 using 2× H800 GPUs (~4 days).

The reference config is at [`configs/attena_fastwam_robotwin_finetune.yaml`](../configs/attena_fastwam_robotwin_finetune.yaml), which mirrors [`third_party/FastWAM/configs/task/robotwin_finetune_action_head.yaml`](../third_party/FastWAM/configs/task/robotwin_finetune_action_head.yaml).

```bash
cd third_party/FastWAM

bash scripts/train_zero2.sh \
  task=robotwin_finetune_action_head \
  resume=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

| Parameter | Value | Notes |
|---|---|---|
| `freeze_strategy` | `action_head_only` | Freeze vision + WAM backbone |
| `learning_rate` | `5e-6` | Smaller LR for RoboTwin fine-tuning |
| `batch_size` | `8` | With `gradient_accumulation_steps=2` |
| `num_epochs` | `2` | Save checkpoints every 1,000 steps |
| `weight_strategy` | `inverse_squared` | Paper default |
| `clip_max_weight` | `2.0` | Paper experiments use 2.0 and 5.0 |

See [FINETUNING_EN.md](../third_party/FastWAM/FINETUNING_EN.md) for additional fine-tuning strategies.

---

## Evaluation

```bash
cd third_party/FastWAM

# LIBERO
python experiments/libero/run_libero_manager.py \
  checkpoint=/path/to/checkpoint \
  task_suite=libero_spatial \
  num_trials=50

# RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  checkpoint=/path/to/checkpoint
```

See [EVALUATION_EN.md](../third_party/FastWAM/EVALUATION_EN.md) for the full evaluation guide.
