# OpenVLA-OFT + AttenA+ Integration Guide

This guide explains how AttenA+ is integrated into the [OpenVLA-OFT](../third_party/openvla-oft) submodule (branch `attena+oft`).

---

## What Changed

AttenA+ adds a **velocity-weighted L1 loss** to OpenVLA-OFT's action head training. The standard unweighted L1 loss is replaced by `compute_weighted_l1_loss()` in `prismatic/training/train_utils.py`.

```
prismatic/training/train_utils.py   ← velocity-weighted loss function
vla-scripts/finetune.py             ← loss switch + new hyperparameters
```

No model architecture changes are required.

---

## Setup

```bash
cd third_party/openvla-oft

# Create environment (Python 3.10 + PyTorch 2.2)
conda env create -f environment.yml
conda activate openvla

# Install flash-attention (required for training)
pip install flash-attn --no-build-isolation

pip install -e .
```

See [SETUP.md](../third_party/openvla-oft/SETUP.md) for detailed hardware requirements.

---

## Training with AttenA+

AttenA+ is enabled by default on the `attena+oft` branch. The key flags in `FinetuneConfig`:

| Parameter | Default | Description |
|---|---|---|
| `use_weighted_l1_loss` | `True` | Enable AttenA+ velocity weighting |
| `weight_strategy` | `"inverse"` | Weight function: `inverse` / `inverse_squared` / `exp_decay` / `log` |
| `clip_max_weight` | `10.0` | Maximum weight value (lower bound = 1/clip_max_weight) |
| `epsilon` | `1e-3` | Speed floor to avoid division by zero |
| `alpha` | `2.0` | Decay rate (only for `exp_decay`) |
| `normalize_weights` | `False` | Rescale weights to [1/clip, 1] |

### LIBERO Fine-tuning Example

```bash
cd third_party/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /path/to/libero/datasets \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/checkpoints \
  --use_weighted_l1_loss True \
  --weight_strategy inverse \
  --clip_max_weight 10.0 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --use_lora True \
  --lora_rank 32
```

To disable AttenA+ and use the standard L1 loss:

```bash
  --use_weighted_l1_loss False
```

---

## How It Works

**Loss computation** (`prismatic/training/train_utils.py`):

```python
def compute_weighted_l1_loss(
    ground_truth_actions,   # (B, T, 7)
    predicted_actions,      # (B, T, 7)
    weight_strategy="inverse",
    clip_max_weight=10.0,
    epsilon=1e-3,
    alpha=2.0,
    normalize_weights=False,
):
    # 1. Compute speed from joint dims only (dim 0:6, excluding gripper)
    joint_gt = ground_truth_actions[..., :6]
    speed = torch.norm(joint_gt, dim=-1, keepdim=True).clamp(min=epsilon)

    # 2. Map speed → weight (monotonically decreasing)
    weights = 1.0 / speed                        # "inverse" strategy

    # 3. Clip to prevent extremes
    weights = weights.clamp(max=clip_max_weight)
    weights = weights.clamp(min=1.0 / clip_max_weight)

    # 4. Apply to all 7 action dimensions
    l1_errors = torch.abs(ground_truth_actions - predicted_actions)
    return (l1_errors * weights).mean()
```

---

## Evaluation on LIBERO

```bash
cd third_party/openvla-oft

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /path/to/checkpoint \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --use_l1_regression True
```

See [LIBERO.md](../third_party/openvla-oft/LIBERO.md) for full evaluation instructions.
