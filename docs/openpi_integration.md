# OpenPI (π₀ / π₀.₅) + AttenA+ Integration Guide

This guide explains how AttenA+ is integrated into the [OpenPI](../third_party/openpi) submodule (branch `attena+pi`), covering both **π₀** and **π₀.₅**.

---

## Architecture Overview

Both π₀ and π₀.₅ are implemented by the same `Pi0` class in `pi0.py`. The `pi05` flag in `Pi0Config` switches between the two variants:

| | π₀ | π₀.₅ |
|---|---|---|
| State input | Continuous projection token | Discrete language tokens |
| Timestep injection | MLP mixed with action tokens | adaRMSNorm (adaRMS) |
| `action_horizon` | 50 (default) | 10 (LIBERO) |
| `discrete_state_input` | False | False (LIBERO) |
| Loss type | Flow-matching MSE | Same |

Because `compute_loss` is **shared**, AttenA+ is implemented once and applies to both variants automatically.

---

## What Changed

```
src/openpi/models/pi0_config.py     ← 7 new AttenA+ config fields in Pi0Config
src/openpi/models/pi0.py            ← _velocity_weights() + weighted compute_loss()
src/openpi/training/config.py       ← attena_pi05_libero training config entry
```

No architectural changes; inference (`sample_actions`) is unaffected.

---

## Setup

```bash
cd third_party/openpi

# JAX + GPU
pip install "jax[cuda12]"
pip install -e ".[dev]"

# PyTorch training path (optional)
pip install -e ".[torch]"
```

See [docker.md](../third_party/openpi/docs/docker.md) for containerised setup.

---

## AttenA+ Config Fields

All new fields are added to `Pi0Config` with safe defaults (`use_velocity_attention=False` preserves backward compatibility):

| Field | Default | Description |
|---|---|---|
| `use_velocity_attention` | `False` | Enable AttenA+ |
| `velocity_weight_strategy` | `"inverse"` | `inverse` / `inverse_squared` / `exp_decay` / `log` |
| `velocity_clip_max_weight` | `2.0` | Weight ceiling; floor = 1/ceiling |
| `velocity_epsilon` | `1e-3` | Speed floor (prevents division by zero) |
| `velocity_alpha` | `2.0` | Decay rate (only for `exp_decay`) |
| `velocity_normalize_weights` | `True` | Rescale weights by `/clip * 2` → average ≈ 1 |
| `velocity_joint_dims` | `6` | Leading action dims used for speed (excludes gripper) |

---

## Training π₀ + AttenA+

Use the pre-registered `attena_pi05_libero` config, or build your own:

```bash
cd third_party/openpi

# π₀ on LIBERO with AttenA+
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
  pi0_libero \
  --exp-name attena_pi0_libero \
  --overrides \
    model.use_velocity_attention=true \
    model.velocity_weight_strategy=inverse \
    model.velocity_clip_max_weight=2.0
```

---

## Training π₀.₅ + AttenA+

### Option A — Use the pre-registered config (recommended)

```bash
cd third_party/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
  attena_pi05_libero \
  --exp-name attena_pi05_libero_run1
```

The `attena_pi05_libero` config in `config.py` is identical to `pi05_libero` with:

```python
Pi0Config(
    pi05=True,
    action_horizon=10,
    discrete_state_input=False,
    use_velocity_attention=True,
    velocity_weight_strategy="inverse",
    velocity_clip_max_weight=2.0,
    velocity_epsilon=1e-3,
    velocity_normalize_weights=True,
    velocity_joint_dims=6,
)
```

### Option B — Override an existing pi05 config

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
  pi05_libero \
  --exp-name attena_pi05_libero_run1 \
  --overrides \
    model.use_velocity_attention=true \
    model.velocity_weight_strategy=inverse \
    model.velocity_clip_max_weight=2.0
```

### PyTorch training path

```bash
python scripts/train_pytorch.py \
  --config attena_pi05_libero \
  --exp-name attena_pi05_libero_run1
```

---

## How It Works

**`compute_loss` (pi0.py)** — unchanged flow-matching logic, with AttenA+ applied to the per-timestep loss:

```python
# Per-timestep flow-matching MSE: (B, T)
per_step_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

# AttenA+: multiply by speed-derived weights
if self.use_velocity_attention:
    per_step_loss = per_step_loss * self._velocity_weights(actions)

return per_step_loss  # training loop reduces to scalar mean
```

**`_velocity_weights` (pi0.py)** — JAX re-implementation matching FastWAM/OpenVLA-OFT:

```python
def _velocity_weights(self, actions):
    # Speed from joint dims only (first 6, excluding gripper)
    joint_actions = actions[..., :self.velocity_joint_dims]        # (B, T, 6)
    speed = jnp.linalg.norm(joint_actions, axis=-1, keepdims=True) # (B, T, 1)
    speed = jnp.clip(speed, a_min=self.velocity_epsilon)

    weights = 1.0 / speed                                          # "inverse"
    weights = jnp.clip(weights, a_max=self.velocity_clip_max_weight)
    weights = jnp.clip(weights, a_min=1.0 / self.velocity_clip_max_weight)

    if self.velocity_normalize_weights:
        weights = weights / self.velocity_clip_max_weight * 2.0

    return weights[..., 0]   # (B, T)
```

---

## Notes

**π₀ vs π₀.₅ behavior**: the velocity-weighting logic is identical — only the model architecture differs. Both benefit equally from AttenA+.

**`velocity_joint_dims`**: defaults to 6 (first 6 dims = joint velocities, dim 7 = gripper). Adjust for robots with different action spaces (e.g., bimanual ALOHA with `action_dim=14` may use `velocity_joint_dims=12`).

**π₀-FAST**: uses tokenised actions with cross-entropy loss — velocity-based weighting is not applicable. AttenA+ is not applied to π₀-FAST.

**Inference**: `sample_actions` is unchanged; deploy and serve as usual.

```bash
python scripts/serve_policy.py \
  --env LIBERO \
  --checkpoint /path/to/attena_pi05_checkpoint

cd examples/libero && python main.py --host localhost --port 8000
```

See [remote_inference.md](../third_party/openpi/docs/remote_inference.md) for remote deployment.
