"""
AttenA+: Velocity Field Action Attention

Canonical implementation of the velocity-weighted action loss described in:
  "AttenA+: Velocity Field Action Attention for Enhancing Action-Centric
   Robotic Foundation Models" (NeurIPS 2026)

Key idea: assign higher learning weights to low-velocity action timesteps,
aligning training with the physical hierarchy of robotic manipulation
(slow = precision-demanding, fast = transitional).

Verified and validated against three model implementations:
  - third_party/openvla-oft/prismatic/training/train_utils.py
      Discriminative VLA (AttenA+OFT); tested on LIBERO and real Franka robot.
  - third_party/FastWAM/src/fastwam/losses/action_loss.py
      World-Action Model (AttenA+WAM); tested on RoboTwin 2.0.
  - third_party/openpi/src/openpi/models/pi0.py
      Generative flow-matching VLA (AttenA+π₀ / AttenA+π₀.₅);
      tested on LIBERO (π₀) and validated in training (π₀.₅).

Usage:
    from attena import VelocityAttention

    # Paper default: inverse_squared, clip_max=2.0
    attena = VelocityAttention(weight_strategy="inverse_squared", clip_max_weight=2.0)

    # 1. Discriminative VLA (AttenA+OFT / OpenVLA-OFT):
    #    ground_truth == loss target, returns scalar
    loss = attena.weighted_loss(ground_truth, predicted, loss_type="l1")

    # 2. Flow-matching generative VLA (AttenA+π₀ / AttenA+π₀.₅ / OpenPI):
    #    weights from ground_truth, loss between predicted and flow target
    loss = attena.weighted_loss(
        ground_truth, predicted,
        target=u_t,            # flow-matching target: noise - actions
        loss_type="mse",
    )

    # 3. World-Action Model (AttenA+WAM / FastWAM):
    #    separate flow target + padding mask + per-sample reduction
    loss_per_sample = attena.weighted_loss(
        ground_truth, predicted,
        target=flow_target,    # noise - actions
        loss_type="l1",
        action_is_pad=pad_mask,
        reduction="none",      # returns (B,) for flexible aggregation
    )

    # One-shot static API (no instantiation):
    loss = VelocityAttention.compute_weighted_loss(ground_truth, predicted)
"""

import torch
from torch import Tensor
from typing import Literal, Optional


WeightStrategy = Literal["inverse", "inverse_squared", "exp_decay", "log"]
LossType = Literal["l1", "mse"]


class VelocityAttention:
    """Velocity Field Action Attention (AttenA+).

    Computes per-timestep attention weights from the speed magnitude of a
    ground-truth action chunk, then returns a weighted action loss.

    Args:
        weight_strategy: How to map speed → weight.
            "inverse"          w = 1 / s
            "inverse_squared"  w = 1 / s²  (paper default — strongest slow/fast contrast)
            "exp_decay"        w = exp(-alpha * s), alpha=5.0
            "log"              w = 1 / log(1 + s)
        clip_max_weight: Upper bound for weights; also sets lower bound as
            1 / clip_max_weight. Paper experiments use 2.0 and 5.0.
        epsilon: Small constant to avoid division by zero.
        alpha: Decay rate for the "exp_decay" strategy (paper: α=5.0).
        normalize_weights: If True, rescale weights by
            ``weights / clip_max_weight * 2.0`` to keep average weight ≈ 1.
            Matches FastWAM default (True) and OpenVLA-OFT default (False).
        joint_dims: Slice of action dimensions used for speed computation.
            Defaults to the first 6 dimensions (joint velocities, excluding
            the gripper channel). D=7 for LIBERO (6 joints + 1 gripper).
    """

    def __init__(
        self,
        weight_strategy: WeightStrategy = "inverse_squared",
        clip_max_weight: float = 2.0,
        epsilon: float = 1e-3,
        alpha: float = 5.0,
        normalize_weights: bool = True,
        joint_dims: slice = slice(0, 6),
    ) -> None:
        self.weight_strategy = weight_strategy
        self.clip_max_weight = clip_max_weight
        self.epsilon = epsilon
        self.alpha = alpha
        self.normalize_weights = normalize_weights
        self.joint_dims = joint_dims

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_weights(self, ground_truth_actions: Tensor) -> Tensor:
        """Compute per-timestep attention weights from ground-truth actions.

        Args:
            ground_truth_actions: Shape (B, T, D).

        Returns:
            weights: Shape (B, T, 1), broadcastable over the D dimension.
        """
        # Use joint dimensions only (exclude gripper) for speed calculation.
        joint_gt = ground_truth_actions[..., self.joint_dims]  # (B, T, joint_d)
        speed = torch.norm(joint_gt, dim=-1, keepdim=True)     # (B, T, 1)
        speed = torch.clamp(speed, min=self.epsilon)

        weights = self._speed_to_weight(speed)                 # (B, T, 1)

        # Clip to [1/clip_max_weight, clip_max_weight].
        weights = torch.clamp(weights, max=self.clip_max_weight)
        weights = torch.clamp(weights, min=1.0 / self.clip_max_weight)

        # Optional normalization: rescale so average weight ≈ 1.
        # Formula matches both FastWAM and OpenVLA-OFT implementations.
        if self.normalize_weights:
            weights = weights / self.clip_max_weight * 2.0

        return weights  # (B, T, 1)

    def weighted_loss(
        self,
        ground_truth: Tensor,
        predicted: Tensor,
        target: Optional[Tensor] = None,
        loss_type: LossType = "l1",
        action_is_pad: Optional[Tensor] = None,
        reduction: Literal["mean", "none"] = "mean",
    ) -> Tensor:
        """Compute velocity-weighted action loss.

        Weights are always derived from ``ground_truth`` (original normalized
        actions). The actual loss is computed between ``predicted`` and
        ``target``; if ``target`` is None, ``ground_truth`` is used as the
        loss target (OpenVLA-OFT behaviour). Pass a separate ``target`` for
        flow-matching models like FastWAM where the training target is the
        noise-minus-action velocity field rather than the original action.

        Args:
            ground_truth:   (B, T, D) original ground-truth actions.
                            Used *only* for speed/weight computation.
            predicted:      (B, T, D) model predictions.
            target:         (B, T, D) loss target. Defaults to ``ground_truth``
                            when None (OpenVLA-OFT / continuous regression).
                            Pass ``noise - actions`` for FastWAM.
            loss_type:      "l1" (MAE) or "mse".
            action_is_pad:  Optional (B, T) bool mask; True = padding timestep
                            (excluded from loss). FastWAM usage.
            reduction:      "mean" → scalar; "none" → (B,) per-sample.

        Returns:
            Scalar loss (or per-sample losses if reduction="none").
        """
        weights = self.compute_weights(ground_truth)   # (B, T, 1)
        loss_target = ground_truth if target is None else target

        if loss_type == "l1":
            errors = torch.abs(predicted - loss_target)    # (B, T, D)
        elif loss_type == "mse":
            errors = (predicted - loss_target) ** 2        # (B, T, D)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type!r}")

        weighted_errors = errors * weights                 # (B, T, D)
        loss_per_timestep = weighted_errors.mean(dim=-1)   # (B, T)

        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=loss_per_timestep.device, dtype=loss_per_timestep.dtype
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)    # (B,)
            loss_per_sample = (loss_per_timestep * valid).sum(dim=1) / valid_sum
        else:
            loss_per_sample = loss_per_timestep.mean(dim=1)  # (B,)

        if reduction == "mean":
            return loss_per_sample.mean()
        return loss_per_sample  # (B,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _speed_to_weight(self, speed: Tensor) -> Tensor:
        """Map speed magnitudes to attention weights (monotonically decreasing)."""
        s = self.weight_strategy
        if s == "inverse":
            return 1.0 / speed
        elif s == "inverse_squared":
            return 1.0 / (speed ** 2)
        elif s == "exp_decay":
            return torch.exp(-self.alpha * speed)
        elif s == "log":
            return 1.0 / torch.log1p(speed)
        else:
            raise ValueError(f"Unknown weight_strategy: {s!r}")

    # ------------------------------------------------------------------
    # Convenience: pure-function API (no class instantiation required)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_weighted_loss(
        ground_truth: Tensor,
        predicted: Tensor,
        target: Optional[Tensor] = None,
        weight_strategy: WeightStrategy = "inverse_squared",
        clip_max_weight: float = 2.0,
        epsilon: float = 1e-3,
        alpha: float = 5.0,
        normalize_weights: bool = True,
        loss_type: LossType = "l1",
        action_is_pad: Optional[Tensor] = None,
        joint_dims: slice = slice(0, 6),
    ) -> Tensor:
        """One-shot velocity-weighted loss (no class instantiation needed).

        Useful when integrating AttenA+ into an existing training loop with
        a single function call. See ``weighted_loss`` for parameter docs.

        Returns:
            Scalar loss.
        """
        va = VelocityAttention(
            weight_strategy=weight_strategy,
            clip_max_weight=clip_max_weight,
            epsilon=epsilon,
            alpha=alpha,
            normalize_weights=normalize_weights,
            joint_dims=joint_dims,
        )
        return va.weighted_loss(
            ground_truth=ground_truth,
            predicted=predicted,
            target=target,
            loss_type=loss_type,
            action_is_pad=action_is_pad,
        )
