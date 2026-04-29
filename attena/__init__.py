"""
AttenA+: Velocity Field Action Attention

Public API
----------
    from attena import VelocityAttention

    attena = VelocityAttention(weight_strategy="inverse", clip_max_weight=2.0)
    loss   = attena.weighted_loss(ground_truth, predicted, loss_type="l1")

Or as a one-shot static function:

    from attena import VelocityAttention
    loss = VelocityAttention.compute_weighted_loss(ground_truth, predicted)
"""

from attena.velocity_attention import VelocityAttention

__all__ = ["VelocityAttention"]
__version__ = "1.0.0"
