"""
Built-in memory optimizations.

This module contains standard optimizations that come with the estimator:
- FlashAttention
- Gradient Checkpointing
- Fused Loss (Liger kernel)
"""

from typing import List, Optional
from ..base import Optimization, OptimizationEffect, EffectType
from ...core.types import MemoryImpactArea


class FlashAttentionOptimization:
    """
    FlashAttention optimization reduces attention memory by not materializing
    the full attention matrix.

    Memory reduction: ~80% for attention activations
    """

    @property
    def name(self) -> str:
        return "flashattention"

    def get_effects(self, config) -> List[OptimizationEffect]:
        """FlashAttention significantly reduces attention activation memory"""
        return [
            OptimizationEffect(
                area=MemoryImpactArea.ACTIVATIONS_ATTENTION,
                effect_type=EffectType.MULTIPLY,
                value=0.2,  # Reduces attention memory to ~20% of baseline
                description="flashattention_reduction"
            )
        ]

    def conflicts_with(self) -> List[str]:
        """FlashAttention doesn't conflict with other optimizations"""
        return []

    def get_probe_modifier(self):
        """No probe modifier needed for now"""
        return None

    def get_documentation(self) -> str:
        return """
FlashAttention: Memory-efficient attention mechanism

Reduces attention activation memory by ~80% by avoiding materialization of
the full attention score matrix. Uses tiled computation and recomputation.

Requirements:
  - flash-attn package installed
  - GPU with compute capability >= 7.5 (Volta+)

Parameters: None
"""


class GradientCheckpointingOptimization:
    """
    Gradient checkpointing trades compute for memory by recomputing activations
    during backward pass instead of storing them.

    Configurable reduction factor.
    """

    def __init__(self, reduction: float = 0.35):
        """
        Args:
            reduction: Fraction of activations retained (0 < reduction <= 1)
                      Default 0.35 means keep 35% of activations
        """
        if not 0 < reduction <= 1:
            raise ValueError(f"reduction must be in (0, 1], got {reduction}")
        self.reduction = reduction

    @property
    def name(self) -> str:
        return "gradient_checkpointing"

    def get_effects(self, config) -> List[OptimizationEffect]:
        """Reduce activation memory by checkpointing"""
        return [
            OptimizationEffect(
                area=MemoryImpactArea.ACTIVATIONS_ATTENTION,
                effect_type=EffectType.MULTIPLY,
                value=self.reduction,
                description=f"grad_checkpoint_{self.reduction}"
            ),
            OptimizationEffect(
                area=MemoryImpactArea.ACTIVATIONS_MLP,
                effect_type=EffectType.MULTIPLY,
                value=self.reduction,
                description=f"grad_checkpoint_{self.reduction}"
            )
        ]

    def conflicts_with(self) -> List[str]:
        """Gradient checkpointing is compatible with all optimizations"""
        return []

    def get_probe_modifier(self):
        """No probe modifier needed for now"""
        return None

    def get_documentation(self) -> str:
        return f"""
Gradient Checkpointing: Trade compute for memory

Recomputes activations during backward pass instead of storing them all.
Current setting retains {self.reduction * 100:.1f}% of activations.

Benefits:
  - Reduces activation memory significantly
  - Compatible with all other optimizations

Tradeoffs:
  - Increases training time by ~20-30%
  - More backward compute required

Parameters:
  - reduction: Fraction of activations to keep (default: 0.35)
"""


class FusedLossOptimization:
    """
    Fused loss computation (e.g., Liger kernel) computes cross-entropy loss
    without materializing the full logits tensor.

    Reduces logits memory significantly.
    """

    @property
    def name(self) -> str:
        return "fused_loss"

    def get_effects(self, config) -> List[OptimizationEffect]:
        """Fused loss reduces logits memory"""
        return [
            OptimizationEffect(
                area=MemoryImpactArea.LOGITS,
                effect_type=EffectType.MULTIPLY,
                value=0.2,  # Reduces logits memory to ~20%
                description="fused_loss_reduction"
            )
        ]

    def conflicts_with(self) -> List[str]:
        """Compatible with all optimizations"""
        return []

    def get_probe_modifier(self):
        """No probe modifier needed for now"""
        return None

    def get_documentation(self) -> str:
        return """
Fused Loss: Memory-efficient cross-entropy

Computes cross-entropy loss in-kernel without materializing full logits tensor.
Reduces logits memory by ~80%.

Requirements:
  - liger-kernel package (or similar fused loss implementation)

Benefits:
  - Significant memory savings for large vocabulary models
  - Slight speedup from kernel fusion

Parameters: None
"""


# Register all built-in optimizations
from ..registry import OPTIMIZATION_REGISTRY

OPTIMIZATION_REGISTRY.register(FlashAttentionOptimization)
OPTIMIZATION_REGISTRY.register(GradientCheckpointingOptimization)
OPTIMIZATION_REGISTRY.register(FusedLossOptimization)
