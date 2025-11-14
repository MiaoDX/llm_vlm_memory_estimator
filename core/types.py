"""
Core type definitions used across multiple modules.

This module contains only cross-cutting types to avoid circular imports.
Module-specific protocols live in their respective modules:
- Optimization: optimizations/base.py
- MemoryComponent: components/base.py
"""

from enum import Enum
from typing import Protocol, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..components.base import ComponentMemory


class MemoryImpactArea(Enum):
    """
    Areas where optimizations can affect memory usage.
    Used to route optimization effects to the correct component calculators.
    """
    WEIGHTS = "weights"
    GRADIENTS = "gradients"
    OPTIMIZER_STATE = "optimizer_state"
    ACTIVATIONS_ATTENTION = "activations_attention"
    ACTIVATIONS_MLP = "activations_mlp"
    LOGITS = "logits"
    KV_CACHE = "kv_cache"
    VISION_ACTIVATIONS = "vision_activations"
    CUDA_OVERHEAD = "cuda_overhead"
    FRAGMENTATION = "fragmentation"
    MISC_OVERHEAD = "misc_overhead"
    DEEPSPEED_OVERHEAD = "deepspeed_overhead"


class ParallelismStrategy(Protocol):
    """
    Protocol for parallelism strategies (DP/TP/PP).

    Implementations handle how memory is scaled based on parallelism settings.
    Each strategy knows how to distribute memory across ranks.
    """

    def scale_components(
        self,
        components: List['ComponentMemory']
    ) -> List['ComponentMemory']:
        """
        Scale component memory based on parallelism settings.

        Args:
            components: List of calculated memory components

        Returns:
            List of scaled components (per-rank memory)
        """
        ...

    def get_peak_overhead(self) -> float:
        """
        Calculate additional peak memory overhead from this parallelism strategy.

        For example, ZeRO-3 has all-gather peaks when materializing full parameters.

        Returns:
            Additional peak memory in bytes
        """
        ...
