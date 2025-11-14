"""
Parallelism strategies for scaling memory across distributed training setups.

This module provides strategies for:
- Data Parallelism (DP)
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- DeepSpeed ZeRO (stages 1/2/3)
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import EstimatorConfig
    from ..components.base import ComponentMemory
    from ..core.types import ParallelismStrategy


def create_parallelism_strategy(config: 'EstimatorConfig') -> 'ParallelismStrategy':
    """
    Create the appropriate parallelism strategy based on configuration.

    Args:
        config: Estimator configuration

    Returns:
        ParallelismStrategy instance
    """
    if config.zero_stage > 0:
        return ZeROStrategy(config)
    elif config.tp > 1 or config.pp > 1:
        return HybridParallelismStrategy(config)
    else:
        return DataParallelismStrategy(config)


class DataParallelismStrategy:
    """
    Pure data parallelism strategy.

    In DP, each GPU holds a full model copy.
    Memory per GPU = full model memory (no reduction).
    """

    def __init__(self, config: 'EstimatorConfig'):
        self.config = config

    def scale_components(
        self,
        components: List['ComponentMemory']
    ) -> List['ComponentMemory']:
        """
        Scale components for data parallelism.

        In DP, memory per GPU is unchanged (full replica).
        """
        # No scaling needed - each GPU has full model
        return components

    def get_peak_overhead(self) -> float:
        """No additional peak overhead for pure DP"""
        return 0.0


class HybridParallelismStrategy:
    """
    Hybrid parallelism (TP + PP) strategy.

    Tensor Parallelism: Split weights/activations across TP dimension
    Pipeline Parallelism: Split layers across PP stages
    """

    def __init__(self, config: 'EstimatorConfig'):
        self.config = config

    def scale_components(
        self,
        components: List['ComponentMemory']
    ) -> List['ComponentMemory']:
        """
        Scale components for tensor/pipeline parallelism.

        TP divides weights and activations by tp degree.
        PP divides layers by pp stages.
        """
        from ..core.types import MemoryImpactArea
        from copy import copy

        scaled = []
        tp = self.config.tp
        pp = self.config.pp

        for comp in components:
            scaled_comp = copy(comp)
            scaled_comp.breakdown = comp.breakdown.copy()

            # Weights are split by both TP and PP
            if comp.area == MemoryImpactArea.WEIGHTS:
                scaled_comp.bytes = comp.bytes / (tp * pp)
                scaled_comp.breakdown["tp_scale"] = scaled_comp.bytes

            # Optimizer states and gradients are split by TP and PP
            elif comp.area in (
                MemoryImpactArea.OPTIMIZER_STATE,
                MemoryImpactArea.GRADIENTS,
            ):
                scaled_comp.bytes = comp.bytes / (tp * pp)
                scaled_comp.breakdown["tp_pp_scale"] = scaled_comp.bytes

            # Activations are split by TP (but not PP - each stage has its own activations)
            elif comp.area in (
                MemoryImpactArea.ACTIVATIONS_ATTENTION,
                MemoryImpactArea.ACTIVATIONS_MLP,
                MemoryImpactArea.VISION_ACTIVATIONS,
            ):
                # TP splits activations, PP divides layers
                scaled_comp.bytes = comp.bytes / (tp * pp)
                scaled_comp.breakdown["tp_pp_scale"] = scaled_comp.bytes

            # Logits are split by TP
            elif comp.area == MemoryImpactArea.LOGITS:
                scaled_comp.bytes = comp.bytes / tp
                scaled_comp.breakdown["tp_scale"] = scaled_comp.bytes

            # KV cache is split by TP and PP
            elif comp.area == MemoryImpactArea.KV_CACHE:
                scaled_comp.bytes = comp.bytes / (tp * pp)
                scaled_comp.breakdown["tp_pp_scale"] = scaled_comp.bytes

            # Overheads are not split
            else:
                # No scaling for overheads
                pass

            scaled.append(scaled_comp)

        return scaled

    def get_peak_overhead(self) -> float:
        """
        TP/PP have minimal peak overhead (mostly communication buffers).
        """
        # Small overhead for TP all-reduce/all-gather buffers
        # Approximate: 100 MB per GPU
        return 100 * 1024 * 1024


class ZeROStrategy:
    """
    DeepSpeed ZeRO optimization strategy.

    Stage 1: Shard optimizer states across DP ranks
    Stage 2: Shard optimizer states + gradients
    Stage 3: Shard optimizer states + gradients + weights
    """

    def __init__(self, config: 'EstimatorConfig'):
        self.config = config
        self.stage = config.zero_stage
        self.dp = config.dp

    def scale_components(
        self,
        components: List['ComponentMemory']
    ) -> List['ComponentMemory']:
        """
        Scale components based on ZeRO stage.
        """
        from ..core.types import MemoryImpactArea
        from copy import copy

        if self.dp <= 1:
            # No sharding with single GPU
            return components

        scaled = []
        for comp in components:
            scaled_comp = copy(comp)
            scaled_comp.breakdown = comp.breakdown.copy()

            # Stage 1+: Shard optimizer states
            if self.stage >= 1 and comp.area == MemoryImpactArea.OPTIMIZER_STATE:
                scaled_comp.bytes = comp.bytes / self.dp
                scaled_comp.breakdown[f"zero_stage{self.stage}_shard"] = scaled_comp.bytes

            # Stage 2+: Shard gradients
            elif self.stage >= 2 and comp.area == MemoryImpactArea.GRADIENTS:
                scaled_comp.bytes = comp.bytes / self.dp
                scaled_comp.breakdown[f"zero_stage{self.stage}_shard"] = scaled_comp.bytes

            # Stage 3: Shard weights
            elif self.stage >= 3 and comp.area == MemoryImpactArea.WEIGHTS:
                scaled_comp.bytes = comp.bytes / self.dp
                scaled_comp.breakdown[f"zero_stage{self.stage}_shard"] = scaled_comp.bytes

            scaled.append(scaled_comp)

        return scaled

    def get_peak_overhead(self) -> float:
        """
        ZeRO-3 has peak overhead for all-gather of full parameters during forward/backward.

        During forward/backward, each layer needs to all-gather its full parameters temporarily.
        """
        if self.stage >= 3:
            # Peak overhead is approximately 1 layer's worth of parameters
            # This is a rough estimate - actual overhead depends on layer size
            # Assume ~5% of model size as peak overhead for parameter all-gather
            # This will be refined based on actual model info in aggregator
            GiB = 1024 ** 3
            return 2.0 * GiB  # Conservative estimate
        else:
            # Stages 1/2 have minimal peak overhead
            return 0.0
