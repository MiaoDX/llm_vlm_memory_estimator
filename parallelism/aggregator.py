"""
Memory aggregator for combining component results into final estimation.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..components.base import ComponentMemory
    from ..core.config import EstimatorConfig
    from ..core.model_info import ModelInfo
    from ..core.results import EstimationResult


class MemoryAggregator:
    """
    Aggregates memory component results into a final estimation.

    Combines scaled component memories and produces:
    - Steady-state memory (normal training)
    - Peak memory (with transient spikes)
    """

    def aggregate(
        self,
        components: List['ComponentMemory'],
        config: 'EstimatorConfig',
        model_info: 'ModelInfo'
    ) -> 'EstimationResult':
        """
        Aggregate component memories into final result.

        Args:
            components: List of calculated and scaled memory components
            config: Estimator configuration
            model_info: Model architecture information

        Returns:
            EstimationResult with per-GPU memory breakdown
        """
        from ..core.results import EstimationResult
        from ..core.types import MemoryImpactArea

        # Categorize components by area
        component_map = {comp.area: comp for comp in components}

        # Steady-state memory = sum of all components
        steady_bytes = sum(comp.bytes for comp in components)

        # Peak memory = steady + transient spikes
        # Transient spikes occur during:
        # - Gradient all-reduce (DP)
        # - Parameter all-gather (ZeRO-3)
        # - Activation recomputation (gradient checkpointing)

        # For simplicity, use a heuristic based on largest component
        # More sophisticated: track specific transient allocations
        peak_overhead_bytes = self._estimate_peak_overhead(components, config, model_info)

        peak_bytes = steady_bytes + peak_overhead_bytes

        # Extract individual component values for breakdown
        def get_bytes(area: MemoryImpactArea) -> float:
            comp = component_map.get(area)
            return comp.bytes if comp else 0.0

        # Separate base weights and LoRA weights if applicable
        weights_comp = component_map.get(MemoryImpactArea.WEIGHTS)
        base_weights_gib = 0.0
        lora_weights_gib = 0.0
        if weights_comp and "lora_weights" in weights_comp.breakdown:
            base_weights_gib = self._to_gib(weights_comp.breakdown.get("base_weights", 0.0))
            lora_weights_gib = self._to_gib(weights_comp.breakdown.get("lora_weights", 0.0))
        elif weights_comp:
            base_weights_gib = self._to_gib(weights_comp.bytes)

        # Combine optimizer state and gradients into trainable_state
        trainable_state_gib = self._to_gib(
            get_bytes(MemoryImpactArea.OPTIMIZER_STATE) +
            get_bytes(MemoryImpactArea.GRADIENTS)
        )

        # Build result matching existing EstimationResult structure
        result = EstimationResult(
            # Per-component breakdown
            base_weights_gib=base_weights_gib,
            lora_weights_gib=lora_weights_gib,
            trainable_state_gib=trainable_state_gib,
            attention_activ_gib=self._to_gib(get_bytes(MemoryImpactArea.ACTIVATIONS_ATTENTION)),
            mlp_activ_gib=self._to_gib(get_bytes(MemoryImpactArea.ACTIVATIONS_MLP)),
            logits_gib=self._to_gib(get_bytes(MemoryImpactArea.LOGITS)),
            kv_cache_gib=self._to_gib(get_bytes(MemoryImpactArea.KV_CACHE)),
            vision_activ_gib=self._to_gib(get_bytes(MemoryImpactArea.VISION_ACTIVATIONS)),

            # Overheads
            cuda_overhead_gib=self._to_gib(get_bytes(MemoryImpactArea.CUDA_OVERHEAD)),
            misc_overhead_gib=self._to_gib(get_bytes(MemoryImpactArea.MISC_OVERHEAD)),
            fragmentation_gib=self._to_gib(get_bytes(MemoryImpactArea.FRAGMENTATION)),
            deepspeed_overhead_gib=self._to_gib(get_bytes(MemoryImpactArea.DEEPSPEED_OVERHEAD)),

            # Totals
            steady_total_gib=self._to_gib(steady_bytes),
            peak_overhead_gib=self._to_gib(peak_overhead_bytes),
            peak_total_gib=self._to_gib(peak_bytes),

            # Component details for advanced users
            breakdown={
                comp.area.value: {
                    "bytes": comp.bytes,
                    "gib": comp.to_gib(),
                    "breakdown": comp.breakdown
                }
                for comp in components
            }
        )

        return result

    def _estimate_peak_overhead(
        self,
        components: List['ComponentMemory'],
        config: 'EstimatorConfig',
        model_info: 'ModelInfo'
    ) -> float:
        """
        Estimate transient peak memory overhead.

        Args:
            components: Component memories
            config: Configuration
            model_info: Model info

        Returns:
            Peak overhead in bytes
        """
        # Base overhead from parallelism strategy
        # (This is already calculated by the strategy)

        # Additional transient spikes:
        # 1. Gradient all-reduce buffers (DP)
        # 2. Activation recomputation (grad checkpoint)
        # 3. Temporary buffers for optimizers

        # Simple heuristic: ~10% of steady-state for transient spikes
        steady_total = sum(comp.bytes for comp in components)
        return steady_total * 0.1

    @staticmethod
    def _to_gib(bytes_val: float) -> float:
        """Convert bytes to GiB"""
        GiB = 1024 ** 3
        return float(bytes_val) / GiB
