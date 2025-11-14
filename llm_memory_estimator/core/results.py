"""
Estimation result types and formatting.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EstimationResult:
    """
    Result of memory estimation with detailed breakdown.

    Attributes:
        Components (GiB):
            base_weights_gib: Base model weights
            lora_weights_gib: LoRA adapter weights (if applicable)
            trainable_state_gib: Gradients + optimizer states + master weights
            attention_activ_gib: Attention activation memory
            mlp_activ_gib: MLP activation memory
            logits_gib: Output logits buffer
            kv_cache_gib: KV cache (typically 0 in training)
            vision_activ_gib: Vision encoder activations (VLMs)

        Overheads (GiB):
            cuda_overhead_gib: CUDA context, cuBLAS
            misc_overhead_gib: Miscellaneous framework overhead
            deepspeed_overhead_gib: DeepSpeed runtime
            fragmentation_gib: Allocator fragmentation cushion

        Totals:
            steady_total_gib: Sum of all steady-state components
            peak_overhead_gib: Transient peak (loss computation, ZeRO-3 gather)
            peak_total_gib: steady + peak (max memory usage)

        Probe:
            measured_alloc_gib: Measured max allocated (if probe ran)
            measured_reserved_gib: Measured max reserved (if probe ran)

        breakdown: Detailed per-component breakdown with sub-items
    """
    # Component memory
    base_weights_gib: float
    lora_weights_gib: float
    trainable_state_gib: float
    attention_activ_gib: float
    mlp_activ_gib: float
    logits_gib: float
    kv_cache_gib: float
    vision_activ_gib: float

    # Overheads
    cuda_overhead_gib: float
    misc_overhead_gib: float
    deepspeed_overhead_gib: float
    fragmentation_gib: float

    # Totals
    steady_total_gib: float
    peak_overhead_gib: float
    peak_total_gib: float

    # Optional probe results
    measured_alloc_gib: Optional[float] = None
    measured_reserved_gib: Optional[float] = None

    # Detailed breakdown
    breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary_table(self) -> str:
        """
        Format results as a readable table.

        Returns:
            Multi-line string with formatted breakdown
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Memory Estimation Summary (per GPU)")
        lines.append("=" * 60)
        lines.append("")

        # Components
        lines.append("Components:")
        lines.append(f"  Base weights         : {self.base_weights_gib:8.2f} GiB")
        if self.lora_weights_gib > 0:
            lines.append(f"  LoRA weights         : {self.lora_weights_gib:8.2f} GiB")
        lines.append(f"  Trainable state      : {self.trainable_state_gib:8.2f} GiB")
        lines.append(f"  Attention activ.     : {self.attention_activ_gib:8.2f} GiB")
        lines.append(f"  MLP activ.           : {self.mlp_activ_gib:8.2f} GiB")
        lines.append(f"  Output logits        : {self.logits_gib:8.2f} GiB")
        if self.kv_cache_gib > 0:
            lines.append(f"  KV cache             : {self.kv_cache_gib:8.2f} GiB")
        if self.vision_activ_gib > 0:
            lines.append(f"  Vision activ.        : {self.vision_activ_gib:8.2f} GiB")

        lines.append("")
        lines.append("Overheads:")
        lines.append(f"  CUDA/cuBLAS          : {self.cuda_overhead_gib:8.2f} GiB")
        lines.append(f"  Miscellaneous        : {self.misc_overhead_gib:8.2f} GiB")
        lines.append(f"  DeepSpeed runtime    : {self.deepspeed_overhead_gib:8.2f} GiB")
        lines.append(f"  Fragmentation        : {self.fragmentation_gib:8.2f} GiB")

        lines.append("")
        lines.append("-" * 60)
        lines.append(f"  Total (steady)       : {self.steady_total_gib:8.2f} GiB")
        lines.append(f"  Peak overhead        : {self.peak_overhead_gib:8.2f} GiB")
        lines.append(f"  Total (peak)         : {self.peak_total_gib:8.2f} GiB")
        lines.append("=" * 60)

        # Probe results if available
        if self.measured_alloc_gib is not None:
            lines.append("")
            lines.append("Empirical Probe Results:")
            lines.append(f"  Max allocated        : {self.measured_alloc_gib:8.2f} GiB")
            if self.measured_reserved_gib is not None:
                lines.append(f"  Max reserved         : {self.measured_reserved_gib:8.2f} GiB")
                diff = self.peak_total_gib - self.measured_reserved_gib
                lines.append(f"  Diff (est - meas)    : {diff:+8.2f} GiB")
            lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "components": {
                "base_weights_gib": self.base_weights_gib,
                "lora_weights_gib": self.lora_weights_gib,
                "trainable_state_gib": self.trainable_state_gib,
                "attention_activ_gib": self.attention_activ_gib,
                "mlp_activ_gib": self.mlp_activ_gib,
                "logits_gib": self.logits_gib,
                "kv_cache_gib": self.kv_cache_gib,
                "vision_activ_gib": self.vision_activ_gib,
            },
            "overheads": {
                "cuda_overhead_gib": self.cuda_overhead_gib,
                "misc_overhead_gib": self.misc_overhead_gib,
                "deepspeed_overhead_gib": self.deepspeed_overhead_gib,
                "fragmentation_gib": self.fragmentation_gib,
            },
            "totals": {
                "steady_total_gib": self.steady_total_gib,
                "peak_overhead_gib": self.peak_overhead_gib,
                "peak_total_gib": self.peak_total_gib,
            },
            "probe": {
                "measured_alloc_gib": self.measured_alloc_gib,
                "measured_reserved_gib": self.measured_reserved_gib,
            } if self.measured_alloc_gib is not None else None,
            "breakdown": self.breakdown,
        }
