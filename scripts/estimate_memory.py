#!/usr/bin/env python3
"""
Example script demonstrating the modular memory estimation API.

This script shows how to use the new component-based architecture
to estimate LLM training memory requirements.
"""

from llm_vlm_memory_estimator.core.config import EstimatorConfig
from llm_vlm_memory_estimator.core.estimator import MemoryEstimator


def main():
    """Example: Estimate memory for Qwen2.5-7B with LoRA"""

    # Configure estimation
    config = EstimatorConfig(
        # Model
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="bf16",

        # Training shape
        seq_len=4096,
        per_device_batch=2,
        grad_accum=4,

        # Parallelism
        dp=8,  # 8-way data parallel
        zero_stage=2,  # DeepSpeed ZeRO-2

        # LoRA configuration
        lora={
            "rank": 16,
            "alpha": 32,
            "quantized": False,  # QLoRA if True
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },

        # Optimizations
        optimizations={
            "flashattention": {},  # Enable FlashAttention
            "gradient_checkpointing": {"reduction": 0.35},  # 35% activation retention
        },
    )

    # Create estimator
    print(f"Estimating memory for {config.model}...")
    estimator = MemoryEstimator(config)

    print(f"Model: {estimator.model_info.param_count:,} parameters")
    print(f"Hidden size: {estimator.model_info.hidden_size}")
    print(f"Layers: {estimator.model_info.num_layers}")
    print()

    # Run estimation
    result = estimator.estimate()

    # Display results
    print(result.summary_table())

    # Access detailed breakdown
    print("\nDetailed Component Breakdown:")
    for area, details in result.breakdown.items():
        print(f"  {area:25s}: {details['gib']:8.2f} GiB")


if __name__ == "__main__":
    main()
