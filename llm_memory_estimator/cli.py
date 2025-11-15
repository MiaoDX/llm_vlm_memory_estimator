#!/usr/bin/env python3
"""
LLM/VLM GPU Memory Estimator - CLI Interface

This module provides a command-line interface to the modular memory estimation
architecture. It handles argument parsing and maps CLI flags to the structured
configuration format used by the core estimator.

Features:
- Estimates per-GPU VRAM requirements for LLM/VLM fine-tuning
- Supports LoRA/QLoRA, FlashAttention, DeepSpeed ZeRO, multi-GPU parallelism
- Optional empirical probe to validate estimates against actual GPU usage
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

# Import modular architecture components
from .core.estimator import MemoryEstimator
from .core.config import EstimatorConfig
from .core.results import EstimationResult

#Constants
GiB = 1024 ** 3

def to_gib(x_bytes: float) -> float:
    """Convert bytes to GiB"""
    return float(x_bytes) / GiB


def parse_bool(s: Optional[str]) -> bool:
    """Parse boolean from string (for argparse)"""
    if s is None:
        return False
    return str(s).lower() in {"1", "y", "yes", "t", "true", "on"}


def pretty_print(config: EstimatorConfig, result: EstimationResult):
    """Print estimation results in a readable format"""
    print("\n=== Memory Estimate (per GPU) ===")
    print(f"Model: {config.model}")
    print(f"dtype={config.dtype}, seq_len={config.seq_len}, micro_batch={config.per_device_batch}, GA={config.grad_accum}")

    lora_enabled = config.lora is not None
    qlora_enabled = lora_enabled and config.lora.get("quantized", False)
    print(f"DP={config.dp}, TP={config.tp}, PP={config.pp}, ZeRO={config.zero_stage}, LoRA={lora_enabled}, QLoRA={qlora_enabled}")
    print("")

    # Extract breakdown from result
    breakdown = result.breakdown

    rows = [
        ("Base weights", breakdown.get("weights", {}).get("gib", 0.0)),
        ("Trainable state", breakdown.get("trainable_state", {}).get("gib", 0.0)),
        ("Attention activ.", breakdown.get("attention_activations", {}).get("gib", 0.0)),
        ("MLP activ.", breakdown.get("mlp_activations", {}).get("gib", 0.0)),
        ("Output logits", breakdown.get("logits", {}).get("gib", 0.0)),
        ("KV cache", breakdown.get("kv_cache", {}).get("gib", 0.0)),
        ("Vision activ.", breakdown.get("vision_activations", {}).get("gib", 0.0)),
        ("Misc. overhead", breakdown.get("misc_overhead", {}).get("gib", 0.0)),
        ("DeepSpeed overhead", breakdown.get("deepspeed_overhead", {}).get("gib", 0.0)),
        ("CUDA/cuBLAS", breakdown.get("cuda_libs", {}).get("gib", 0.0)),
        ("Fragmentation", breakdown.get("fragmentation", {}).get("gib", 0.0)),
        ("Total (steady)", result.steady_total_gib),
        ("Peak overhead", result.peak_overhead_gib),
        ("Total (peak)", result.peak_total_gib),
    ]

    colw = max(len(n) for n, _ in rows) + 2
    for name, val in rows:
        print(f"{name:<{colw}} : {val:6.2f} GiB")

    # Print empirical probe results if available
    if hasattr(result, 'measured_alloc_gib') and result.measured_alloc_gib is not None:
        print("\n--- Empirical Probe ---")
        print(f"max_memory_allocated : {result.measured_alloc_gib:.2f} GiB")
        if hasattr(result, 'measured_reserved_gib') and result.measured_reserved_gib is not None:
            print(f"max_memory_reserved  : {result.measured_reserved_gib:.2f} GiB")
            diff = result.peak_total_gib - result.measured_reserved_gib
            print(f"Diff (estimate - reserved): {diff:+.2f} GiB")


def main():
    """Main CLI entry point"""
    p = argparse.ArgumentParser(description="LLM/VLM GPU Memory Estimator + Probe")

    # Model & topology
    p.add_argument("--model", type=str, required=True, help="HF model id or path (config.json available)")
    p.add_argument("--is-seq2seq", type=parse_bool, default=False, help="Seq2seq model (else causal-lm)")

    # Training shape
    p.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    p.add_argument("--per-device-batch", type=int, default=1, help="Micro-batch size per rank")
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")

    # Parallelism
    p.add_argument("--dp", type=int, default=1, help="Data parallel replicas")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel degree")
    p.add_argument("--pp", type=int, default=1, help="Pipeline stages")

    # Precision & optimizer
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["fp32", "float32", "bf16", "bfloat16", "fp16", "float16", "int8", "fp8", "int4", "nf4"],
                   help="Weights/activations dtype")
    p.add_argument("--optimizer", type=str, default="adamw", help="Optimizer type (for reference only)")
    p.add_argument("--optimizer-bits", type=int, default=32, choices=[8, 32], help="Optimizer precision (for reference only)")
    p.add_argument("--master-weights", type=parse_bool, default=True, help="Keep FP32 master copy (for reference only)")

    # PEFT / LoRA
    p.add_argument("--lora", type=parse_bool, default=False, dest="use_lora", help="Enable LoRA")
    p.add_argument("--qlora", type=parse_bool, default=False, help="Enable QLoRA (4-bit base + LoRA)")
    p.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--lora-target-modules", nargs="*", default=None, help="LoRA target modules")
    p.add_argument("--lora-params-override", type=int, default=None, help="Override LoRA param count")

    # Memory-saving knobs
    p.add_argument("--flashattn", type=parse_bool, default=False, help="Enable FlashAttention")
    p.add_argument("--liger", type=parse_bool, default=False, help="Enable Liger kernels")
    p.add_argument("--fused-loss", type=parse_bool, default=False, help="Enable fused loss")
    p.add_argument("--grad-checkpoint", type=parse_bool, default=False, help="Enable gradient checkpointing")
    p.add_argument("--kv-cache", type=parse_bool, default=False, dest="use_kv_cache", help="Use KV cache")

    # ZeRO & offload
    p.add_argument("--zero", type=int, default=0, choices=[0, 1, 2, 3], help="DeepSpeed ZeRO stage")
    p.add_argument("--dp-offload-optimizer", type=parse_bool, default=False, dest="cpu_offload_optimizer",
                   help="CPU offload optimizer (reserved)")

    # Vision branch (VLM)
    p.add_argument("--use-vision", type=parse_bool, default=False, help="Enable vision encoder")
    p.add_argument("--vision-hidden", type=int, default=1024, help="Vision hidden size")
    p.add_argument("--vision-layers", type=int, default=24, help="Vision encoder layers")
    p.add_argument("--vision-image-size", type=int, default=448, help="Vision image size")
    p.add_argument("--vision-patch", type=int, default=14, help="Vision patch size")

    # Overheads & calibration
    p.add_argument("--cuda-libs-gib", type=float, default=2.5, help="CUDA context overhead (GiB)")
    p.add_argument("--misc-overhead-gib", type=float, default=3.0, help="Miscellaneous overhead (GiB)")
    p.add_argument("--deepspeed-overhead-gib", type=float, default=None, help="DeepSpeed overhead (GiB, auto if None)")
    p.add_argument("--fragmentation-gib", type=float, default=5.0, help="Memory fragmentation cushion (GiB)")
    p.add_argument("--attn-act-factor-no-fa", type=float, default=4.0, help="Attention activation factor (no FA)")
    p.add_argument("--attn-act-factor-fa", type=float, default=0.8, help="Attention activation factor (with FA)")
    p.add_argument("--mlp-act-factor", type=float, default=6.0, help="MLP activation factor")
    p.add_argument("--logits-factor", type=float, default=1.0, help="Logits memory factor")
    p.add_argument("--grad-ckpt-reduction", type=float, default=0.35, help="Gradient checkpoint reduction factor")

    # Empirical probe
    p.add_argument("--probe", type=parse_bool, default=False, help="Run empirical probe on GPU")
    p.add_argument("--probe-steps", type=int, default=6, help="Training steps for probe")
    p.add_argument("--probe-warmup", type=int, default=3, help="Warmup steps before measurement")
    p.add_argument("--probe-device", type=str, default="cuda", help="Device for probe (cuda, cuda:0, etc)")
    p.add_argument("--probe-device-id", type=int, default=0, help="GPU device ID for probe")
    p.add_argument("--estimate-only", type=parse_bool, default=False, help="Skip probe even if --probe is set")

    args = p.parse_args()

    # Build optimizations dict
    optimizations = {}
    if args.flashattn:
        optimizations["flashattention"] = {}
    if args.grad_checkpoint:
        optimizations["gradient_checkpointing"] = {
            "reduction": args.grad_ckpt_reduction
        }
    if args.fused_loss:
        optimizations["fused_loss"] = {}

    # Build LoRA config
    lora = None
    if args.use_lora or args.qlora:
        lora = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "quantized": args.qlora,
            "target_modules": args.lora_target_modules if args.lora_target_modules else None,
        }
        if args.lora_params_override is not None:
            lora["params_override"] = args.lora_params_override

    # Build vision config
    vision = None
    if args.use_vision:
        vision = {
            "enabled": True,
            "image_size": args.vision_image_size,
            "patch_size": args.vision_patch,
            "layers": args.vision_layers,
            "hidden": args.vision_hidden,
        }

    # Calculate DeepSpeed overhead if not specified
    deepspeed_overhead_gib = args.deepspeed_overhead_gib
    if deepspeed_overhead_gib is None:
        if args.zero == 0:
            deepspeed_overhead_gib = 0.0
        elif args.zero == 1:
            deepspeed_overhead_gib = 1.5
        elif args.zero == 2:
            deepspeed_overhead_gib = 2.0
        else:  # ZeRO-3
            deepspeed_overhead_gib = 3.0

    # Create configuration
    config = EstimatorConfig(
        # Model & topology
        model=args.model,
        is_seq2seq=args.is_seq2seq,

        # Training shape
        seq_len=args.seq_len,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,

        # Parallelism
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        zero_stage=args.zero,

        # Precision
        dtype=args.dtype,

        # PEFT
        lora=lora,

        # Memory optimizations
        optimizations=optimizations,

        # Vision
        vision=vision,

        # KV cache
        use_kv_cache=args.use_kv_cache,

        # Overheads & calibration
        cuda_libs_gib=args.cuda_libs_gib,
        misc_overhead_gib=args.misc_overhead_gib,
        deepspeed_overhead_gib=deepspeed_overhead_gib,
        fragmentation_gib=args.fragmentation_gib,
        attn_act_factor_no_fa=args.attn_act_factor_no_fa,
        attn_act_factor_fa=args.attn_act_factor_fa,
        mlp_act_factor=args.mlp_act_factor,
        logits_factor=args.logits_factor,

        # Probe settings
        probe_enabled=args.probe,
        probe_steps=args.probe_steps,
        probe_warmup=args.probe_warmup,
        probe_device=args.probe_device,
        probe_device_id=args.probe_device_id,
    )

    # Create estimator and run estimation
    estimator = MemoryEstimator(config)
    result = estimator.estimate()

    # Print estimation results
    pretty_print(config, result)

    # Run empirical probe if requested
    if args.probe and not args.estimate_only:
        try:
            from .probe import ProbeRunner

            print("\n" + "=" * 70)
            print("Running empirical GPU probe...")
            print("=" * 70)

            # Get model info from estimator
            model_info = estimator.model_info

            # Create and run probe
            runner = ProbeRunner(config, model_info)
            probe_result = runner.run()

            # Print probe summary
            probe_result.print_summary()

            # Compare with estimation
            comparison = probe_result.compare_with_estimate(result.peak_total_gib)
            comparison.print_comparison()

        except ImportError as e:
            print(f"\n[ERROR] Failed to import probe modules: {e}")
            print("[INFO] Install GPU probe dependencies with:")
            print("       uv sync --extra cuda-full")
            print("       OR: pip install peft bitsandbytes flash-attn deepspeed liger-kernel")
            sys.exit(1)
        except ValueError as e:
            print(f"\n[ERROR] Configuration error: {e}")
            sys.exit(1)
        except RuntimeError as e:
            print(f"\n[ERROR] Runtime error during probe: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Unexpected error during probe: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
