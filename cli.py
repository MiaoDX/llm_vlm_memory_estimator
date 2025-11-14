#!/usr/bin/env python3
"""
LLM/VLM GPU Memory Estimator + Empirical Probe

Features
- Parse Hugging Face model config (by name or local config.json)
- Estimate per-GPU steady and peak VRAM with a transparent breakdown
- Account for: dtype (fp32/bf16/fp16/int8/int4), LoRA/QLoRA, optimizer states (AdamW, 8-bit),
  gradient memory, activations (attention + MLP), FlashAttention, Liger/fused loss, KV cache, vision branch,
  ZeRO sharding (0/1/2/3), Data Parallel / Tensor Parallel / Pipeline Parallel
- Add fixed overheads (CUDA context, misc), fragmentation cushion
- Optional empirical probe (dry-run) with synthetic data to measure max memory

Notes
- This is a practical estimator with tunable coefficients. Real VRAM usage depends on kernels and framework versions.
- Use the activation/logits factors and overhead flags to calibrate the model against probe runs on your hardware.
- Empirical probe currently targets HF CausalLM/Seq2Seq. VLM probing is stubbed (you can extend it for your vision encoder).

"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple

# Optional deps are imported lazily in probe() to avoid hard requirements for estimation.

GiB = 1024 ** 3

DTYPE_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "fp8": 1.0,      # many FP8 formats are 1 byte per element
    "int4": 0.5,      # 4-bit
    # QLoRA NF4 quantized weights; include scales/zeros overhead (~12%)
    "nf4": 0.56,
}


def to_gib(x_bytes: float) -> float:
    return float(x_bytes) / GiB


@dataclass
class EstimatorInputs:
    # Model & topology
    model: str                       # HF model id or local path (config.json)
    is_seq2seq: bool = False         # else causal-lm
    vocab_size: Optional[int] = None # override if needed

    # Training shape
    seq_len: int = 2048
    per_device_batch: int = 1        # micro-batch size per rank
    grad_accum: int = 1              # gradient accumulation steps

    # Parallelism
    dp: int = 1                      # data parallel replicas
    tp: int = 1                      # tensor parallel degree
    pp: int = 1                      # pipeline stages

    # Precision & optimizer
    dtype: str = "bf16"              # weights/activations dtype
    optimizer: str = "adamw"         # adamw | sgd | adafactor
    optimizer_bits: int = 32         # 32 | 8 (for 8-bit optimizers)
    master_weights: bool = True      # keep fp32 master copy for stability (full FT)

    # PEFT / LoRA
    use_lora: bool = False
    qlora: bool = False              # 4-bit base weights + LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: Optional[List[str]] = None
    lora_params_override: Optional[int] = None # if provided, skip auto-estimation; count in parameters

    # Memory-saving knobs
    flashattn: bool = False
    # Note: liger flag is currently used for config mirroring and probe warnings only;
    # it does not change the estimator math yet.
    liger: bool = False
    fused_loss: bool = False
    grad_checkpoint: bool = False
    use_kv_cache: bool = False       # typically False in training

    # ZeRO & offload (simplified rules)
    zero: int = 0                    # 0/1/2/3
    # cpu_offload_optimizer is reserved for future modeling of optimizer offload;
    # at the moment it is tracked for completeness but does not change on-GPU VRAM estimates.
    cpu_offload_optimizer: bool = False

    # Vision branch (VLM) — rough knobs
    use_vision: bool = False
    vision_hidden: int = 1024
    vision_layers: int = 24
    vision_image_size: int = 448
    vision_patch: int = 14           # ViT patch size

    # Overheads & cushions (GiB)
    cuda_libs_gib: float = 2.5
    misc_overhead_gib: float = 3.0
    deepspeed_overhead_gib: float = 1.0
    fragmentation_gib: float = 5.0

    # Activation multipliers (tunable)
    # These factors approximate how many *hidden_size elements* per token per layer are saved in memory.
    attn_act_factor_no_fa: float = 4.0
    attn_act_factor_fa: float = 0.8
    mlp_act_factor: float = 6.0
    logits_factor: float = 1.0       # ~B*L*vocab elements. If fused loss, set lower (e.g., 0.2)
    grad_ckpt_reduction: float = 0.35 # proportion of activations retained when checkpointing (0<r<=1)

    # Empirical probe
    probe: bool = False
    probe_steps: int = 6
    warmup_steps: int = 3
    estimate_only: bool = False      # if true, skip probe even if probe=True


@dataclass
class EstimatorOutputs:
    base_weights_gib: float
    trainable_state_gib: float
    attention_activ_gib: float
    mlp_activ_gib: float
    logits_gib: float
    kv_cache_gib: float
    vision_activ_gib: float
    misc_overhead_gib: float
    deepspeed_overhead_gib: float
    cuda_libs_gib: float
    fragmentation_gib: float
    steady_total_gib: float
    peak_overhead_gib: float
    peak_total_gib: float

    # optional empirical probe results
    measured_alloc_gib: Optional[float] = None
    measured_reserved_gib: Optional[float] = None


class MemoryEstimator:
    def __init__(self, cfg: EstimatorInputs):
        self.cfg = cfg
        self._hf_config = None
        self._param_count_total = None
        self._hidden_size = None
        self._num_layers = None
        self._num_heads = None
        self._vocab_size = None

    # ---- HF config helpers ----
    def _load_hf_config(self):
        if self._hf_config is not None:
            return
        try:
            from transformers import AutoConfig
        except Exception:
            raise RuntimeError("transformers is required to parse model config; pip install transformers")
        self._hf_config = AutoConfig.from_pretrained(self.cfg.model)
        # Guess some common fields
        self._hidden_size = getattr(self._hf_config, "hidden_size", None) or getattr(self._hf_config, "n_embd", None)
        self._num_layers = getattr(self._hf_config, "num_hidden_layers", None) or getattr(self._hf_config, "n_layer", None)
        self._num_heads = getattr(self._hf_config, "num_attention_heads", None)
        self._vocab_size = self.cfg.vocab_size or getattr(self._hf_config, "vocab_size", None)
        if self._hidden_size is None or self._num_layers is None:
            print("[warn] Could not infer hidden_size/num_layers from config; activation estimates may be off.")

    def _count_params_meta(self) -> int:
        if self._param_count_total is not None:
            return self._param_count_total
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
            from accelerate import init_empty_weights
            import torch
        except Exception:
            raise RuntimeError("transformers and accelerate are required for param counting; pip install transformers accelerate")
        cfg = AutoConfig.from_pretrained(self.cfg.model)
        with init_empty_weights():
            if self.cfg.is_seq2seq:
                model = AutoModelForSeq2SeqLM.from_config(cfg)
            else:
                model = AutoModelForCausalLM.from_config(cfg)
        total = 0
        for p in model.parameters():
            total += p.numel()
        self._param_count_total = int(total)
        return self._param_count_total

    # ---- LoRA param estimation (rough) ----
    def _estimate_lora_params(self) -> int:
        if not self.cfg.use_lora:
            return 0
        if self.cfg.lora_params_override is not None:
            return int(self.cfg.lora_params_override)
        # Heuristic: estimate LoRA params from HF config and target module names.
        # For a linear layer with shape (out, in), LoRA adds A (out, r) and B (r, in):
        # params_per_module ≈ r * (in + out).
        self._load_hf_config()
        H = self._hidden_size or 4096
        L = self._num_layers or 32
        # Try to infer MLP intermediate size; fallback to 4*H
        intermediate_size = getattr(self._hf_config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = 4 * H

        # If no target list, assume common modern LLaMA/Mistral modules
        targets = self.cfg.lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]

        params = 0
        for name in targets:
            # Approximate (in_dim, out_dim) by module role
            if name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                in_dim = H
                out_dim = H
            elif name in {"up_proj", "gate_proj"}:
                in_dim = H
                out_dim = intermediate_size
            elif name in {"down_proj"}:
                in_dim = intermediate_size
                out_dim = H
            else:
                # Fallback: treat as HxH
                in_dim = H
                out_dim = H
            params_per_layer = self.cfg.lora_rank * (in_dim + out_dim)
            params += params_per_layer * L
        return int(params)

    # ---- Core estimation ----
    def estimate(self) -> EstimatorOutputs:
        self._load_hf_config()
        param_count = self._count_params_meta()
        if self._hidden_size is None or self._num_layers is None:
            raise RuntimeError("Could not infer hidden_size/num_layers from HF config; please use a supported model or override config.")
        H = self._hidden_size
        L = self._num_layers
        num_heads = self._num_heads or max(1, H // 128)
        V = self._vocab_size or 50272

        cfg = self.cfg
        dtype_bytes_w = DTYPE_BYTES.get(cfg.dtype.lower(), 2.0)
        dtype_bytes_act = DTYPE_BYTES.get(cfg.dtype.lower(), 2.0)

        # Parallel scaling
        tp = max(1, cfg.tp)
        pp = max(1, cfg.pp)
        dp = max(1, cfg.dp)

        # Approximate per-pipeline-stage share of layers/params
        params_per_stage = param_count / pp

        # ---- Base weights (may be quantized) ----
        weight_storage_bytes = params_per_stage * dtype_bytes_w
        if cfg.qlora:
            # 4-bit with scales/zeros; allow user to override later.
            weight_storage_bytes = params_per_stage * DTYPE_BYTES["nf4"]
        # ZeRO-3 partitions params across DP. TP partitions layer params across shards too.
        # Approx per-rank parameter residency.
        if cfg.zero >= 3 and not cfg.qlora:
            weight_storage_bytes /= dp
        # TP shards parameter matrices across tensor-parallel ranks for full FT.
        # For QLoRA we treat base weights as TP-sharded but ZeRO-replicated.
        weight_storage_bytes /= tp
        base_weights_gib = to_gib(weight_storage_bytes)

        # ---- Trainable states ----
        total_lora_params = self._estimate_lora_params() if cfg.use_lora else 0
        lora_params = total_lora_params / pp if cfg.use_lora else 0
        full_train_params = 0 if cfg.use_lora else params_per_stage
        train_params = lora_params if cfg.use_lora else full_train_params

        # Gradients (same dtype as weights typically)
        grad_bytes = train_params * dtype_bytes_w
        # Optimizer states
        if cfg.optimizer.lower() == "adamw":
            if cfg.optimizer_bits == 8:
                # very rough: ~0.5 bytes/param for moments in 8-bit land (+ small fp32 stats); simplify to 1B
                opt_bytes_per_param = 1.0
            else:
                opt_bytes_per_param = 8.0  # 2 moments in fp32
            master_bytes = 4.0 if cfg.master_weights else 0.0
        else:
            # SGD/Adafactor rougher
            opt_bytes_per_param = 4.0
            master_bytes = 0.0
        opt_bytes = train_params * opt_bytes_per_param
        master_w_bytes = train_params * master_bytes

        # ZeRO partition rules (simplified):
        # stage1: optimizer / dp; stage2: optimizer+grads / dp; stage3: optimizer+grads+params / dp
        if cfg.zero >= 1:
            opt_bytes /= dp
        if cfg.zero >= 2:
            grad_bytes /= dp
        if cfg.zero >= 3:
            # when stage-3, per-rank also only keeps shards of *trainable params* (for full FT). For LoRA, small effect.
            if not cfg.use_lora:
                master_w_bytes /= dp
        # Tensor parallel also splits trainable tensors across shards (roughly)
        grad_bytes /= tp
        opt_bytes /= tp
        master_w_bytes /= tp

        trainable_state_gib = to_gib(grad_bytes + opt_bytes + master_w_bytes)

        # ---- Activations ----
        B = cfg.per_device_batch
        Lseq = cfg.seq_len
        # approximate layers per pipeline stage
        layers_per_stage = max(1, math.ceil(L / pp))

        # attention activations
        if cfg.flashattn:
            # FA roughly linear in sequence length; factor absorbs details
            attn_act_elems = B * Lseq * layers_per_stage * cfg.attn_act_factor_fa * H
        else:
            # Non-FA: include explicit O(S^2) score/prob term plus linear-in-S term.
            # Score/prob matrices: B * heads * S * S * layers_per_stage
            score_elems = float(B) * float(num_heads) * float(Lseq) * float(Lseq) * float(layers_per_stage)
            # Additional per-token Q/K/V and context tensors approximated via attn_act_factor_no_fa
            dense_elems = B * Lseq * layers_per_stage * cfg.attn_act_factor_no_fa * H
            attn_act_elems = score_elems + dense_elems

        mlp_act_elems = B * Lseq * layers_per_stage * cfg.mlp_act_factor * H
        # gradient checkpointing shrinks what we keep resident
        gc_scale = cfg.grad_ckpt_reduction if cfg.grad_checkpoint else 1.0
        # Attention scores are often stored in fp32 for stability even with bf16 activations.
        if cfg.flashattn:
            attn_act_bytes = attn_act_elems * dtype_bytes_act * gc_scale
        else:
            score_bytes = score_elems * max(dtype_bytes_act, DTYPE_BYTES["fp32"]) * gc_scale
            dense_bytes = dense_elems * dtype_bytes_act * gc_scale
            attn_act_bytes = score_bytes + dense_bytes
        mlp_act_bytes = mlp_act_elems * dtype_bytes_act * gc_scale
        # TP: per-shard activations in many compute paths; approximate 1/tp scaling
        if tp > 1:
            attn_act_bytes /= tp
            mlp_act_bytes /= tp
        attention_activ_gib = to_gib(attn_act_bytes)
        mlp_activ_gib = to_gib(mlp_act_bytes)

        # ---- Output logits ----
        # Baseline: need B*L*V elements; with fused-loss (Liger) this can drop a lot.
        logits_elems = B * Lseq * V * cfg.logits_factor * (0.2 if cfg.fused_loss else 1.0)
        # TP: vocab-parallel final projection often shards logits across tp ranks
        if tp > 1:
            logits_elems /= tp
        logits_bytes = logits_elems * dtype_bytes_act
        logits_gib = to_gib(logits_bytes)

        # ---- KV cache (typically off for training) ----
        kv_cache_gib = 0.0
        if cfg.use_kv_cache:
            # keys + values per head per token per layer; rough: 2 * B * L * layers * H * bytes
            kv_bytes = 2.0 * B * Lseq * layers_per_stage * H * dtype_bytes_act
            if tp > 1:
                kv_bytes /= tp
            kv_cache_gib = to_gib(kv_bytes)

        # ---- Vision activations (very rough ViT-style) ----
        vision_activ_gib = 0.0
        if cfg.use_vision:
            num_patches = (cfg.vision_image_size // cfg.vision_patch) ** 2
            vision_elems = B * num_patches * cfg.vision_layers * cfg.vision_hidden
            # Vision encoder is typically replicated (not TP-sharded) in many VLMs (e.g., LLaVA/BLIP-style).
            # We assume same dtype and apply ckpt reduction if enabled.
            vision_bytes = vision_elems * dtype_bytes_act * gc_scale
            vision_activ_gib = to_gib(vision_bytes)

        # ---- Overheads ----
        misc_overhead_gib = cfg.misc_overhead_gib
        cuda_libs_gib = cfg.cuda_libs_gib
        fragmentation_gib = cfg.fragmentation_gib
        ds_overhead_gib = cfg.deepspeed_overhead_gib
        if cfg.zero == 0:
            ds_overhead_gib = max(ds_overhead_gib, 1.0)
        elif cfg.zero == 1:
            ds_overhead_gib = max(ds_overhead_gib, 1.5)
        elif cfg.zero == 2:
            ds_overhead_gib = max(ds_overhead_gib, 2.0)
        elif cfg.zero >= 3:
            ds_overhead_gib = max(ds_overhead_gib, 3.0)

        # ---- Totals ----
        steady_total = (
            base_weights_gib + trainable_state_gib + attention_activ_gib + mlp_activ_gib + logits_gib +
            kv_cache_gib + vision_activ_gib + misc_overhead_gib + ds_overhead_gib + cuda_libs_gib + fragmentation_gib
        )
        # Extra peak for transient buffers (loss/softmax/optimizer steps, etc.)
        peak_overhead_gib = 0.1 * max(1.0, B)

        # ZeRO-3 all-gather peaks: approximate incremental peak from materializing full params
        zero3_gather_gib = 0.0
        if cfg.zero >= 3 and dp > 1 and not cfg.qlora and train_params > 0:
            # Full trainable param bytes per rank when gathered (still TP-sharded, per pipeline stage)
            full_param_bytes = (train_params * dtype_bytes_w) / tp
            shard_bytes = full_param_bytes / dp
            zero3_gather_gib = to_gib(max(0.0, full_param_bytes - shard_bytes))
            peak_overhead_gib += zero3_gather_gib

        peak_total = steady_total + peak_overhead_gib

        return EstimatorOutputs(
            base_weights_gib=base_weights_gib,
            trainable_state_gib=trainable_state_gib,
            attention_activ_gib=attention_activ_gib,
            mlp_activ_gib=mlp_activ_gib,
            logits_gib=logits_gib,
            kv_cache_gib=kv_cache_gib,
            vision_activ_gib=vision_activ_gib,
            misc_overhead_gib=misc_overhead_gib,
            deepspeed_overhead_gib=ds_overhead_gib,
            cuda_libs_gib=cuda_libs_gib,
            fragmentation_gib=fragmentation_gib,
            steady_total_gib=steady_total,
            peak_overhead_gib=peak_overhead_gib,
            peak_total_gib=peak_total,
        )

    # ---- Empirical probe ----
    def probe(self, outputs: EstimatorOutputs) -> Tuple[Optional[float], Optional[float]]:
        if self.cfg.estimate_only:
            return None, None
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
            )
        except Exception as e:
            print(f"[warn] probe skipped (missing deps): {e}")
            return None, None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("[warn] probe skipped: CUDA device not available")
            return None, None

        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map.get(self.cfg.dtype.lower(), torch.bfloat16)

        # Build model
        cfg = AutoConfig.from_pretrained(self.cfg.model)
        if self.cfg.is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_config(cfg)
        else:
            model = AutoModelForCausalLM.from_config(cfg)

        # Precision & GC
        if self.cfg.grad_checkpoint:
            model.gradient_checkpointing_enable()
        model.to(device=device, dtype=torch_dtype)
        model.train()

        # LoRA attachment (optional)
        if self.cfg.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                target_modules = self.cfg.lora_target_modules or [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "gate_proj",
                    "down_proj",
                ]
                peft_cfg = LoraConfig(
                    r=self.cfg.lora_rank,
                    lora_alpha=self.cfg.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM" if not self.cfg.is_seq2seq else "SEQ_2_SEQ_LM",
                )
                model = get_peft_model(model, peft_cfg)
                model.to(device=device, dtype=torch_dtype)
            except Exception as e:
                print(f"[warn] LoRA not attached (missing peft or incompatible model): {e}")

        # FlashAttention
        if self.cfg.flashattn:
            # Many modern HF models auto-select FlashAttn2 if available via --flash-attn flag at build time or env vars.
            # We can't reliably toggle here. We leave it to the installed model kernels.
            pass

        # Optimizer (simple AdamW)
        try:
            from torch.optim import AdamW
            opt = AdamW(model.parameters(), lr=1e-4)
        except Exception as e:
            print(f"[warn] could not create optimizer: {e}")
            opt = None

        # Synthetic data
        B = self.cfg.per_device_batch
        Lseq = self.cfg.seq_len
        vocab = self._vocab_size or getattr(cfg, "vocab_size", 50272)
        input_ids = torch.randint(0, vocab, (B, Lseq), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        labels = input_ids.clone()

        # Warn if estimate assumes fused loss or FlashAttention the probe cannot enforce
        if self.cfg.fused_loss:
            print("[warn] fused-loss is enabled in estimate; probe uses standard HF loss, adjust logits_factor based on measured peak.")
        if self.cfg.flashattn:
            print("[warn] flashattn flag does not toggle kernels here; ensure your model actually uses FlashAttention when interpreting probe results.")

        torch.cuda.reset_peak_memory_stats(device)

        steps = self.cfg.warmup_steps + self.cfg.probe_steps
        for step in range(steps):
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            # Simulate gradient accumulation to better match per-step memory
            for _ in range(self.cfg.grad_accum):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
            if opt is not None:
                opt.step()

        alloc = torch.cuda.max_memory_allocated(device) / GiB
        reserved = torch.cuda.max_memory_reserved(device) / GiB
        return float(alloc), float(reserved)


# ---- CLI, printing, and utilities ----

def pretty_print(cfg: EstimatorInputs, out: EstimatorOutputs):
    print("\n=== Memory Estimate (per GPU) ===")
    print(f"Model: {cfg.model}")
    print(f"dtype={cfg.dtype}, seq_len={cfg.seq_len}, micro_batch={cfg.per_device_batch}, GA={cfg.grad_accum}")
    print(f"DP={cfg.dp}, TP={cfg.tp}, PP={cfg.pp}, ZeRO={cfg.zero}, LoRA={cfg.use_lora}, QLoRA={cfg.qlora}")
    print("")
    rows = [
        ("Base weights", out.base_weights_gib),
        ("Trainable state", out.trainable_state_gib),
        ("Attention activ.", out.attention_activ_gib),
        ("MLP activ.", out.mlp_activ_gib),
        ("Output logits", out.logits_gib),
        ("KV cache", out.kv_cache_gib),
        ("Vision activ.", out.vision_activ_gib),
        ("Misc. overhead", out.misc_overhead_gib),
        ("DeepSpeed overhead", out.deepspeed_overhead_gib),
        ("CUDA/cuBLAS", out.cuda_libs_gib),
        ("Fragmentation", out.fragmentation_gib),
        ("Total (steady)", out.steady_total_gib),
        ("Peak overhead", out.peak_overhead_gib),
        ("Total (peak)", out.peak_total_gib),
    ]
    colw = max(len(n) for n,_ in rows) + 2
    for name, val in rows:
        print(f"{name:<{colw}} : {val:6.2f} GiB")

    if out.measured_alloc_gib is not None:
        print("\n--- Empirical Probe ---")
        print(f"max_memory_allocated : {out.measured_alloc_gib:.2f} GiB")
        if out.measured_reserved_gib is not None:
            print(f"max_memory_reserved  : {out.measured_reserved_gib:.2f} GiB")
            diff = out.peak_total_gib - out.measured_reserved_gib
            print(f"Diff (estimate - reserved): {diff:+.2f} GiB")


def parse_bool(s: Optional[str]) -> bool:
    if s is None:
        return False
    return str(s).lower() in {"1","y","yes","t","true","on"}


def main():
    p = argparse.ArgumentParser(description="LLM/VLM GPU Memory Estimator + Probe")
    p.add_argument("--model", type=str, required=True, help="HF model id or path (config.json available)")
    p.add_argument("--is-seq2seq", type=parse_bool, default=False)

    # Training shape
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)

    # Parallelism
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)

    # Precision & optimizer
    p.add_argument("--dtype", type=str, default="bf16", choices=list(DTYPE_BYTES.keys()))
    p.add_argument("--optimizer", type=str, default="adamw")
    p.add_argument("--optimizer-bits", type=int, default=32, choices=[8,32])
    p.add_argument("--master-weights", type=parse_bool, default=True)

    # PEFT
    p.add_argument("--lora", dest="use_lora", type=parse_bool, default=False)
    p.add_argument("--qlora", type=parse_bool, default=False)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-target-modules", type=str, nargs="*", default=None)
    p.add_argument("--lora-params-override", type=int, default=None)

    # Memory-saving knobs
    p.add_argument("--flashattn", type=parse_bool, default=False)
    p.add_argument("--liger", type=parse_bool, default=False)
    p.add_argument("--fused-loss", type=parse_bool, default=False)
    p.add_argument("--grad-checkpoint", type=parse_bool, default=False)
    p.add_argument("--kv-cache", dest="use_kv_cache", type=parse_bool, default=False)

    # ZeRO
    p.add_argument("--zero", type=int, default=0, choices=[0,1,2,3])
    p.add_argument("--dp-offload-optimizer", dest="cpu_offload_optimizer", type=parse_bool, default=False)

    # Vision (VLM)
    p.add_argument("--use-vision", type=parse_bool, default=False)
    p.add_argument("--vision-hidden", type=int, default=1024)
    p.add_argument("--vision-layers", type=int, default=24)
    p.add_argument("--vision-image-size", type=int, default=448)
    p.add_argument("--vision-patch", type=int, default=14)

    # Overheads
    p.add_argument("--cuda-libs-gib", type=float, default=2.5)
    p.add_argument("--misc-overhead-gib", type=float, default=3.0)
    p.add_argument("--deepspeed-overhead-gib", type=float, default=1.0)
    p.add_argument("--fragmentation-gib", type=float, default=5.0)

    # Activation multipliers
    p.add_argument("--attn-act-factor-no-fa", type=float, default=4.0)
    p.add_argument("--attn-act-factor-fa", type=float, default=0.8)
    p.add_argument("--mlp-act-factor", type=float, default=6.0)
    p.add_argument("--logits-factor", type=float, default=1.0)
    p.add_argument("--grad-ckpt-reduction", type=float, default=0.35)

    # Probe
    p.add_argument("--probe", type=parse_bool, default=False)
    p.add_argument("--probe-steps", type=int, default=6)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--estimate-only", type=parse_bool, default=False)

    args = p.parse_args()

    if args.seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {args.seq_len}")
    if args.per_device_batch <= 0:
        raise ValueError(f"per-device-batch must be > 0, got {args.per_device_batch}")
    if args.grad_accum <= 0:
        raise ValueError(f"grad-accum must be > 0, got {args.grad_accum}")
    if args.dp <= 0 or args.tp <= 0 or args.pp <= 0:
        raise ValueError(f"dp, tp, pp must all be >= 1, got dp={args.dp}, tp={args.tp}, pp={args.pp}")
    if args.vision_patch == 0:
        raise ValueError("vision-patch must be non-zero")

    cfg = EstimatorInputs(
        model=args.model,
        is_seq2seq=args.is_seq2seq,
        seq_len=args.seq_len,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        dtype=args.dtype,
        optimizer=args.optimizer,
        optimizer_bits=args.optimizer_bits,
        master_weights=args.master_weights,
        use_lora=args.use_lora,
        qlora=args.qlora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        lora_params_override=args.lora_params_override,
        flashattn=args.flashattn,
        liger=args.liger,
        fused_loss=args.fused_loss,
        grad_checkpoint=args.grad_checkpoint,
        use_kv_cache=args.use_kv_cache,
        zero=args.zero,
        cpu_offload_optimizer=args.cpu_offload_optimizer,
        use_vision=args.use_vision,
        vision_hidden=args.vision_hidden,
        vision_layers=args.vision_layers,
        vision_image_size=args.vision_image_size,
        vision_patch=args.vision_patch,
        cuda_libs_gib=args.cuda_libs_gib,
        misc_overhead_gib=args.misc_overhead_gib,
        deepspeed_overhead_gib=args.deepspeed_overhead_gib,
        fragmentation_gib=args.fragmentation_gib,
        attn_act_factor_no_fa=args.attn_act_factor_no_fa,
        attn_act_factor_fa=args.attn_act_factor_fa,
        mlp_act_factor=args.mlp_act_factor,
        logits_factor=args.logits_factor,
        grad_ckpt_reduction=args.grad_ckpt_reduction,
        probe=args.probe,
        probe_steps=args.probe_steps,
        warmup_steps=args.warmup_steps,
        estimate_only=args.estimate_only,
    )

    estimator = MemoryEstimator(cfg)
    outs = estimator.estimate()

    # Empirical probe (optional)
    measured_alloc = measured_reserved = None
    if cfg.probe and not cfg.estimate_only:
        measured_alloc, measured_reserved = estimator.probe(outs)
        outs.measured_alloc_gib = measured_alloc
        outs.measured_reserved_gib = measured_reserved

    pretty_print(cfg, outs)


if __name__ == "__main__":
    main()
