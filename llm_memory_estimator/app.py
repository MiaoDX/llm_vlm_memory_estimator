#!/usr/bin/env python3
"""
Gradio UI for LLM/VLM GPU Memory Estimator
Hugging Face Spaces deployment
"""
import gradio as gr
import pandas as pd
from .core.estimator import MemoryEstimator
from .core.config import EstimatorConfig
from .core.results import EstimationResult

def estimate_memory(
    model_name, dtype, seq_len, batch_size, grad_accum,
    dp, tp, pp, zero_stage,
    flashattn, grad_ckpt, fused_loss,
    use_lora, qlora, lora_rank, lora_targets,
    use_kv_cache,
    grad_ckpt_reduction, attn_factor_no_fa, attn_factor_fa, mlp_factor,
    fragmentation_gib, misc_overhead_gib
):
    """Run memory estimation with given parameters."""
    # Parse LoRA target modules if provided
    target_modules = None
    if lora_targets and lora_targets.strip():
        target_modules = [m.strip() for m in lora_targets.split(",") if m.strip()]

    # Build optimizations dict
    optimizations = {}
    if flashattn:
        optimizations["flashattention"] = {}
    if grad_ckpt:
        optimizations["gradient_checkpointing"] = {"reduction": grad_ckpt_reduction}
    if fused_loss:
        optimizations["fused_loss"] = {}

    # Build LoRA config
    lora = None
    if use_lora or qlora:
        lora = {
            "rank": int(lora_rank),
            "quantized": qlora,
            "target_modules": target_modules,
        }

    # Calculate DeepSpeed overhead
    deepspeed_overhead_gib = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}.get(int(zero_stage), 0.0)

    # Create configuration
    cfg = EstimatorConfig(
        model=model_name,
        dtype=dtype,
        seq_len=int(seq_len),
        per_device_batch=int(batch_size),
        grad_accum=int(grad_accum),
        dp=int(dp),
        tp=int(tp),
        pp=int(pp),
        zero_stage=int(zero_stage),
        optimizations=optimizations,
        lora=lora,
        use_kv_cache=use_kv_cache,
        attn_act_factor_no_fa=attn_factor_no_fa,
        attn_act_factor_fa=attn_factor_fa,
        mlp_act_factor=mlp_factor,
        fragmentation_gib=fragmentation_gib,
        misc_overhead_gib=misc_overhead_gib,
        deepspeed_overhead_gib=deepspeed_overhead_gib,
    )

    # Run estimation (only wrap this critical section)
    try:
        estimator = MemoryEstimator(cfg)
        result = estimator.estimate()
    except FileNotFoundError as e:
        error_msg = f"## ❌ Model Not Found\n\n{str(e)}\n\nPlease check the model name and ensure it exists on HuggingFace or locally."
        empty_df = pd.DataFrame({'Category': [], 'Memory (GiB)': []})
        return error_msg, empty_df
    except ValueError as e:
        error_msg = f"## ❌ Invalid Configuration\n\n{str(e)}\n\nPlease check your parameter values."
        empty_df = pd.DataFrame({'Category': [], 'Memory (GiB)': []})
        return error_msg, empty_df
    except Exception as e:
        # Catch-all for unexpected errors - show full details
        error_msg = f"## ❌ Estimation Error\n\n```\n{str(e)}\n```\n\nPlease report this if the error persists."
        empty_df = pd.DataFrame({'Category': [], 'Memory (GiB)': []})
        return error_msg, empty_df

    # Extract values from breakdown
    breakdown_data = result.breakdown

    # Create summary
    summary = f"""
## 📊 Memory Estimate (per GPU)

### Key Metrics
- **Steady State:** `{result.steady_total_gib:.2f} GiB`
- **Peak Memory:** `{result.peak_total_gib:.2f} GiB`
- **Peak Overhead:** `{result.peak_overhead_gib:.2f} GiB`

### Configuration
- Model: `{model_name}`
- Precision: `{dtype}`, Sequence: `{seq_len}`, Batch: `{batch_size}`, GA: `{grad_accum}`
- Parallelism: DP={dp}, TP={tp}, PP={pp}, ZeRO={zero_stage}
- Optimizations: FA={flashattn}, GradCkpt={grad_ckpt}, FusedLoss={fused_loss}
- LoRA: {use_lora}, QLoRA: {qlora}
"""

    # Create detailed breakdown table
    breakdown = pd.DataFrame({
        'Category': [
            'Base Weights',
            'Trainable State',
            'Attention Activations',
            'MLP Activations',
            'Output Logits',
            'KV Cache',
            'Vision Activations',
            'CUDA/cuBLAS',
            'Misc Overhead',
            'DeepSpeed Overhead',
            'Fragmentation',
        ],
        'Memory (GiB)': [
            f"{breakdown_data.get('weights', {}).get('gib', 0.0):.2f}",
            f"{breakdown_data.get('trainable_state', {}).get('gib', 0.0):.2f}",
            f"{breakdown_data.get('attention_activations', {}).get('gib', 0.0):.2f}",
            f"{breakdown_data.get('mlp_activations', {}).get('gib', 0.0):.2f}",
            f"{breakdown_data.get('logits', {}).get('gib', 0.0):.2f}",
            f"{breakdown_data.get('kv_cache', {}).get('gib', 0.0):.2f}",
            f"{breakdown_data.get('vision_activations', {}).get('gib', 0.0):.2f}",
            f"{breakdown_data.get('cuda_libs', {}).get('gib', 0.0):.2f}",
            f"{breakdown_data.get('misc_overhead', {}).get('gib', 0.0):.2f}",
            f"{breakdown_data.get('deepspeed_overhead', {}).get('gib', 0.0):.2f}",
            f"{breakdown_data.get('fragmentation', {}).get('gib', 0.0):.2f}",
        ]
    })

    # Hardware recommendation
    gpu_options = []
    peak_mem = result.peak_total_gib

    if peak_mem <= 16:
        gpu_options.append("✅ RTX 4090 (24GB), RTX 3090 (24GB), A10 (24GB)")
    if peak_mem <= 24:
        gpu_options.append("✅ RTX 4090 (24GB), A10 (24GB), L4 (24GB)")
    if peak_mem <= 40:
        gpu_options.append("✅ A100 40GB, A6000 (48GB)")
    if peak_mem <= 48:
        gpu_options.append("✅ A6000 (48GB), L40S (48GB)")
    if peak_mem <= 80:
        gpu_options.append("✅ A100 80GB, H100 80GB")
    if peak_mem <= 141:
        gpu_options.append("✅ H200 (141GB)")

    if not gpu_options:
        gpu_options.append("⚠️ Requires multiple GPUs or model parallelism")

    hardware_rec = "### 🖥️ Recommended GPUs\n" + "\n".join(gpu_options)

    full_summary = summary + "\n" + hardware_rec

    return full_summary, breakdown


# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="LLM/VLM Memory Estimator",
    css="""
    .gradio-container {max-width: 1200px !important}
    #estimate-btn {height: 60px; font-size: 18px;}
    """
) as demo:
    gr.Markdown("""
    # 🧮 LLM/VLM GPU Memory Estimator

    Estimate per-GPU VRAM requirements for LLM/VLM fine-tuning.
    Supports **LoRA**, **QLoRA**, **FlashAttention**, **DeepSpeed ZeRO**, and **multi-GPU setups**.

    📖 [Documentation](https://github.com/YOUR_REPO) | 🐛 [Report Issues](https://github.com/YOUR_REPO/issues)
    """)

    with gr.Tabs():
        # ===== BASIC TAB =====
        with gr.Tab("🎛️ Basic Configuration"):
            with gr.Row():
                with gr.Column():
                    model_name = gr.Textbox(
                        value="Qwen/Qwen2.5-7B-Instruct",
                        label="Model Name",
                        info="HuggingFace model ID or local path",
                        placeholder="Qwen/Qwen2.5-7B-Instruct"
                    )
                    dtype = gr.Dropdown(
                        choices=["bf16", "fp16", "fp32", "fp8", "int8", "int4", "nf4"],
                        value="bf16",
                        label="Precision (dtype)",
                        info="Training precision"
                    )

                with gr.Column():
                    seq_len = gr.Slider(
                        minimum=128,
                        maximum=32768,
                        value=4096,
                        step=128,
                        label="Sequence Length",
                        info="Max sequence length for training"
                    )

            with gr.Row():
                batch_size = gr.Number(
                    value=1,
                    label="Micro Batch Size",
                    info="Per-device batch size",
                    precision=0,
                    minimum=1
                )
                grad_accum = gr.Number(
                    value=1,
                    label="Gradient Accumulation Steps",
                    info="Number of steps to accumulate gradients",
                    precision=0,
                    minimum=1
                )

        # ===== PARALLELISM TAB =====
        with gr.Tab("🔀 Parallelism & ZeRO"):
            gr.Markdown("### Multi-GPU Parallelism")
            with gr.Row():
                dp = gr.Number(
                    value=1,
                    label="Data Parallel (DP)",
                    info="Number of data parallel replicas",
                    precision=0,
                    minimum=1
                )
                tp = gr.Number(
                    value=1,
                    label="Tensor Parallel (TP)",
                    info="Tensor parallelism degree",
                    precision=0,
                    minimum=1
                )
                pp = gr.Number(
                    value=1,
                    label="Pipeline Parallel (PP)",
                    info="Pipeline parallelism stages",
                    precision=0,
                    minimum=1
                )

            gr.Markdown("### DeepSpeed ZeRO")
            zero_stage = gr.Radio(
                choices=[0, 1, 2, 3],
                value=0,
                label="ZeRO Stage",
                info="0: None, 1: Optimizer, 2: Optimizer+Gradients, 3: Optimizer+Gradients+Parameters"
            )

        # ===== OPTIMIZATIONS TAB =====
        with gr.Tab("⚡ Memory Optimizations"):
            gr.Markdown("### Kernel Optimizations")
            flashattn = gr.Checkbox(
                value=True,
                label="FlashAttention",
                info="Enable FlashAttention-2 for reduced memory usage"
            )
            grad_ckpt = gr.Checkbox(
                value=True,
                label="Gradient Checkpointing",
                info="Trade compute for memory by recomputing activations"
            )
            fused_loss = gr.Checkbox(
                value=False,
                label="Fused Loss (Liger Kernel)",
                info="Fused cross-entropy loss for reduced memory"
            )
            use_kv_cache = gr.Checkbox(
                value=False,
                label="KV Cache",
                info="Enable KV cache (typically for inference, not training)"
            )

        # ===== LORA TAB =====
        with gr.Tab("🎯 LoRA / QLoRA"):
            gr.Markdown("### Parameter-Efficient Fine-Tuning")
            with gr.Row():
                use_lora = gr.Checkbox(
                    value=False,
                    label="Enable LoRA",
                    info="Use Low-Rank Adaptation"
                )
                qlora = gr.Checkbox(
                    value=False,
                    label="Enable QLoRA",
                    info="4-bit quantized base weights + LoRA"
                )

            lora_rank = gr.Slider(
                minimum=1,
                maximum=256,
                value=8,
                step=1,
                label="LoRA Rank (r)",
                info="Rank of LoRA adapters"
            )
            lora_targets = gr.Textbox(
                value="",
                label="LoRA Target Modules (optional)",
                placeholder="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                info="Comma-separated list. Leave empty for defaults."
            )

        # ===== ADVANCED TAB =====
        with gr.Tab("⚙️ Advanced Tuning"):
            gr.Markdown("### Activation Memory Factors")
            gr.Markdown("Adjust these multipliers to calibrate estimates against empirical probe results.")

            with gr.Row():
                grad_ckpt_reduction = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.35,
                    step=0.05,
                    label="Gradient Checkpoint Reduction",
                    info="Proportion of activations retained (lower = more savings)"
                )
                attn_factor_no_fa = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.5,
                    label="Attention Factor (no FA)",
                    info="Activation multiplier without FlashAttention"
                )

            with gr.Row():
                attn_factor_fa = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=0.8,
                    step=0.1,
                    label="Attention Factor (FA)",
                    info="Activation multiplier with FlashAttention"
                )
                mlp_factor = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=6.0,
                    step=0.5,
                    label="MLP Factor",
                    info="MLP activation multiplier"
                )

            gr.Markdown("### Overhead Cushions (GiB)")
            with gr.Row():
                fragmentation_gib = gr.Slider(
                    minimum=0.0,
                    maximum=20.0,
                    value=5.0,
                    step=0.5,
                    label="Fragmentation Reserve",
                    info="Memory fragmentation cushion"
                )
                misc_overhead_gib = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.5,
                    label="Misc Overhead",
                    info="Framework overhead cushion"
                )

    # ===== CALCULATE BUTTON =====
    calculate_btn = gr.Button(
        "💾 Calculate Memory Estimate",
        variant="primary",
        size="lg",
        elem_id="estimate-btn"
    )

    # ===== RESULTS =====
    gr.Markdown("## Results")

    with gr.Row():
        with gr.Column(scale=1):
            summary_output = gr.Markdown(label="Summary")
        with gr.Column(scale=1):
            breakdown_output = gr.Dataframe(
                headers=['Category', 'Memory (GiB)'],
                label="Detailed Breakdown",
                wrap=True
            )

    # ===== EXAMPLES =====
    gr.Markdown("## 🚀 Example Configurations")
    gr.Examples(
        examples=[
            # [model, dtype, seq, batch, ga, dp, tp, pp, zero, fa, gc, fl, lora, qlora, rank, targets, kv, ckpt_red, attn_nofa, attn_fa, mlp, frag, misc]
            [
                "Qwen/Qwen2.5-7B-Instruct", "bf16", 4096, 2, 128, 8, 1, 1, 2,
                True, True, True, False, False, 8, "", False,
                0.35, 4.0, 0.8, 6.0, 5.0, 3.0
            ],
            [
                "Qwen/Qwen2.5-7B-Instruct", "bf16", 4096, 1, 64, 4, 2, 1, 3,
                True, True, False, True, True, 8, "q_proj,k_proj,v_proj,o_proj", False,
                0.35, 4.0, 0.8, 6.0, 5.0, 3.0
            ],
            [
                "Qwen/Qwen2.5-14B-Instruct", "bf16", 8192, 1, 256, 16, 1, 1, 3,
                True, True, True, False, False, 8, "", False,
                0.35, 4.0, 0.8, 6.0, 5.0, 3.0
            ],
            [
                "Qwen/Qwen2.5-72B-Instruct", "bf16", 4096, 1, 512, 32, 4, 1, 3,
                True, True, True, True, False, 64, "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", False,
                0.35, 4.0, 0.8, 6.0, 5.0, 3.0
            ],
            [
                "Qwen/Qwen2.5-VL-7B-Instruct", "bf16", 4096, 1, 64, 4, 1, 1, 2,
                True, True, False, False, False, 8, "", False,
                0.35, 4.0, 0.8, 6.0, 5.0, 3.0
            ],
        ],
        inputs=[
            model_name, dtype, seq_len, batch_size, grad_accum,
            dp, tp, pp, zero_stage, flashattn, grad_ckpt, fused_loss,
            use_lora, qlora, lora_rank, lora_targets, use_kv_cache,
            grad_ckpt_reduction, attn_factor_no_fa, attn_factor_fa, mlp_factor,
            fragmentation_gib, misc_overhead_gib
        ],
        label="Try these presets",
        examples_per_page=4
    )

    # ===== FOOTER =====
    gr.Markdown("""
    ---
    ### 📝 Notes
    - **Estimates are approximate.** Real memory usage depends on kernels, framework versions, and runtime behavior.
    - Use empirical probe (see [estimator script](https://github.com/YOUR_REPO/scripts/estimate_memory_budget.py)) to calibrate factors.
    - Keep peak ≤ 75-80% of physical HBM for safety margin.
    - For QLoRA: base weights use ~0.56 bytes/param (4-bit + scales/zeros overhead).

    ### 🔧 Calibration
    Run the estimator script with `--probe true` to measure actual memory usage and adjust the advanced tuning factors accordingly.
    """)

    # ===== CONNECT BUTTON TO FUNCTION =====
    calculate_btn.click(
        fn=estimate_memory,
        inputs=[
            model_name, dtype, seq_len, batch_size, grad_accum,
            dp, tp, pp, zero_stage, flashattn, grad_ckpt, fused_loss,
            use_lora, qlora, lora_rank, lora_targets, use_kv_cache,
            grad_ckpt_reduction, attn_factor_no_fa, attn_factor_fa, mlp_factor,
            fragmentation_gib, misc_overhead_gib
        ],
        outputs=[summary_output, breakdown_output]
    )

def launch_app(**kwargs):
    """Launch the Gradio app.

    Args:
        **kwargs: Additional arguments to pass to demo.launch()
    """
    return demo.launch(**kwargs)

# Launch the app
if __name__ == "__main__":
    import sys
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("Starting Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False
    )
