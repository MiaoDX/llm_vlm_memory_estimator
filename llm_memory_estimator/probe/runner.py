"""
Empirical GPU memory probing runner.
"""

import time
import logging
from typing import Callable, Optional

from ..core.config import EstimatorConfig
from ..core.model_info import ModelInfo
from .results import DeviceInfo, PhaseMetrics, ProbeResult


logger = logging.getLogger(__name__)


class ProbeRunner:
    """
    Empirical GPU memory probe runner.

    Loads a model on GPU with configured optimizations and measures
    actual memory usage during training simulation.
    """

    def __init__(self, config: EstimatorConfig, model_info: ModelInfo):
        """
        Initialize probe runner.

        Args:
            config: Estimator configuration
            model_info: Model information

        Raises:
            ValueError: If configuration is invalid
            ImportError: If required libraries are not installed
        """
        self.config = config
        self.model_info = model_info

        # Validate configuration
        config.validate_config()

        # Select device
        self.device = self._select_device()

        # Initialize placeholders
        self.model = None
        self.optimizer = None

    def _select_device(self):
        """Select and validate GPU device"""
        import torch

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Probe requires a GPU with CUDA support."
            )

        # Validate device ID
        device_count = torch.cuda.device_count()
        if self.config.probe_device_id >= device_count:
            raise ValueError(
                f"Invalid device ID {self.config.probe_device_id}. "
                f"Available devices: 0-{device_count-1}"
            )

        device = torch.device(f"cuda:{self.config.probe_device_id}")
        logger.info(
            f"Selected device: {torch.cuda.get_device_name(device)} "
            f"(ID: {self.config.probe_device_id})"
        )
        return device

    def _get_device_info(self) -> DeviceInfo:
        """Get GPU device information"""
        import torch

        props = torch.cuda.get_device_properties(self.device)
        return DeviceInfo(
            device_id=self.config.probe_device_id,
            device_name=torch.cuda.get_device_name(self.device),
            total_memory_gib=props.total_memory / (1024**3),
            compute_capability=f"{props.major}.{props.minor}",
        )

    def _get_dtype(self):
        """Get PyTorch dtype from config"""
        import torch

        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }

        dtype = dtype_map.get(self.config.dtype)
        if dtype is None:
            raise ValueError(
                f"Unsupported dtype for probe: {self.config.dtype}. "
                f"Supported: fp32, bf16, fp16"
            )
        return dtype

    def _load_model(self):
        """
        Load model with optimizations.

        Returns loaded model on GPU.
        """
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

        logger.info(f"Loading model: {self.config.model}")

        # Branch on quantization
        if self.config.qlora_enabled:
            model = self._load_quantized_model()
        else:
            model = self._load_standard_model()

        # Apply gradient checkpointing
        if self.config.is_optimization_enabled("gradient_checkpointing"):
            logger.info("Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()

        # Apply LoRA (standard, non-quantized)
        if self.config.lora_enabled:
            model = self._apply_lora(model)

        # Apply Liger kernels
        if self.config.is_optimization_enabled("liger"):
            self._apply_liger_kernels(model)

        return model

    def _load_standard_model(self):
        """Load model with standard precision"""
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
        import torch

        dtype = self._get_dtype()

        # Prepare kwargs
        kwargs = {
            "torch_dtype": dtype,
            "device_map": {"": self.device},  # Load everything on selected device
        }

        # Enable FlashAttention if configured
        if self.config.is_optimization_enabled("flashattention"):
            logger.info("Loading with FlashAttention 2")
            kwargs["attn_implementation"] = "flash_attention_2"
            # This will fail-fast if flash-attn is not installed

        # Load model
        if self.model_info.is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model, **kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model, **kwargs
            )

        return model

    def _load_quantized_model(self):
        """Load model with quantization (QLoRA)"""
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            BitsAndBytesConfig,
        )
        import torch

        logger.info("Loading model with 4-bit quantization (QLoRA)")

        # Compute dtype for QLoRA
        compute_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float16

        # BitsAndBytes config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        # Load model
        if self.model_info.is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model,
                quantization_config=quant_config,
                device_map={"": self.device},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                quantization_config=quant_config,
                device_map={"": self.device},
            )

        # Apply LoRA on top of quantized model
        if self.config.has_lora:
            model = self._apply_lora(model)

        return model

    def _apply_lora(self, model):
        """Apply LoRA to model"""
        from peft import get_peft_model, LoraConfig, TaskType

        logger.info(
            f"Applying LoRA (rank={self.config.lora['rank']}, "
            f"alpha={self.config.lora['alpha']})"
        )

        # Get target modules
        target_modules = self.config.lora.get("target_modules", "all-linear")

        # Create LoRA config
        task_type = TaskType.SEQ_2_SEQ_LM if self.model_info.is_seq2seq else TaskType.CAUSAL_LM
        lora_config = LoraConfig(
            task_type=task_type,
            r=self.config.lora["rank"],
            lora_alpha=self.config.lora.get("alpha", self.config.lora["rank"] * 2),
            target_modules=target_modules,
            lora_dropout=self.config.lora.get("dropout", 0.05),
            bias="none",
        )

        # Apply PEFT
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def _apply_liger_kernels(self, model):
        """Apply Liger fused kernels"""
        # Liger kernel application varies by model architecture
        # For now, just log that it's enabled
        # TODO: Implement architecture-specific kernel application
        logger.warning("Liger kernel probe support not yet implemented")

    def _setup_optimizer(self):
        """Setup optimizer"""
        import torch

        logger.info("Setting up AdamW optimizer")

        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-5,  # Dummy learning rate for probe
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

    def _create_synthetic_batch(self):
        """Create synthetic input batch"""
        import torch

        batch_size = self.config.per_device_batch
        seq_len = self.config.seq_len

        # Create random input IDs
        input_ids = torch.randint(
            0,
            self.model_info.vocab_size,
            (batch_size, seq_len),
            device=self.device,
        )

        # Create attention mask (all ones)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _measure_phase(self, phase_name: str, fn: Callable) -> PhaseMetrics:
        """
        Measure memory for a specific phase.

        Args:
            phase_name: Name of the phase
            fn: Function to execute and measure

        Returns:
            PhaseMetrics with memory and timing info
        """
        import torch

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)

        # Execute phase
        start_time = time.perf_counter()
        result = fn()
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start_time

        # Get peak memory
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        max_reserved = torch.cuda.max_memory_reserved(self.device) / (1024**3)

        logger.info(
            f"{phase_name}: allocated={max_allocated:.2f} GiB, "
            f"reserved={max_reserved:.2f} GiB, time={elapsed:.2f}s"
        )

        return PhaseMetrics(
            phase=phase_name,
            max_allocated_gib=max_allocated,
            max_reserved_gib=max_reserved,
            elapsed_sec=elapsed,
        )

    def _run_warmup(self):
        """Run warmup steps"""
        logger.info(f"Running {self.config.probe_warmup} warmup steps...")

        for step in range(self.config.probe_warmup):
            batch = self._create_synthetic_batch()
            outputs = self.model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _run_training_step(self):
        """Run a single training step"""
        batch = self._create_synthetic_batch()
        outputs = self.model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def run(self) -> ProbeResult:
        """
        Execute GPU memory probe.

        Returns:
            ProbeResult with measurements and device info
        """
        import torch

        logger.info("Starting GPU memory probe...")

        # Get device info
        device_info = self._get_device_info()

        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        # Measure model loading
        def load_model_fn():
            self.model = self._load_model()
            return self.model

        phases = {}
        phases["model_load"] = self._measure_phase("Model Load", load_model_fn)

        # Setup optimizer
        self._setup_optimizer()

        # Run warmup (not measured)
        self._run_warmup()

        # Measure training phases
        logger.info(f"Running {self.config.probe_steps} measurement steps...")

        # Clear stats after warmup
        torch.cuda.reset_peak_memory_stats(self.device)

        # Run measurement steps
        for step in range(self.config.probe_steps):
            self._run_training_step()

        # Get peak memory across all training steps
        peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024**3)

        # Record training peak as a phase
        phases["training_peak"] = PhaseMetrics(
            phase="Training Peak",
            max_allocated_gib=peak_allocated,
            max_reserved_gib=peak_reserved,
            elapsed_sec=0.0,  # Not timing individual steps
        )

        # Create result
        result = ProbeResult(
            phases=phases,
            device_info=device_info,
            peak_allocated_gib=peak_allocated,
            peak_reserved_gib=peak_reserved,
        )

        logger.info(f"Probe complete. Peak allocated: {peak_allocated:.2f} GiB")

        return result
