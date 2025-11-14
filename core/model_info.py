"""
Model information extraction from HuggingFace configs.

This module handles:
- Loading HF model configurations
- Counting parameters via meta-device instantiation
- Extracting architecture details (hidden_size, layers, heads, etc.)
- LoRA parameter estimation
"""

from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EstimatorConfig


@dataclass
class ModelInfo:
    """
    Model architecture information extracted from HF config.

    Attributes:
        param_count: Total parameter count
        hidden_size: Model hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        vocab_size: Vocabulary size
        intermediate_size: MLP intermediate dimension (for LoRA estimation)
        head_dim: Attention head dimension
    """
    param_count: int
    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    intermediate_size: Optional[int] = None
    head_dim: Optional[int] = None

    def __post_init__(self):
        """Calculate derived fields"""
        if self.head_dim is None and self.num_heads > 0:
            self.head_dim = self.hidden_size // self.num_heads
        if self.intermediate_size is None:
            # Common default for MLP: 4x hidden
            self.intermediate_size = 4 * self.hidden_size

    @classmethod
    def from_config(cls, model_name_or_path: str, vocab_size_override: Optional[int] = None) -> 'ModelInfo':
        """
        Load model info from HuggingFace config.

        Args:
            model_name_or_path: HF model ID or local path
            vocab_size_override: Optional vocab size override

        Returns:
            ModelInfo with extracted architecture details

        Raises:
            RuntimeError: If transformers/accelerate not installed or config invalid
        """
        try:
            from transformers import AutoConfig
        except ImportError:
            raise RuntimeError(
                "transformers is required for model config loading. "
                "Install with: pip install transformers"
            )

        # Load config
        hf_config = AutoConfig.from_pretrained(model_name_or_path)

        # Extract common fields with fallbacks for different architectures
        hidden_size = (
            getattr(hf_config, "hidden_size", None) or
            getattr(hf_config, "n_embd", None) or
            getattr(hf_config, "d_model", None)
        )
        num_layers = (
            getattr(hf_config, "num_hidden_layers", None) or
            getattr(hf_config, "n_layer", None) or
            getattr(hf_config, "num_layers", None)
        )
        num_heads = (
            getattr(hf_config, "num_attention_heads", None) or
            getattr(hf_config, "n_head", None)
        )
        vocab_size = vocab_size_override or getattr(hf_config, "vocab_size", None)
        intermediate_size = getattr(hf_config, "intermediate_size", None)

        if hidden_size is None or num_layers is None:
            raise RuntimeError(
                f"Could not extract hidden_size/num_layers from {model_name_or_path}. "
                "Please check the model config or use a supported architecture."
            )

        if num_heads is None:
            # Fallback: assume 128-dim heads
            num_heads = max(1, hidden_size // 128)
            print(f"[warn] num_attention_heads not found, assuming {num_heads} heads")

        if vocab_size is None:
            vocab_size = 50272  # Common default
            print(f"[warn] vocab_size not found, assuming {vocab_size}")

        # Count parameters using meta-device
        param_count = cls._count_params_meta(model_name_or_path, hf_config)

        return cls(
            param_count=param_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
        )

    @staticmethod
    def _count_params_meta(model_name_or_path: str, hf_config) -> int:
        """
        Count parameters using meta-device instantiation (no memory allocation).

        Args:
            model_name_or_path: HF model ID or local path
            hf_config: Already-loaded HF config

        Returns:
            Total parameter count

        Raises:
            RuntimeError: If required packages not installed
        """
        try:
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
            from accelerate import init_empty_weights
            import torch
        except ImportError:
            raise RuntimeError(
                "transformers and accelerate are required for parameter counting. "
                "Install with: pip install transformers accelerate"
            )

        # Determine model type from config
        is_seq2seq = hasattr(hf_config, "is_encoder_decoder") and hf_config.is_encoder_decoder

        try:
            with init_empty_weights():
                if is_seq2seq:
                    model = AutoModelForSeq2SeqLM.from_config(hf_config)
                else:
                    model = AutoModelForCausalLM.from_config(hf_config)

            total = sum(p.numel() for p in model.parameters())
            return int(total)

        except Exception as e:
            raise RuntimeError(
                f"Failed to count parameters for {model_name_or_path}: {e}"
            )

    def estimate_lora_params(
        self,
        config: 'EstimatorConfig'
    ) -> int:
        """
        Estimate LoRA parameter count based on target modules.

        This is a heuristic estimation. For exact counts, use config.lora["params_override"].

        Args:
            config: Estimator configuration with LoRA settings

        Returns:
            Estimated LoRA parameter count
        """
        if not config.has_lora:
            return 0

        lora_cfg = config.lora
        if "params_override" in lora_cfg and lora_cfg["params_override"] is not None:
            return int(lora_cfg["params_override"])

        rank = lora_cfg.get("rank", 8)
        target_modules = lora_cfg.get("target_modules", None)

        # Default targets for modern LLaMA/Mistral-style models
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "gate_proj", "down_proj"
            ]

        H = self.hidden_size
        I = self.intermediate_size or (4 * H)
        L = self.num_layers

        params = 0
        for module_name in target_modules:
            # Estimate (in_dim, out_dim) based on module role
            if module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                # Attention projections: H -> H
                in_dim = H
                out_dim = H
            elif module_name in {"up_proj", "gate_proj"}:
                # MLP up projections: H -> I
                in_dim = H
                out_dim = I
            elif module_name in {"down_proj"}:
                # MLP down projection: I -> H
                in_dim = I
                out_dim = H
            else:
                # Fallback: assume H -> H
                in_dim = H
                out_dim = H

            # LoRA adds A (out_dim, rank) and B (rank, in_dim)
            # Total params per module: rank * (in_dim + out_dim)
            params_per_layer = rank * (in_dim + out_dim)
            params += params_per_layer * L

        return int(params)
