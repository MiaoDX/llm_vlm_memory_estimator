"""
Configuration file loading (YAML/JSON).
"""

import json
from pathlib import Path
from typing import Union, Any, Dict

from ..core.config import EstimatorConfig
from .validator import validate_config


def load_config(path: Union[str, Path]) -> EstimatorConfig:
    """
    Load configuration from YAML or JSON file.

    Args:
        path: Path to config file (.yaml, .yml, or .json)

    Returns:
        EstimatorConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format unsupported or config invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        return load_yaml_config(path)
    elif suffix == ".json":
        return load_json_config(path)
    else:
        raise ValueError(
            f"Unsupported config file format: {suffix}. "
            "Supported: .yaml, .yml, .json"
        )


def load_yaml_config(path: Path) -> EstimatorConfig:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        EstimatorConfig instance

    Raises:
        RuntimeError: If PyYAML not installed
        ValueError: If YAML invalid
    """
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML is required for YAML config loading. "
            "Install with: pip install pyyaml"
        )

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return _dict_to_config(data)


def load_json_config(path: Path) -> EstimatorConfig:
    """
    Load configuration from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        EstimatorConfig instance
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return _dict_to_config(data)


def _dict_to_config(data: Dict[str, Any]) -> EstimatorConfig:
    """
    Convert dictionary to EstimatorConfig.

    Handles nested structures (e.g., training.seq_len, parallelism.dp).

    Args:
        data: Configuration dictionary

    Returns:
        EstimatorConfig instance
    """
    # Flatten nested structures
    flat_data = {}

    # Top-level fields
    flat_data["model"] = data.get("model")
    flat_data["is_seq2seq"] = data.get("is_seq2seq", False)
    flat_data["vocab_size_override"] = data.get("vocab_size_override")

    # Training section
    training = data.get("training", {})
    flat_data["seq_len"] = training.get("seq_len", data.get("seq_len", 2048))
    flat_data["per_device_batch"] = training.get("batch_size", data.get("batch", 1))
    flat_data["grad_accum"] = training.get("grad_accum", data.get("grad_accum", 1))

    # Precision
    flat_data["dtype"] = training.get("dtype", data.get("dtype", "bf16"))

    # Parallelism section
    parallelism = data.get("parallelism", {})
    flat_data["dp"] = parallelism.get("dp", data.get("dp", 1))
    flat_data["tp"] = parallelism.get("tp", data.get("tp", 1))
    flat_data["pp"] = parallelism.get("pp", data.get("pp", 1))
    flat_data["zero_stage"] = parallelism.get("zero_stage", data.get("zero", 0))

    # LoRA section
    lora = data.get("lora")
    if lora is not None and (isinstance(lora, dict) or lora):
        flat_data["lora"] = lora if isinstance(lora, dict) else {}

    # Optimizations - can be list or dict
    optimizations = data.get("optimizations", {})
    if isinstance(optimizations, list):
        # Convert list to dict with empty params
        flat_data["optimizations"] = {opt: {} for opt in optimizations}
    elif isinstance(optimizations, dict):
        flat_data["optimizations"] = optimizations
    else:
        flat_data["optimizations"] = {}

    # Vision section
    vision = data.get("vision")
    if vision is not None:
        flat_data["vision"] = vision

    # Probe section
    probe = data.get("probe", {})
    flat_data["probe_enabled"] = probe.get("enabled", data.get("probe", False))
    flat_data["probe_steps"] = probe.get("steps", 6)
    flat_data["probe_warmup"] = probe.get("warmup", 3)

    # Overheads section
    overheads = data.get("overheads", {})
    flat_data["cuda_libs_gib"] = overheads.get("cuda_libs_gib", data.get("cuda_libs_gib", 2.5))
    flat_data["misc_overhead_gib"] = overheads.get("misc_gib", data.get("misc_overhead_gib", 3.0))
    flat_data["fragmentation_gib"] = overheads.get("fragmentation_gib", data.get("fragmentation_gib", 5.0))
    flat_data["deepspeed_overhead_gib"] = overheads.get("deepspeed_gib", data.get("deepspeed_overhead_gib", 1.0))

    # Tunable factors (keep defaults if not specified)
    flat_data["attn_act_factor_no_fa"] = data.get("attn_act_factor_no_fa", 4.0)
    flat_data["attn_act_factor_fa"] = data.get("attn_act_factor_fa", 0.8)
    flat_data["mlp_act_factor"] = data.get("mlp_act_factor", 6.0)
    flat_data["logits_factor"] = data.get("logits_factor", 1.0)
    flat_data["grad_ckpt_reduction"] = data.get("grad_ckpt_reduction", 0.35)

    # KV cache
    flat_data["use_kv_cache"] = data.get("use_kv_cache", False)

    # Create config
    config = EstimatorConfig(**flat_data)

    # Validate
    validate_config(config)

    return config
