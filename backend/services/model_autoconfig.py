"""3-layer candidate configuration builder for llama_cpp models.

Layer 1: Saved per-engine setup defaults (engine_setup_defaults.yaml).
Layer 2: Architecture / model-family defaults & thinking/sampling mode presets.
Layer 3: Quant / finetune sampling overrides from local generation_config.json or HF metadata.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from backend.routes import engine_setup_defaults


_HARDWARE_PROTECTED_KEYS = {
    "ctx_size",
    "batch_size",
    "ubatch_size",
    "cache_type_k",
    "cache_type_v",
    "parallel",
    "threads",
    "gpu_devices",
    "gpu_mode",
    "n_gpu_layers",
}

ARCHITECTURE_PRESETS: dict[str, dict[str, Any]] = {
    "lfm2moe": {
        "label": "LFM2 / LFM2.5 MoE",
        "engine_params": {
            "jinja": True,
            "temperature": 0.2,
            "top_k": 64,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
        },
        "modes": {
            "off": {"reasoning_effort": "off"},
            "low": {"reasoning_effort": "low"},
            "medium": {"reasoning_effort": "medium"},
            "high": {"reasoning_effort": "high"},
            "ultra": {"reasoning_effort": "ultra"},
            "omni": {"reasoning_effort": "omni"},
        },
    },
    "qwen3": {
        "label": "Qwen3 / Qwen3.5 / Qwen3.6",
        "engine_params": {
            "jinja": True,
            "flash_attn": "on",
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
        },
        "modes": {
            "think": {"chat_template_kwargs.enable_thinking": True},
            "no-think": {"chat_template_kwargs.enable_thinking": False},
            "off-precise": {"chat_template_kwargs.enable_thinking": False, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "off-balanced": {"chat_template_kwargs.enable_thinking": False, "temperature": 1.0, "top_p": 0.95, "top_k": 20},
            "off-creative": {"chat_template_kwargs.enable_thinking": False, "temperature": 1.1, "top_p": 0.95, "top_k": 40},
            "low-precise": {"chat_template_kwargs.enable_thinking": True, "chat_template_kwargs.reasoning_effort": "low", "temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "low-balanced": {"chat_template_kwargs.enable_thinking": True, "chat_template_kwargs.reasoning_effort": "low", "temperature": 1.0, "top_p": 0.95, "top_k": 20},
            "high-precise": {"chat_template_kwargs.enable_thinking": True, "chat_template_kwargs.reasoning_effort": "high", "temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "high-balanced": {"chat_template_kwargs.enable_thinking": True, "chat_template_kwargs.reasoning_effort": "high", "temperature": 1.0, "top_p": 0.95, "top_k": 20},
        },
    },
    "spark": {
        "label": "Spark X2.6",
        "engine_params": {
            "jinja": True,
            "flash_attn": "on",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "reasoning_effort": "low",
        },
        "modes": {
            "thinking-off": {"chat_template_kwargs.enable_thinking": False},
            "thinking-low": {"chat_template_kwargs.enable_thinking": True, "chat_template_kwargs.reasoning_effort": "low"},
            "thinking-high": {"chat_template_kwargs.enable_thinking": True, "chat_template_kwargs.reasoning_effort": "high"},
            "off-precise": {"chat_template_kwargs.enable_thinking": False, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "off-balanced": {"chat_template_kwargs.enable_thinking": False, "temperature": 1.0, "top_p": 0.95, "top_k": 20},
            "low-precise": {"chat_template_kwargs.enable_thinking": True, "chat_template_kwargs.reasoning_effort": "low", "temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "high-precise": {"chat_template_kwargs.enable_thinking": True, "chat_template_kwargs.reasoning_effort": "high", "temperature": 0.6, "top_p": 0.95, "top_k": 20},
        },
    },
    "deepseek": {
        "label": "DeepSeek",
        "engine_params": {
            "jinja": True,
            "flash_attn": "on",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        },
        "modes": {
            "think-precise": {"chat_template_kwargs.enable_thinking": True, "temperature": 0.3, "top_p": 0.8, "top_k": 20},
            "think-balanced": {"chat_template_kwargs.enable_thinking": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20},
            "no-think-precise": {"chat_template_kwargs.enable_thinking": False, "temperature": 0.3, "top_p": 0.8, "top_k": 20},
            "no-think-balanced": {"chat_template_kwargs.enable_thinking": False, "temperature": 0.7, "top_p": 0.9, "top_k": 20},
        },
    },
    "generic": {
        "label": "Generic GGUF",
        "engine_params": {
            "jinja": True,
            "flash_attn": "on",
        },
        "modes": {
            "precise": {"temperature": 0.2, "top_p": 0.85},
            "balanced": {"temperature": 0.6, "top_p": 0.95},
            "creative": {"temperature": 0.9, "top_p": 0.95, "min_p": 0.05},
        },
    },
}


def detect_model_family(model: dict[str, Any]) -> str:
    """Identify the architecture/preset family from Studio model metadata."""
    haystack = " ".join(
        str(model.get(key) or "")
        for key in ("model_type", "architecture", "huggingface_id", "display_name", "id", "path")
    ).lower()
    if "lfm2" in haystack or "lfm2.5" in haystack or "l25-" in haystack:
        return "lfm2moe"
    if "spark" in haystack:
        return "spark"
    if "qwen" in haystack:
        return "qwen3"
    if "deepseek" in haystack:
        return "deepseek"
    return "generic"


def extract_local_generation_overrides(model: dict[str, Any]) -> dict[str, Any]:
    """Read sampling parameters from generation_config.json next to the weight, if present."""
    raw_path = model.get("path") or model.get("local_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {}
    weight_path = Path(raw_path).expanduser()
    parent = weight_path if weight_path.is_dir() else weight_path.parent
    gen_path = parent / "generation_config.json"
    if not gen_path.is_file() or gen_path.is_symlink():
        return {}
    try:
        data = json.loads(gen_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    overrides: dict[str, Any] = {}
    for src, dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("repetition_penalty", "repeat_penalty"),
    ):
        val = data.get(src)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            overrides[dst] = val
    return overrides


def build_llama_cpp_autoconfig(
    store: Any,
    model: dict[str, Any],
    *,
    existing_config: dict[str, Any] | None = None,
    hf_overrides: dict[str, Any] | None = None,
    overwrite_existing: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a 3-layer candidate llama_cpp config for a model."""
    defaults_doc = engine_setup_defaults._read_defaults(store, "llama_cpp")
    engine_defaults = copy.deepcopy(defaults_doc.get("values") or {})
    family = detect_model_family(model)
    arch_preset = ARCHITECTURE_PRESETS[family]
    local_overrides = extract_local_generation_overrides(model)
    combined_overrides = {**local_overrides, **(hf_overrides or {})}

    after = copy.deepcopy(existing_config or {})
    if not isinstance(after, dict):
        after = {}
    after["engine"] = "llama_cpp"
    engines = after.setdefault("engines", {})
    if not isinstance(engines, dict):
        engines = {}
        after["engines"] = engines
    section = copy.deepcopy(engines.get("llama_cpp") or {})
    if not isinstance(section, dict):
        section = {}

    applied_engine_defaults: list[str] = []
    for key, value in engine_defaults.items():
        if overwrite_existing or key not in section or section[key] in (None, ""):
            section[key] = copy.deepcopy(value)
            applied_engine_defaults.append(key)

    applied_arch_params: list[str] = []
    for key, value in arch_preset["engine_params"].items():
        if overwrite_existing or key not in section or section[key] in (None, ""):
            section[key] = copy.deepcopy(value)
            applied_arch_params.append(key)

    applied_hf_overrides: list[str] = []
    for key, value in combined_overrides.items():
        if key in _HARDWARE_PROTECTED_KEYS:
            continue
        section[key] = copy.deepcopy(value)
        applied_hf_overrides.append(key)

    existing_modes = section.get("set_params_by_id")
    if not isinstance(existing_modes, list):
        existing_modes = []
    existing_ids = {
        row["sub_id"].strip()
        for row in existing_modes
        if isinstance(row, dict) and isinstance(row.get("sub_id"), str) and row["sub_id"].strip()
    }
    added_modes: list[str] = []
    for mode_id, params in arch_preset["modes"].items():
        if mode_id not in existing_ids:
            existing_modes.append({"sub_id": mode_id, "params": copy.deepcopy(params)})
            added_modes.append(mode_id)
    if existing_modes:
        section["set_params_by_id"] = existing_modes

    engines["llama_cpp"] = section
    summary = {
        "family": family,
        "family_label": arch_preset["label"],
        "applied_engine_defaults": applied_engine_defaults,
        "applied_arch_params": applied_arch_params,
        "applied_hf_overrides": applied_hf_overrides,
        "added_modes": added_modes,
    }
    return after, summary


def maybe_initialize_new_gguf_model(store: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Populate candidate llama_cpp defaults when a new non-embedding GGUF model is added."""
    if not isinstance(record, dict) or record.get("format") != "gguf":
        return record
    if record.get("is_embedding_model") or (record.get("config") or {}):
        return record
    defaults_doc = engine_setup_defaults._read_defaults(store, "llama_cpp")
    if not (defaults_doc.get("values") or {}):
        return record
    config, _summary = build_llama_cpp_autoconfig(store, record, existing_config={})
    updated = copy.deepcopy(record)
    updated["config"] = config
    updated["config_review_source"] = "auto_candidate_llama_cpp"
    return updated
