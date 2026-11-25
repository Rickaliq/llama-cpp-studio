"""Read-only per-engine model setup defaults.

These values are kept outside models.yaml and are never consumed by runtime
launch code. The model setup UI may use them to seed an unsaved form section.
The companion owns editing and persistence.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.concurrency import run_blocking
from backend.data_store import get_store
from backend.engines.params import get_version_entry, param_index_from_entry
from backend.engines.params import CATALOG_FILENAME
from backend.engines.registry import VALID_ENGINE_IDS, active_engine_row_is_runnable

router = APIRouter()
FILENAME = "engine_setup_defaults.yaml"

# Only portable model/runtime tuning belongs in setup defaults. In particular,
# never persist host paths, model selectors, credentials, network endpoints,
# TLS material, RPC targets, or files/media references from a broad CLI scan.
PORTABLE_RUNTIME_KEYS = frozenset({
    "n_ctx", "ctx_size", "context_size", "contextsize", "ctx", "max_model_len", "max_seq_len",
    "n_batch", "batch_size", "batchsize", "max_num_batched_tokens", "n_ubatch", "ubatch_size", "ubatchsize", "ubatch",
    "max_num_seqs", "n_threads", "threads", "n_threads_batch", "threads_batch",
    "n_gpu_layers", "gpu_layers", "gpulayers", "tensor_parallel_size", "pipeline_parallel_size",
    "data_parallel_size", "main_gpu", "split_mode", "tensor_split", "gpu_memory_utilization",
    "dtype", "cache_type_k", "cache_type_v", "cache_type_k_draft", "cache_type_v_draft",
    "cache_type_k_first", "cache_type_k_last", "cache_type_v_first", "cache_type_v_last",
    "indexer_cache_type_k", "kv_cache_dtype", "flash_attn", "offload_kqv", "kv_offload", "no_kv_offload",
    "rope_freq_base", "rope_freq_scale", "rope_scaling",
    "mmap", "use_mmap", "no_mmap", "mlock", "use_mlock", "numa", "cont_batching", "parallel", "device", "dev", "usevulkan", "usecpu", "quantkv",
    "cpu_moe", "n_cpu_moe", "moe_cache_mib", "no_fused_moe", "n_gpu_layers_draft", "gpu_layers_draft", "ctx_size_draft",
    "spec_type", "spec_n_max", "spec_n_min", "spec_p_min", "spec_p_split", "spec_sampling", "spec_autotune",
    "spec_draft_sampling", "spec_draft_n_max", "spec_draft_n_min", "spec_draft_n_ubatch", "spec_draft_p_min",
    "spec_draft_p_split", "spec_draft_threads", "spec_draft_threads_batch", "spec_draft_backend_sampling",
    "spec_draft_cpu_moe", "spec_draft_n_cpu_moe", "spec_ngram_simple_size_m", "spec_ngram_simple_size_n",
    "spec_ngram_simple_min_hits", "spec_ngram_map_k_size_m", "spec_ngram_map_k_size_n", "spec_ngram_map_k_min_hits",
    "spec_ngram_map_k4v_size_m", "spec_ngram_map_k4v_size_n", "spec_ngram_map_k4v_min_hits",
    "spec_ngram_mod_n_match", "spec_ngram_mod_n_min", "spec_ngram_mod_n_max",
    "autofit", "autofitpadding", "noflashattention", "nommq", "noshift",
    "temp", "temperature", "top_k", "top_p", "min_p", "typical_p", "tfs_z",
    "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
    "mirostat", "mirostat_tau", "mirostat_eta", "seed", "n_predict", "max_tokens", "max_new",
    "jinja", "chat_template", "chat_template_kwargs", "reasoning_format", "think", "no_think", "reasoning_effort",
    "enforce_eager", "enable_prefix_caching", "swap_space",
})


def _safe_runtime_params(store: Any, engine: str) -> tuple[str | None, dict[str, dict]]:
    active = store.get_active_engine_version(engine)
    if not active or not active.get("version") or not active_engine_row_is_runnable(engine, active):
        return None, {}
    version = str(active["version"])
    entry = get_version_entry(store, engine, version)
    params = param_index_from_entry(entry)
    # Defaults are limited to ordinary runtime server flags. File selection,
    # build/install controls, reserved switches, and read-only scan rows are
    # intentionally excluded.
    params = {
        key: row
        for key, row in params.items()
        if row.get("scope") == "process"
        and row.get("transport") == "server_flag"
        and not row.get("reserved")
        and not row.get("read_only")
        and row.get("primary_flag")
        and (key in PORTABLE_RUNTIME_KEYS or (engine == "moe4all" and key == "set"))
    }
    return version, params


def _validate_value(engine: str, key: str, value: Any, row: dict) -> None:
    kind = row.get("value_kind")
    if key == "set":
        from backend.proxy.infr_launch import _SAFE_SET_VALUES

        seen = set()
        valid = engine == "moe4all" and isinstance(value, list) and all(isinstance(item, str) for item in value)
        if valid:
            for item in value:
                config_key, separator, config_value = item.partition("=")
                allowed = _SAFE_SET_VALUES.get(config_key, frozenset())
                if not separator or config_key not in _SAFE_SET_VALUES or config_value not in allowed or config_key in seen:
                    valid = False
                    break
                seen.add(config_key)
        if not valid:
            raise HTTPException(status_code=422, detail="MoE4All setup defaults contain an unsupported config value")
        return
    if kind == "flag":
        valid = isinstance(value, bool)
    elif kind == "repeatable":
        valid = isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value)
    elif kind in {"space_list", "csv_enum", "semicolon_enum"}:
        valid = isinstance(value, list)
    else:
        scalar_type = row.get("scalar_type") or row.get("type")
        if scalar_type == "int":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif scalar_type == "float":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        else:
            valid = isinstance(value, str)
    if not valid:
        raise HTTPException(status_code=422, detail=f"{key} has an invalid value type")
    options = [option.get("value") for option in row.get("options") or [] if isinstance(option, dict) and "value" in option]
    selected = value if isinstance(value, list) else [value]
    if options and any(item not in options for item in selected):
        raise HTTPException(status_code=422, detail=f"{key} contains an unsupported value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if row.get("minimum") is not None and value < row["minimum"]:
            raise HTTPException(status_code=422, detail=f"{key} is below the scanned minimum")
        if row.get("maximum") is not None and value > row["maximum"]:
            raise HTTPException(status_code=422, detail=f"{key} is above the scanned maximum")


def _stored_value_matches_scan(engine: str, key: str, value: Any, row: dict) -> bool:
    """Hide stale stored values when a new scan changes a parameter's contract."""
    try:
        _validate_value(engine, key, value, row)
    except HTTPException:
        return False
    return True


def _read_defaults(store: Any, engine: str) -> dict:
    with store.exclusive_documents():
        document = store._read_yaml(FILENAME)
        row = (document.get("engines") or {}).get(engine) or {}
        version, params = _safe_runtime_params(store, engine)
        # A newly installed/re-scanned engine can make a prior key unsafe or
        # unavailable. Preserve it in storage, but do not return it to clients.
        stored_values = row.get("values") or {}
        values = {
            key: value
            for key, value in stored_values.items()
            if key in params and _stored_value_matches_scan(engine, key, value, params[key])
        }
        unavailable_values = {key: value for key, value in stored_values.items() if key not in values}
        return {
            "values": values,
            "unavailable_values": unavailable_values,
            "engine": engine,
            "active_version": version,
            "catalog": {key: params[key] for key in sorted(params)},
        }

@router.get("/{engine}")
async def get_engine_setup_defaults(engine: str):
    if engine not in VALID_ENGINE_IDS:
        raise HTTPException(status_code=404, detail="Unknown engine")
    return await run_blocking(_read_defaults, get_store(), engine)
