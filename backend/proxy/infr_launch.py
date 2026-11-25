"""Studio launch adapter for MoE4All's local GGUF `infr serve` runtime."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping


LAUNCH_FAMILY = "infr_gguf_server"
CONFIG_ARTIFACT = "infr.toml"
EMPTY_CONFIG_ARTIFACT = {
    "__studio_artifact_format": "raw-text",
    "content": "# Empty Studio-owned config; prevents ambient infr.toml discovery.\n",
}

_KV_FORMATS = frozenset({
    "f16", "f32", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1",
    "iq4_nl", "turbo2", "turbo3", "turbo4",
})
_SAFE_SET_VALUES = {
    "kv.dynamic": frozenset({"true", "false"}),
    "kv.type_k": _KV_FORMATS,
    "kv.type_v": _KV_FORMATS,
    # Explicit Vulkan A/B overrides documented by MoE4All. Do not imply that
    # any split count is a cross-model default; the engine's heuristic remains
    # the default when this path is omitted.
    "kernels.vulkan.flash_splits": frozenset({"1", "2", "4"}),
}


def _validate_set_tokens(tokens: List[str]) -> None:
    """Limit arbitrary config-path flags to reviewed, non-secret controls."""
    index = 0
    seen_keys = set()
    while index < len(tokens):
        token = tokens[index]
        if token == "--set":
            index += 1
            if index >= len(tokens):
                raise ValueError
            setting = tokens[index]
        elif token.startswith("--set="):
            setting = token[len("--set=") :]
        else:
            index += 1
            continue
        key, separator, value = setting.partition("=")
        if (
            not separator
            or key in seen_keys
            or value not in _SAFE_SET_VALUES.get(key, frozenset())
        ):
            raise ValueError
        seen_keys.add(key)
        index += 1


def compile_infr_gguf_launch(
    *,
    definition: Any,
    spec: Mapping[str, Any],
    model: Any,
    config: Mapping[str, Any],
    stable_id: str,
    engine: str,
):
    """Compile a local GGUF launch using MoE4All's distinct CLI contract.

    `infr serve` accepts a positional GGUF path and binds with
    `--addr 127.0.0.1:<port>`; it does not accept llama-server's `--model`,
    `--alias`, or bare `--port` conventions.
    """
    from backend.proxy.commands import (
        _active_engine_param_index,
        _emit_structured_tokens,
        _model_attr,
    )
    from backend.proxy.launch_spec import (
        _OWNED_PORT_FLAGS,
        LOOPBACK_ADDR_PLACEHOLDER,
        LaunchCompileError,
        LaunchSpec,
        _active_row,
        _compile_env,
        _file_identity,
        _install_identity,
        _reject_owned_flags,
    )
    from backend.proxy.profile_launch import profile_owned_flags

    active = _active_row(engine)
    binary = os.path.realpath(str((active or {}).get("binary_path") or ""))
    if not binary or not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        raise LaunchCompileError(
            "No runnable MoE4All `infr` binary is registered",
            field="executable",
        )

    artifact = _model_attr(model, "artifact")
    if not isinstance(artifact, Mapping):
        artifact = {}
    local_artifacts = _model_attr(model, "local_artifacts")
    local_artifacts = local_artifacts if isinstance(local_artifacts, Mapping) else {}
    model_path = str(
        local_artifacts.get("weight")
        or artifact.get("path")
        or _model_attr(model, "local_path")
        or _model_attr(model, "path")
        or ""
    )
    if not model_path:
        files = _model_attr(model, "files")
        if isinstance(files, list):
            model_path = next(
                (
                    str(item.get("path") or item.get("local_path") or "")
                    for item in files
                    if isinstance(item, Mapping)
                    and str(item.get("role") or "weight") in {"weight", "shard"}
                    and str(item.get("format") or "gguf").lower() == "gguf"
                ),
                "",
            )
    if not model_path:
        # Reuse Studio's local-only GGUF cache resolver. Deliberately ignore
        # its Hugging Face repo fallback: launching an uncached model here
        # could download weights as a side effect.
        from backend.proxy.commands import _resolve_llama_model_source

        cached_path, _hf_repo, _hf_id = _resolve_llama_model_source(model)
        model_path = str(cached_path or "")
    if not model_path or not os.path.isfile(model_path) or not model_path.lower().endswith(".gguf"):
        raise LaunchCompileError(
            "MoE4All requires an existing single-file GGUF model",
            field="model",
        )
    # Preserve the registered path spelling: MoE4All derives the OpenAI model
    # id from the supplied GGUF basename, and Hugging Face snapshot paths may
    # be symlinks to hash-named blobs. `_file_identity` records the real target
    # separately.

    params = _active_engine_param_index(engine)
    structured = _emit_structured_tokens(
        dict(config),
        engine=engine,
        param_index=params,
    )
    try:
        _validate_set_tokens(structured)
    except ValueError as exc:
        raise LaunchCompileError(
            "MoE4All --set accepts approved KV settings and documented Vulkan flash split candidates (1, 2, 4)"
            " — K/V formats: "
            + ", ".join(sorted(_KV_FORMATS)),
            field="set",
        ) from exc
    port_flag = str((spec.get("port_binding") or {}).get("flag") or "--addr")
    owned = set(profile_owned_flags(definition)) | set(_OWNED_PORT_FLAGS) | {"--config"}
    _reject_owned_flags(structured, owned=owned)

    # The model id returned by MoE4All is the GGUF basename without its suffix;
    # matching that keeps Studio's stable id internal to its own proxy.
    argv: List[Any] = [
        "serve",
        model_path,
        "--config",
        {"artifact": CONFIG_ARTIFACT},
        port_flag,
        dict(LOOPBACK_ADDR_PLACEHOLDER),
    ]
    argv.extend(structured)
    env = _compile_env(
        config,
        engine=engine,
        required_ld=[],
        required_path=[],
        engine_defaults={},
        active=active,
    )
    # MoE4All reads engine settings from INFR_* environment variables before
    # command-line flags. Do not let Studio's ambient service environment or
    # free-form runtime_env config inject settings or API credentials.
    ambient_infr = {name for name in os.environ if name.startswith("INFR_")}
    configured_infr = {name for name in env.set if name.startswith("INFR_")}
    env.unset = sorted(set(env.unset) | ambient_infr | configured_infr)
    env.set = {name: value for name, value in env.set.items() if not name.startswith("INFR_")}

    return LaunchSpec(
        model_id=stable_id,
        engine_id=engine,
        engine_install_id=_install_identity(active, binary),
        executable=binary,
        argv=argv,
        cwd=None,
        env=env,
        file_identities={
            "executable": _file_identity(binary),
            "model": _file_identity(model_path),
        },
        artifacts={CONFIG_ARTIFACT: dict(EMPTY_CONFIG_ARTIFACT)},
    )


def upstream_model_name(definition: Any, model: Any, stable_id: str) -> str:
    """Return the API model name required by the declared family profile."""
    if str(definition.profile("").get("upstream_model_name") or "stable_id") != "model_stem":
        return stable_id
    artifact = model.get("artifact") if isinstance(model, Mapping) else None
    path = ""
    if isinstance(artifact, Mapping):
        path = str(artifact.get("path") or "")
    if not path and isinstance(model, Mapping):
        local_artifacts = model.get("local_artifacts")
        local_artifacts = local_artifacts if isinstance(local_artifacts, Mapping) else {}
        path = str(local_artifacts.get("weight") or model.get("local_path") or model.get("path") or "")
    if not path:
        from backend.proxy.commands import _resolve_llama_model_source

        cached_path, _hf_repo, _hf_id = _resolve_llama_model_source(model)
        path = str(cached_path or "")
    if not path:
        raise ValueError("MoE4All model path is required to resolve its API model id")
    return os.path.splitext(os.path.basename(path))[0]
