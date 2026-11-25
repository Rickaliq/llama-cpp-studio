"""Constrained local-GGUF launch support for externally installed KoboldCpp."""

from __future__ import annotations

import os
from typing import Any, List, Mapping


LAUNCH_FAMILY = "koboldcpp_gguf_server"
_SUPPORTED_CONFIG_FLAGS = frozenset(
    {
        "--autofit",
        "--autofitpadding",
        "--batchsize",
        "--blasthreads",
        "--contextsize",
        "--defaultgenamt",
        "--n-gpu-layers",
        "--n-gpu-layers-draft",
        "--main-gpu",
        "--nommq",
        "--noflashattention",
        "--noshift",
        "--noswa",
        "--parallelrequests",
        "--quantkv",
        "--threads",
        "--tensor_split",
        "--ubatchsize",
        "--usecpu",
        "--usemmap",
        "--usemlock",
        "--usevulkan",
    }
)


def compile_koboldcpp_gguf_launch(
    *,
    definition: Any,
    spec: Mapping[str, Any],
    model: Any,
    config: Mapping[str, Any],
    stable_id: str,
    engine: str,
):
    """Compile a local single-file GGUF launch; never resolve/download HF data."""
    from backend.proxy.commands import _active_engine_param_index, _emit_structured_tokens, _model_attr
    from backend.proxy.launch_spec import (
        PORT_PLACEHOLDER,
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
            "No runnable KoboldCpp executable is registered", field="executable"
        )

    artifact = _model_attr(model, "artifact")
    artifact = artifact if isinstance(artifact, Mapping) else {}
    artifact_format = str(
        artifact.get("format") or _model_attr(model, "format") or ""
    ).lower()
    if artifact_format and artifact_format != "gguf":
        raise LaunchCompileError(
            "KoboldCpp requires an explicitly registered GGUF model",
            field="model",
        )
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
    if (
        not model_path
        or not model_path.lower().endswith(".gguf")
        or not os.path.isfile(model_path)
    ):
        raise LaunchCompileError(
            "KoboldCpp requires an existing local single-file GGUF model",
            field="model",
        )

    if str(config.get("custom_args") or "").strip():
        raise LaunchCompileError(
            "KoboldCpp custom arguments are disabled; use a supported scanned setting",
            field="custom_args",
        )
    try:
        structured = _emit_structured_tokens(
            dict(config),
            engine=engine,
            param_index=_active_engine_param_index(engine),
        )
    except ValueError as exc:
        raise LaunchCompileError(str(exc), field="config") from exc

    for token in structured:
        if token.startswith("-") and token.split("=", 1)[0] not in _SUPPORTED_CONFIG_FLAGS:
            raise LaunchCompileError(
                f"KoboldCpp setting {token!r} is outside Studio's supported local-server controls",
                field="config",
            )

    prefix = [str(token) for token in (spec.get("argv_prefix") or ())]
    port_flag = str((spec.get("port_binding") or {}).get("flag") or "--port")
    owned = set(profile_owned_flags(definition)) | {"--model", "--host", port_flag}
    owned |= {token.split("=", 1)[0] for token in prefix if token.startswith("-")}
    try:
        _reject_owned_flags(structured, owned=owned)
    except ValueError as exc:
        raise LaunchCompileError(str(exc), field="custom_args") from exc

    argv: List[Any] = list(prefix)
    argv.extend(["--model", model_path, port_flag, dict(PORT_PLACEHOLDER)])
    argv.extend(structured)
    identities = {
        "executable": _file_identity(binary),
        "model": _file_identity(model_path),
    }
    env = _compile_env(
        dict(config), engine=engine, required_ld=[], required_path=[],
        engine_defaults={}, active=active,
    )
    return LaunchSpec(
        model_id=stable_id,
        engine_id=engine,
        engine_install_id=_install_identity(active, binary),
        executable=binary,
        argv=argv,
        cwd=None,
        env=env,
        file_identities=identities,
    )


def launch_family_supported(family: str | None) -> bool:
    return str(family or "") == LAUNCH_FAMILY


__all__ = ["LAUNCH_FAMILY", "compile_koboldcpp_gguf_launch", "launch_family_supported"]
