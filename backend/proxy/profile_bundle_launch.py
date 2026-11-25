"""Contained launch adapter for explicit local Strata profile bundles.

The profile file is the model record's ``local_path`` (format
``strata_bundle``). Studio reads it without modifying it, validates its exact
schema, then writes a sanitized private server config into the launch manifest.
This adapter intentionally supports the reviewed local profile shape only; it
does not launch Strata's worker directly or accept arbitrary server options.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


SERVER_CONFIG = "strata-server.json"
LAUNCH_FAMILY = "profile_bundle_http_server"
_PROFILE_KEYS = {
    "exe", "args", "cwd", "tokenizer", "model_name", "log", "lib_dirs",
    "port", "host", "parallel", "max_queue", "offline", "backend", "env",
}
_PATH_FLAGS = {"--pack", "--native", "--ple-gguf", "--expert-profile"}
_VALUE_FLAGS = {
    "--expert-cache", "--prefill", "--spec", "--max-context", "--kv",
    "--vram-reserve-mib", "--gpu", "--memory-limit-mib",
    "--conversation-cache-mib", "--prompt-cache",
}
_BOOL_FLAGS = {"--mmap-experts"}
_ALLOWED_KV = {"fp16", "int8", "q4_0", "k8v4"}


def _fail(message: str, field: str = "model"):
    from backend.proxy.launch_spec import LaunchCompileError

    raise LaunchCompileError(message, field=field)


def _profile_path(model: Any) -> Path:
    from backend.proxy.commands import _model_attr

    artifact = _model_attr(model, "artifact")
    artifact = artifact if isinstance(artifact, Mapping) else {}
    artifact_format = str(artifact.get("format") or _model_attr(model, "format") or "").lower()
    if artifact_format != "strata_bundle":
        _fail("Strata models must be explicitly registered as strata_bundle", "model")
    profile = (
        _model_attr(model, "local_path")
        or _model_attr(model, "path")
        or artifact.get("path")
    )
    if not isinstance(profile, str) or not profile:
        _fail("Strata models require an explicit local profile bundle", "model")
    path = Path(profile).expanduser().resolve()
    if path.suffix.lower() != ".json" or not path.is_file():
        _fail("Strata profile bundle must be an existing JSON file", "model")
    return path


def _read_profile(model: Any) -> tuple[Path, dict[str, Any]]:
    path = _profile_path(model)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        _fail(f"Cannot read Strata profile bundle: {exc}", "model")
    if not isinstance(profile, dict):
        _fail("Strata profile must contain a JSON object", "model")
    unknown = set(profile) - _PROFILE_KEYS
    if unknown:
        _fail(f"Unsupported Strata profile fields: {', '.join(sorted(unknown))}", "model")
    # Credentials must never enter manifests, previews, or argv. The only
    # supported profile is unauthenticated and loopback-only.
    if "api_key" in profile or any("key" in str(key).lower() for key in profile):
        _fail("Strata profile credentials are not supported by Studio", "model")
    return path, profile


def _abs_file(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"Strata {label} path is required", "model")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        _fail(f"Strata {label} file does not exist", "model")
    return str(path)


def _abs_dir(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"Strata {label} directory is required", "model")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        _fail(f"Strata {label} directory does not exist", "model")
    return str(path)


def _sanitize_worker_args(raw: Any) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        _fail("Strata profile args must be a string list", "model")
    args: list[str] = []
    index = 0
    seen: set[str] = set()
    while index < len(raw):
        flag = raw[index]
        index += 1
        if flag in _BOOL_FLAGS:
            if flag in seen:
                _fail(f"Duplicate Strata option {flag}", "model")
            seen.add(flag)
            args.append(flag)
            continue
        if flag not in _PATH_FLAGS | _VALUE_FLAGS or flag in seen or index >= len(raw):
            _fail(f"Unsupported Strata worker option {flag!r}", "model")
        seen.add(flag)
        value = raw[index]
        index += 1
        if flag in _PATH_FLAGS:
            value = _abs_dir(value, flag) if flag == "--pack" else _abs_file(value, flag)
        elif flag == "--kv" and value not in _ALLOWED_KV:
            _fail("Unsupported Strata KV format", "model")
        elif flag in {"--expert-cache", "--prefill", "--spec", "--max-context", "--vram-reserve-mib", "--memory-limit-mib", "--conversation-cache-mib", "--prompt-cache", "--gpu"}:
            if not value.isdecimal() and not (flag == "--expert-cache" and value == "auto"):
                _fail(f"Strata option {flag} must be an integer", "model")
        args.extend((flag, value))
    required = _PATH_FLAGS | {"--expert-cache", "--max-context", "--kv", "--vram-reserve-mib", "--mmap-experts"}
    missing = required - seen
    if missing:
        _fail(f"Strata profile is missing required options: {', '.join(sorted(missing))}", "model")
    # `auto` cache sizing caused the prior profile to exhaust the desktop GPU.
    # Studio's adapter caps it at the host-qualified 5000 slots and leaves at
    # least 3 GiB free; a smaller explicit user value is preserved.
    cache_index = args.index("--expert-cache") + 1
    if args[cache_index] == "auto":
        args[cache_index] = "5000"
    elif args[cache_index].isdecimal():
        args[cache_index] = str(min(5000, int(args[cache_index])))
    else:
        _fail("Strata expert-cache must be a slot count or auto", "model")
    reserve_index = args.index("--vram-reserve-mib") + 1
    args[reserve_index] = str(max(3072, int(args[reserve_index])))
    return args


def _validated_profile(profile: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    cwd = _abs_dir(profile.get("cwd"), "working")
    executable = _abs_file(profile.get("exe"), "worker executable")
    tokenizer = _abs_dir(profile.get("tokenizer"), "tokenizer")
    model_name = profile.get("model_name")
    if not isinstance(model_name, str) or not model_name or len(model_name) > 200:
        _fail("Strata profile requires a valid model_name", "model")
    if profile.get("host", "127.0.0.1") not in {"127.0.0.1", "localhost"}:
        _fail("Strata Studio profiles must bind to loopback", "model")
    if profile.get("offline") is not True:
        _fail("Strata Studio profiles must enable offline mode", "model")
    if profile.get("backend") != "hip":
        _fail("This Strata profile adapter currently supports the HIP runtime only", "model")
    if profile.get("parallel", 1) != 1 or profile.get("max_queue", 0) != 0:
        _fail("Strata Studio profiles currently require parallel=1 and max_queue=0", "model")
    lib_dirs = profile.get("lib_dirs")
    if not isinstance(lib_dirs, list) or not lib_dirs:
        _fail("Strata profile requires explicit lib_dirs", "model")
    lib_dirs = [_abs_dir(item, "library") for item in lib_dirs]
    env = profile.get("env") or {}
    if not isinstance(env, dict) or set(env) - {"STRATA_ROCBLAS_TUNING"}:
        _fail("Strata profile has unsupported environment settings", "model")
    safe_env = {}
    if "STRATA_ROCBLAS_TUNING" in env:
        safe_env["STRATA_ROCBLAS_TUNING"] = _abs_dir(env["STRATA_ROCBLAS_TUNING"], "ROCm tuning")
    args = _sanitize_worker_args(profile.get("args"))
    server_module = Path(cwd) / "serve" / "server.py"
    if not server_module.is_file():
        _fail("Strata HTTP server module is missing", "executable")
    generated = {
        "exe": executable,
        "args": args,
        "cwd": cwd,
        "tokenizer": tokenizer,
        "model_name": model_name,
        "lib_dirs": lib_dirs,
        "host": "127.0.0.1",
        "port": 0,
        "parallel": 1,
        "max_queue": 0,
        "offline": True,
        "backend": "hip",
        "env": safe_env,
    }
    return generated, model_name


def upstream_model_name(model: Any, stable_id: str) -> str:
    _path, profile = _read_profile(model)
    name = profile.get("model_name")
    return str(name) if isinstance(name, str) and name else stable_id


def compile_profile_bundle_launch(*, definition: Any, spec: Mapping[str, Any], model: Any,
                                 config: Mapping[str, Any], stable_id: str, engine: str):
    from backend.proxy.commands import _model_attr
    from backend.proxy.launch_spec import (
        PORT_PLACEHOLDER, LaunchCompileError, LaunchSpec, _active_row,
        _compile_env, _file_identity, _install_identity,
    )

    profile_path, profile = _read_profile(model)
    generated, _model_name = _validated_profile(profile)
    runtime_env = config.get("runtime_env") or {}
    runtime_env_unset = config.get("runtime_env_unset") or []
    if not isinstance(runtime_env, Mapping) or not isinstance(runtime_env_unset, list):
        raise LaunchCompileError("Invalid Strata runtime environment settings", field="runtime_env")
    if any(str(name).startswith("STRATA_") for name in runtime_env) or any(
        str(name).startswith("STRATA_") for name in runtime_env_unset
    ):
        raise LaunchCompileError(
            "Strata environment controls must come from the validated local profile",
            field="runtime_env",
        )
    active = _active_row(engine)
    python = str((active or {}).get("binary_path") or "")
    if not python or not os.path.isfile(python) or not os.access(python, os.X_OK):
        raise LaunchCompileError("No runnable Strata server Python is registered", field="executable")
    argv = ["-m", "serve.server", "--engine", "strata", "--config",
            {"artifact": SERVER_CONFIG}, "--host", "127.0.0.1", "--port",
            PORT_PLACEHOLDER, "--offline"]
    required_ld = list(generated["lib_dirs"])
    env = _compile_env(
        config, engine=engine, required_ld=required_ld, required_path=[],
        engine_defaults=generated["env"], active=active,
    )
    # Do not inherit an API key into a no-auth local server, or allow ambient
    # Strata tuning to override this reviewed profile.
    env.unset = sorted(
        (set(env.unset) | {"STRATA_API_KEY"} | {key for key in os.environ if key.startswith("STRATA_")})
        - set(env.set)
    )
    return LaunchSpec(
        model_id=stable_id,
        engine_id=engine,
        engine_install_id=_install_identity(active, python),
        executable=python,
        argv=argv,
        cwd=generated["cwd"],
        env=env,
        artifacts={SERVER_CONFIG: generated},
        file_identities={
            "profile": _file_identity(str(profile_path)),
            "executable": _file_identity(python),
            "worker": _file_identity(generated["exe"]),
        },
        health_requires_loaded=True,
    )
