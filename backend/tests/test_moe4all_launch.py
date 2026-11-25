from __future__ import annotations

import tomllib

import pytest

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.scan.help_parsers import parse_infr_serve_help_to_sections
from backend.proxy import infr_launch, launch_spec, profile_launch
from backend.proxy.launch_spec import format_argv
from backend.proxy.manifests import _write_artifact
from backend.proxy.launcher import _substitute_argv
from backend.proxy.studio.registry import _resolve_arg


HELP = """Usage: infr serve [OPTIONS] <MODEL>

Options:
  --addr <ADDR>              Bind address
  --config <PATH>            Config file
  -n, --parallel <N>         Concurrent generation slots
  --set <PATH=VALUE>         Set a configuration field; repeatable
  --dev <DEV>                Vulkan device or cpu
  --ctx <TOKENS>             Context window
  -u, --ubatch <N>           Physical batch
  -t, --threads <N>          CPU threads
  --temp <T>                 Sampling temperature
  --top-k <K>                Top-k
  --top-p <P>                Top-p
  --seed <N>                 RNG seed
  --max-new <N>              Max generated tokens
  --no-think                 Force reasoning off
  --think                    Force reasoning on
  --reasoning-effort <LEVEL> Reasoning level
  --embedding-model <MODEL>  Embedding model
  -h, --help                 Print help
"""


def _param_index():
    from backend.engines.params import param_index_from_entry

    return param_index_from_entry(
        {"sections": parse_infr_serve_help_to_sections(HELP, "moe4all")}
    )


def test_moe4all_definition_declares_distinct_runtime_and_launch_family():
    definition = load_default_engine_definitions().get("moe4all")
    assert definition is not None
    assert definition.runtime_kind == "moe4all"
    assert definition.scanner_kind == "infr_help"
    assert definition.profile("")["launch_family"] == "infr_gguf_server"


def test_infr_help_parser_exposes_supported_generation_controls_only():
    sections = parse_infr_serve_help_to_sections(HELP, "moe4all")
    params = {param["key"]: param for section in sections for param in section["params"]}

    assert {"ctx", "parallel", "dev", "ubatch", "threads", "temp", "top_k", "top_p", "max_new", "no_think", "think", "reasoning_effort", "set"} <= set(params)
    assert params["config"]["reserved"] is True
    assert params["addr"]["reserved"] is True
    assert params["embedding_model"]["reserved"] is True
    assert params["set"]["multiple"] is True


def test_infr_launch_uses_loopback_and_native_infr_argv(tmp_path, monkeypatch):
    binary = tmp_path / "infr"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    model_path = tmp_path / "Qwen-local-Q4_K_M.gguf"
    model_path.write_bytes(b"GGUF fixture")
    active = {"version": "1.3.0", "binary_path": str(binary), "build_status": "ready"}

    monkeypatch.setattr(launch_spec, "_active_row", lambda engine: active)
    monkeypatch.setattr(
        "backend.proxy.commands._active_engine_param_index", lambda engine: _param_index()
    )

    definition = profile_launch.profile_launch_definition("moe4all")
    assert definition is not None
    compiled = infr_launch.compile_infr_gguf_launch(
        definition=definition,
        spec=definition.profile(""),
        model={"artifact": {"format": "gguf", "path": str(model_path)}},
        config={"ctx": "4096", "parallel": 1, "dev": "Vulkan0", "set": ["kv.dynamic=true", "kv.type_k=q8_0", "kernels.vulkan.flash_splits=2"]},
        stable_id="model-stable-id",
        engine="moe4all",
    )

    assert compiled.executable == str(binary)
    assert compiled.argv == [
        "serve",
        str(model_path),
        "--config",
        {"artifact": "infr.toml"},
        "--addr",
        {"runtime": "loopback_addr"},
        "--ctx",
        "4096",
        "--dev",
        "Vulkan0",
        "--parallel",
        "1",
        "--set",
        "kv.dynamic=true",
        "--set",
        "kv.type_k=q8_0",
        "--set",
        "kernels.vulkan.flash_splits=2",
    ]
    assert compiled.file_identities["model"]["present"] is True
    assert compiled.artifacts["infr.toml"]["__studio_artifact_format"] == "raw-text"


def test_infr_launch_rejects_address_override_and_invalid_model(tmp_path, monkeypatch):
    binary = tmp_path / "infr"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF fixture")
    active = {"version": "1.3.0", "binary_path": str(binary)}
    monkeypatch.setattr(launch_spec, "_active_row", lambda engine: active)
    monkeypatch.setattr(
        "backend.proxy.commands._active_engine_param_index", lambda engine: _param_index()
    )
    definition = profile_launch.profile_launch_definition("moe4all")

    with pytest.raises(launch_spec.LaunchCompileError, match="Studio-owned"):
        infr_launch.compile_infr_gguf_launch(
            definition=definition,
            spec=definition.profile(""),
            model={"artifact": {"path": str(model_path)}},
            config={"custom_args": "--addr 0.0.0.0:9000"},
            stable_id="m",
            engine="moe4all",
        )

    with pytest.raises(launch_spec.LaunchCompileError, match="single-file GGUF"):
        infr_launch.compile_infr_gguf_launch(
            definition=definition,
            spec=definition.profile(""),
            model={"artifact": {"path": str(tmp_path / "missing.gguf")}},
            config={},
            stable_id="m",
            engine="moe4all",
        )

    with pytest.raises(launch_spec.LaunchCompileError, match="Studio-owned"):
        infr_launch.compile_infr_gguf_launch(
            definition=definition,
            spec=definition.profile(""),
            model={"artifact": {"path": str(model_path)}},
            config={"custom_args": "--config /tmp/user-infr.toml"},
            stable_id="m",
            engine="moe4all",
        )


def test_infr_config_is_generated_raw_toml_and_ignores_ambient_config(tmp_path, monkeypatch):
    binary = tmp_path / "infr"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF fixture")
    ambient_cwd = tmp_path / "ambient"
    ambient_cwd.mkdir()
    ambient = ambient_cwd / "infr.toml"
    ambient.write_text('[serve]\napi_key = "ambient-secret"\n')
    active = {"version": "1.3.0", "binary_path": str(binary)}
    monkeypatch.setattr(launch_spec, "_active_row", lambda engine: active)
    monkeypatch.setattr(
        "backend.proxy.commands._active_engine_param_index", lambda engine: _param_index()
    )
    definition = profile_launch.profile_launch_definition("moe4all")
    compiled = infr_launch.compile_infr_gguf_launch(
        definition=definition,
        spec=definition.profile(""),
        model={"artifact": {"path": str(model_path)}},
        config={"set": ["kv.dynamic=true", "kv.type_k=q8_0", "kv.type_v=f16"]},
        stable_id="m",
        engine="moe4all",
    )

    artifact_dir = tmp_path / "generation" / "artifacts"
    artifact_dir.mkdir(parents=True)
    _write_artifact(str(artifact_dir), "infr.toml", compiled.artifacts["infr.toml"])
    config_path = artifact_dir / "infr.toml"
    assert tomllib.loads(config_path.read_text()) == {}
    assert "ambient-secret" not in config_path.read_text()
    assert str(ambient) not in format_argv(compiled.executable, compiled.argv)
    assert "ambient-secret" not in format_argv(compiled.executable, compiled.argv)
    assert compiled.argv[compiled.argv.index("--config") + 1] == {"artifact": "infr.toml"}

    launcher_argv = _substitute_argv(
        compiled.argv,
        port="4321",
        generation=str(tmp_path / "generation"),
        root=str(tmp_path),
    )
    control_argv = [str(item) if not isinstance(item, dict) else _resolve_arg(item, 4321, str(artifact_dir)) for item in compiled.argv]
    assert launcher_argv[launcher_argv.index("--config") + 1] == str(config_path)
    assert control_argv[control_argv.index("--config") + 1] == str(config_path)


def test_infr_set_blocks_auth_and_non_kv_paths_without_echoing_values(tmp_path, monkeypatch):
    binary = tmp_path / "infr"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF fixture")
    monkeypatch.setattr(launch_spec, "_active_row", lambda engine: {"binary_path": str(binary)})
    monkeypatch.setattr(
        "backend.proxy.commands._active_engine_param_index", lambda engine: _param_index()
    )
    definition = profile_launch.profile_launch_definition("moe4all")

    for setting in (
        "serve.api_key=hidden-value",
        "paging.cache=8g",
        "kv.type_k=hidden-value",
        "kv.type_k=q8_0;serve.api_key=hidden-value",
        "kernels.vulkan.flash_splits=3",
        "kernels.vulkan.flash_splits=2;serve.api_key=hidden-value",
    ):
        with pytest.raises(launch_spec.LaunchCompileError, match="approved KV settings") as error:
            infr_launch.compile_infr_gguf_launch(
                definition=definition,
                spec=definition.profile(""),
                model={"artifact": {"path": str(model_path)}},
                config={"set": [setting]},
                stable_id="m",
                engine="moe4all",
            )
        assert "hidden-value" not in str(error.value)


def test_infr_launch_strips_ambient_and_custom_infr_environment(tmp_path, monkeypatch):
    binary = tmp_path / "infr"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF fixture")
    monkeypatch.setattr(launch_spec, "_active_row", lambda engine: {"binary_path": str(binary)})
    monkeypatch.setattr(
        "backend.proxy.commands._active_engine_param_index", lambda engine: _param_index()
    )
    monkeypatch.setenv("INFR_SERVE_API_KEY", "ambient-secret")
    definition = profile_launch.profile_launch_definition("moe4all")
    compiled = infr_launch.compile_infr_gguf_launch(
        definition=definition,
        spec=definition.profile(""),
        model={"artifact": {"path": str(model_path)}},
        config={"runtime_env": {"INFR_DEV": "Vulkan1"}},
        stable_id="m",
        engine="moe4all",
    )

    assert "ambient-secret" not in repr(compiled.env.set)
    assert "INFR_SERVE_API_KEY" in compiled.env.unset
    assert "INFR_DEV" in compiled.env.unset
    assert "INFR_DEV" not in compiled.env.set


def test_loopback_address_placeholder_resolves_in_both_launchers():
    placeholder = {"runtime": "loopback_addr"}
    assert _substitute_argv(["--addr", placeholder], port="4321", generation="/g", root="/r") == [
        "--addr",
        "127.0.0.1:4321",
    ]
    assert _resolve_arg(placeholder, 4321, None) == "127.0.0.1:4321"


def test_upstream_model_name_uses_gguf_basename():
    definition = profile_launch.profile_launch_definition("moe4all")
    assert infr_launch.upstream_model_name(
        definition,
        {"artifact": {"path": "/models/Qwen3.6-local-Q4_K_M.gguf"}},
        "stable-id",
    ) == "Qwen3.6-local-Q4_K_M"


def test_upstream_model_name_keeps_snapshot_symlink_basename(tmp_path):
    definition = profile_launch.profile_launch_definition("moe4all")
    blob = tmp_path / "a1b2c3d4e5.blob"
    blob.write_bytes(b"GGUF fixture")
    snapshot = tmp_path / "Qwen3.6-14B-A3B-vibetuned-Q4_K_M.gguf"
    snapshot.symlink_to(blob)

    assert infr_launch.upstream_model_name(
        definition,
        {"local_artifacts": {"weight": str(snapshot)}},
        "stable-id",
    ) == "Qwen3.6-14B-A3B-vibetuned-Q4_K_M"
