from __future__ import annotations

import pytest

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.scan import koboldcpp_help
from backend.proxy import koboldcpp_launch, launch_spec, profile_launch


def _param_index():
    from backend.engines.params import param_index_from_entry

    return param_index_from_entry(
        {
            "sections": [
                {
                    "id": "server",
                    "label": "Server",
                    "params": [
                        {
                            "key": "contextsize",
                            "label": "Context size",
                            "description": "Context window",
                            "flags": ["--contextsize"],
                            "primary_flag": "--contextsize",
                            "value_kind": "scalar",
                            "scalar_type": "int",
                            "multiple": False,
                            "options": None,
                            "default": None,
                            "reserved": False,
                        }
                    ],
                }
            ]
        }
    )


def _runtime(tmp_path, monkeypatch):
    binary = tmp_path / "koboldcpp-test"
    binary.write_bytes(b"\x7fELF kobold fixture")
    binary.chmod(0o755)
    model = tmp_path / "tiny-model.gguf"
    model.write_bytes(b"GGUF fixture")
    monkeypatch.setattr(
        launch_spec,
        "_active_row",
        lambda _engine: {"version": "test", "binary_path": str(binary), "build_status": "ready"},
    )
    monkeypatch.setattr(
        "backend.proxy.commands._active_engine_param_index",
        lambda _engine: _param_index(),
    )
    definition = profile_launch.profile_launch_definition("koboldcpp")
    assert definition is not None
    return binary, model, definition


def test_koboldcpp_definition_uses_distinct_external_runtime():
    definition = load_default_engine_definitions().get("koboldcpp")
    assert definition is not None
    assert definition.runtime_kind == "koboldcpp"
    assert definition.install_kind == "prebuilt"
    assert definition.profile("")["launch_family"] == "koboldcpp_gguf_server"


def test_kobold_help_parser_filters_unsupported_application_controls(monkeypatch):
    monkeypatch.setattr(
        koboldcpp_help,
        "parse_llama_help_to_sections",
        lambda _text, _engine: [
            {
                "id": "all",
                "label": "All",
                "params": [
                    {"key": "contextsize", "primary_flag": "--contextsize"},
                    {"key": "config", "primary_flag": "--config"},
                    {"key": "remote", "primary_flag": "--remotetunnel"},
                    {"key": "host", "primary_flag": "--host"},
                ],
            }
        ],
    )
    sections = koboldcpp_help.parse_koboldcpp_help_to_sections("help", "koboldcpp")
    assert [p["primary_flag"] for p in sections[0]["params"]] == ["--contextsize"]


def test_kobold_launch_uses_local_gguf_loopback_and_managed_port(tmp_path, monkeypatch):
    binary, model, definition = _runtime(tmp_path, monkeypatch)
    compiled = koboldcpp_launch.compile_koboldcpp_gguf_launch(
        definition=definition,
        spec=definition.profile(""),
        model={"artifact": {"format": "gguf", "path": str(model)}},
        config={"contextsize": 4096},
        stable_id="model-stable-id",
        engine="koboldcpp",
    )
    assert compiled.executable == str(binary)
    assert compiled.argv == [
        "--skiplauncher",
        "--host",
        "127.0.0.1",
        "--model",
        str(model),
        "--port",
        {"runtime": "port"},
        "--contextsize",
        "4096",
    ]
    assert compiled.file_identities["model"]["present"] is True
    assert compiled.env.set.get("LD_LIBRARY_PATH") is None


def test_kobold_launch_rejects_unbounded_custom_flags_and_non_gguf(tmp_path, monkeypatch):
    _binary, model, definition = _runtime(tmp_path, monkeypatch)
    kwargs = {
        "definition": definition,
        "spec": definition.profile(""),
        "model": {"artifact": {"format": "gguf", "path": str(model)}},
        "stable_id": "model",
        "engine": "koboldcpp",
    }
    with pytest.raises(launch_spec.LaunchCompileError, match="custom arguments are disabled"):
        koboldcpp_launch.compile_koboldcpp_gguf_launch(**kwargs, config={"custom_args": "--remotetunnel"})

    with pytest.raises(launch_spec.LaunchCompileError, match="explicitly registered GGUF"):
        koboldcpp_launch.compile_koboldcpp_gguf_launch(
            **{**kwargs, "model": {"artifact": {"format": "safetensors", "path": str(model)}}},
            config={},
        )
