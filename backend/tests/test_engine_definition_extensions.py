from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from backend.engines.definitions import DefinitionError
from backend.engines.definitions import loader


def _plugin_definition(tmp_path: Path, engine_id: str = "local_fixture") -> Path:
    source = Path(loader.DEFINITIONS_DIR) / "llama_cpp.yaml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["id"] = engine_id
    document["label"] = "Local fixture"
    target = tmp_path / "config" / "engine-definitions" / f"{engine_id}.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return target


def test_default_loader_adds_user_definition_from_studio_data_dir(tmp_path, monkeypatch):
    _plugin_definition(tmp_path)
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("STUDIO_ENGINE_DEFINITIONS_DIR", raising=False)
    monkeypatch.setattr(loader, "_default_registry", None)

    definitions = loader.load_default_engine_definitions()

    assert "llama_cpp" in definitions.ids()
    assert "local_fixture" in definitions.ids()


def test_external_directory_override_is_supported(tmp_path, monkeypatch):
    target = tmp_path / "extensions"
    target.mkdir()
    document = yaml.safe_load((Path(loader.DEFINITIONS_DIR) / "llama_cpp.yaml").read_text())
    document["id"] = "explicit_fixture"
    document["label"] = "Explicit fixture"
    (target / "explicit_fixture.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    monkeypatch.setenv("STUDIO_ENGINE_DEFINITIONS_DIR", str(target))
    monkeypatch.setattr(loader, "_default_registry", None)

    assert "explicit_fixture" in loader.load_default_engine_definitions().ids()


def test_external_definition_cannot_replace_shipped_engine(tmp_path, monkeypatch):
    target = tmp_path / "config" / "engine-definitions"
    target.mkdir(parents=True)
    document = yaml.safe_load((Path(loader.DEFINITIONS_DIR) / "llama_cpp.yaml").read_text())
    (target / "llama_cpp.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("STUDIO_ENGINE_DEFINITIONS_DIR", raising=False)
    monkeypatch.setattr(loader, "_default_registry", None)

    with pytest.raises(DefinitionError, match="cannot replace shipped IDs"):
        loader.load_default_engine_definitions()


def test_invalid_external_definition_fails_validation(tmp_path, monkeypatch):
    target = tmp_path / "config" / "engine-definitions"
    target.mkdir(parents=True)
    (target / "bad.yaml").write_text("schema_version: 2\nid: wrong_filename\n")
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("STUDIO_ENGINE_DEFINITIONS_DIR", raising=False)
    monkeypatch.setattr(loader, "_default_registry", None)

    with pytest.raises(DefinitionError):
        loader.load_default_engine_definitions()
