from backend.data_store import DataStore
from backend.engines.params import upsert_version_entry
from backend.routes import engine_setup_defaults as routes


def _store_with_scan(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.add_engine_version("llama_cpp", {"version": "test"})
    store.set_active_engine_version("llama_cpp", "test")
    upsert_version_entry(
        store,
        "llama_cpp",
        "test",
        {
            "sections": [{"params": [
                {"key": "temperature", "flags": ["--temp"], "type": "float", "minimum": 0, "maximum": 2},
                {"key": "use_mmap", "flags": ["--mmap", "--no-mmap"], "type": "bool", "value_kind": "flag"},
                {"key": "unsafe", "flags": ["--unsafe"], "type": "int", "reserved": True},
            ]}],
        },
    )
    return store


def _store_with_moe4all_scan(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.add_engine_version("moe4all", {"version": "test"})
    store.set_active_engine_version("moe4all", "test")
    upsert_version_entry(
        store,
        "moe4all",
        "test",
        {"sections": [{"params": [
            {"key": "set", "flags": ["--set"], "type": "array", "value_kind": "repeatable", "multiple": True},
        ]}]},
    )
    return store


def test_get_returns_only_scanned_safe_setup_values(tmp_path, monkeypatch):
    store = _store_with_scan(tmp_path)
    monkeypatch.setattr(routes, "get_store", lambda: store)
    monkeypatch.setattr(routes, "active_engine_row_is_runnable", lambda *_: True)
    store.write_document(routes.FILENAME, {"engines": {"llama_cpp": {"values": {
        "temperature": 0, "use_mmap": False, "unsafe": 1,
    }}}})

    result = routes._read_defaults(store, "llama_cpp")

    assert result["values"] == {"temperature": 0, "use_mmap": False}
    assert result["unavailable_values"] == {"unsafe": 1}
    assert "temperature" in result["catalog"]
    assert "revision" not in result


def test_get_hides_values_that_no_longer_match_scanned_type(tmp_path, monkeypatch):
    store = _store_with_scan(tmp_path)
    monkeypatch.setattr(routes, "active_engine_row_is_runnable", lambda *_: True)
    store.write_document(routes.FILENAME, {"engines": {"llama_cpp": {"values": {
        "temperature": 0.5,
    }}}})
    upsert_version_entry(
        store,
        "llama_cpp",
        "test",
        {"sections": [{"params": [
            {"key": "temperature", "flags": ["--temp"], "type": "int", "minimum": 0, "maximum": 2},
        ]}]},
    )

    result = routes._read_defaults(store, "llama_cpp")

    assert result["values"] == {}
    assert result["unavailable_values"] == {"temperature": 0.5}


def test_get_accepts_approved_moe4all_flash_split_default_and_hides_invalid_value(tmp_path, monkeypatch):
    store = _store_with_moe4all_scan(tmp_path)
    monkeypatch.setattr(routes, "active_engine_row_is_runnable", lambda *_: True)
    store.write_document(routes.FILENAME, {"engines": {"moe4all": {"values": {
        "set": ["kv.dynamic=true", "kernels.vulkan.flash_splits=2"],
    }}}})

    result = routes._read_defaults(store, "moe4all")

    assert result["values"] == {"set": ["kv.dynamic=true", "kernels.vulkan.flash_splits=2"]}
    assert result["unavailable_values"] == {}

    store.write_document(routes.FILENAME, {"engines": {"moe4all": {"values": {
        "set": ["kernels.vulkan.flash_splits=3"],
    }}}})
    invalid = routes._read_defaults(store, "moe4all")
    assert invalid["values"] == {}
    assert invalid["unavailable_values"] == {"set": ["kernels.vulkan.flash_splits=3"]}


def test_setup_defaults_route_is_read_only():
    route_methods = {
        method
        for route in routes.router.routes
        for method in route.methods
        if route.path == "/{engine}"
    }
    assert route_methods == {"GET"}
