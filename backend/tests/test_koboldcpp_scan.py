from __future__ import annotations

from backend.data_store import DataStore
from backend.engines.scan import scanner


HELP_TEXT = """usage: koboldcpp [--model [filenames]] [--contextsize [256 to 524288]]
                [--gpulayers [[GPU layers]]] [--host [ipaddr]] [--port [portnumber]]
                [--usevulkan [[Device IDs]]] [--config [filename]]

Options:
  --model [filenames]  Model file to load
  --contextsize [256 to 524288], --ctx-size [256 to 524288]
                        Controls memory allocated for context. (default 16384).
  --gpulayers [[GPU layers]], --gpu-layers [[GPU layers]], --n-gpu-layers [[GPU layers]]
                        Set number of layers to offload to GPU. (default -1).
  --host [ipaddr]       Host IP to listen on.
  --port [portnumber]   Port to listen on. (Defaults to 5001)
  --usevulkan [[Device IDs]]
                        Use Vulkan for GPU acceleration.
  --config [filename]   Load settings from a file.
"""


class _FakeScanSession:
    _finished = False

    def set_stage(self, *_args, **_kwargs):
        return None

    def log_catalog(self, _entry):
        return None

    def finish_from_entry(self, *_args, **_kwargs):
        self._finished = True

    def detach(self):
        return None


def test_registered_kobold_row_scans_its_binary_and_filters_controls(tmp_path, monkeypatch):
    binary = tmp_path / "koboldcpp-linux-x64"
    marker = tmp_path / "probe-argv.txt"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"open({str(marker)!r}, 'w').write(' '.join(sys.argv[1:]))\n"
        f"sys.stdout.write({HELP_TEXT!r})\n"
    )
    binary.chmod(0o755)

    store = DataStore(config_dir=str(tmp_path / "studio-config"))
    store.add_engine_version(
        "koboldcpp",
        {
            "version": "fixture-v1",
            "binary_path": str(binary),
            "build_status": "ready",
            "compatibility_status": "unverified",
        },
    )
    row = store.get_engine_versions("koboldcpp")[0]
    session = _FakeScanSession()
    monkeypatch.setattr(scanner, "get_param_scan_session", lambda: None)
    monkeypatch.setattr(scanner, "start_param_scan", lambda *_a, **_kw: session)
    monkeypatch.setattr(scanner, "upsert_version_entry", lambda *_a, **_kw: None)
    monkeypatch.setattr(scanner, "_clear_llama_flags_cache", lambda: None)

    entry = scanner.scan_engine_version(store, "koboldcpp", row)

    assert entry["scan_error"] is None, entry
    assert entry["binary_path"] == str(binary.resolve())
    assert marker.read_text() == "--help"
    params = {
        param["primary_flag"]
        for section in entry["sections"]
        for param in section["params"]
    }
    assert {"--contextsize", "--n-gpu-layers", "--usevulkan"} <= params
    assert not {"--host", "--port", "--config"} & params
    assert session._finished is True
