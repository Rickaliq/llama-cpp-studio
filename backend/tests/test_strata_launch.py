import json
from pathlib import Path

import pytest

from backend.proxy import launch_spec, profile_bundle_launch
from backend.proxy.launch_spec import LaunchCompileError, PORT_PLACEHOLDER


def _profile(tmp_path: Path, **overrides):
    root = tmp_path / "Strata"
    (root / "serve").mkdir(parents=True)
    (root / "serve" / "server.py").write_text("# fixture; never executed\n")
    (root / "venv" / "lib").mkdir(parents=True)
    (root / "tools" / "hip").mkdir(parents=True)
    (root / "engine").mkdir()
    worker = root / "engine" / "strata"
    worker.write_text("fixture\n")
    worker.chmod(0o755)
    pack = root / "packs" / "q2_0"
    pack.mkdir(parents=True)
    tokenizer = pack / "tokenizer"
    tokenizer.mkdir()
    files = {}
    for name in ("shard1.gguf", "shard2.gguf", "expert-profile.bin"):
        path = root / name
        path.write_text("fixture\n")
        files[name] = str(path)
    profile = {
        "exe": str(worker),
        "args": [
            "--pack", str(pack), "--native", files["shard1.gguf"],
            "--ple-gguf", files["shard2.gguf"], "--expert-profile", files["expert-profile.bin"],
            "--expert-cache", "auto", "--prefill", "2048", "--spec", "2",
            "--max-context", "4096", "--kv", "q4_0", "--vram-reserve-mib", "1024",
            "--mmap-experts", "--gpu", "0", "--memory-limit-mib", "16384",
            "--conversation-cache-mib", "0", "--prompt-cache", "0",
        ],
        "cwd": str(root),
        "tokenizer": str(tokenizer),
        "model_name": "qwen3.8-flash-next-q2_0",
        "log": str(root / "original.log"),
        "lib_dirs": [str(root / "venv" / "lib")],
        "port": 8090,
        "host": "127.0.0.1",
        "parallel": 1,
        "max_queue": 0,
        "offline": True,
        "backend": "hip",
        "env": {"STRATA_ROCBLAS_TUNING": str(root / "tools" / "hip")},
    }
    profile.update(overrides)
    path = tmp_path / "qwen38-profile.json"
    path.write_text(json.dumps(profile))
    return path, root, profile


@pytest.fixture
def registered_strata_python(monkeypatch, tmp_path):
    python = Path("/bin/true")
    monkeypatch.setattr(
        launch_spec,
        "_active_row",
        lambda engine: {"binary_path": str(python), "version": "test"} if engine == "strata" else {},
    )
    return python


def _model(profile: Path, *, fmt="strata_bundle"):
    return {
        "id": "local-strata-qwen",
        "local_path": str(profile),
        "format": fmt,
        "artifact": {"format": fmt, "package_kind": "prepared_bundle", "path": str(profile)},
        "config": {"engine": "strata"},
    }


def test_strata_compiles_explicit_profile_as_private_eager_server_config(
    tmp_path, registered_strata_python
):
    path, root, source = _profile(tmp_path)
    definition = __import__("backend.proxy.profile_launch", fromlist=["profile_launch_definition"]).profile_launch_definition("strata")
    spec = definition.profile("")
    launch = profile_bundle_launch.compile_profile_bundle_launch(
        definition=definition, spec=spec, model=_model(path), config={},
        stable_id="local-strata-qwen", engine="strata",
    )

    assert launch.executable == str(registered_strata_python)
    assert launch.cwd == str(root)
    assert "--lazy" not in launch.argv
    assert "--offline" in launch.argv
    assert PORT_PLACEHOLDER in launch.argv
    assert launch.health_requires_loaded is True
    runtime = launch.artifacts[profile_bundle_launch.SERVER_CONFIG]
    assert runtime["host"] == "127.0.0.1"
    assert runtime["port"] == 0
    assert runtime["offline"] is True
    assert runtime["model_name"] == source["model_name"]
    assert runtime["args"][runtime["args"].index("--expert-cache") + 1] == "5000"
    assert runtime["args"][runtime["args"].index("--vram-reserve-mib") + 1] == "3072"
    assert runtime["args"] != source["args"]
    assert json.loads(path.read_text()) == source
    assert "api_key" not in runtime
    assert "STRATA_API_KEY" in launch.env.unset


def test_strata_preserves_smaller_cache_and_larger_reserve(tmp_path, registered_strata_python):
    path, _root, source = _profile(tmp_path)
    args = source["args"]
    args[args.index("--expert-cache") + 1] = "2500"
    args[args.index("--vram-reserve-mib") + 1] = "4096"
    path.write_text(json.dumps(source))
    _p, profile = profile_bundle_launch._read_profile(_model(path))
    runtime, _model_name = profile_bundle_launch._validated_profile(profile)
    assert runtime["args"][runtime["args"].index("--expert-cache") + 1] == "2500"
    assert runtime["args"][runtime["args"].index("--vram-reserve-mib") + 1] == "4096"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"api_key": "never-copy"}, "Unsupported Strata profile fields"),
        ({"host": "0.0.0.0"}, "must bind to loopback"),
        ({"offline": False}, "must enable offline mode"),
        ({"port": 8091, "unknown": True}, "Unsupported Strata profile fields"),
    ],
)
def test_strata_rejects_unsafe_or_unknown_profile_fields(tmp_path, changes, message):
    path, _root, profile = _profile(tmp_path)
    profile.update(changes)
    path.write_text(json.dumps(profile))
    with pytest.raises(LaunchCompileError, match=message):
        _p, parsed = profile_bundle_launch._read_profile(_model(path))
        profile_bundle_launch._validated_profile(parsed)


def test_strata_rejects_non_bundle_gguf_records(tmp_path):
    path, _root, _ = _profile(tmp_path)
    with pytest.raises(LaunchCompileError, match="explicitly registered as strata_bundle"):
        profile_bundle_launch._read_profile(_model(path, fmt="gguf"))


def test_strata_rejects_unrecognized_worker_args(tmp_path):
    _path, _root, profile = _profile(tmp_path)
    with pytest.raises(LaunchCompileError, match="Unsupported Strata worker option"):
        profile_bundle_launch._sanitize_worker_args(["--api-key", "secret"])


def test_strata_rejects_unreviewed_runtime_environment(tmp_path, registered_strata_python):
    path, _root, _ = _profile(tmp_path)
    definition = __import__("backend.proxy.profile_launch", fromlist=["profile_launch_definition"]).profile_launch_definition("strata")
    with pytest.raises(LaunchCompileError, match="environment controls must come"):
        profile_bundle_launch.compile_profile_bundle_launch(
            definition=definition, spec=definition.profile(""), model=_model(path),
            config={"runtime_env": {"STRATA_API_KEY": "secret"}},
            stable_id="local-strata-qwen", engine="strata",
        )

