# AGENTS.md — Collaborator Guide for `Rickaliq/llama-cpp-studio`

This repository is a fork of [`lapy/llama-cpp-studio`](https://github.com/lapy/llama-cpp-studio) (`upstream`) maintained on branch `custom-fork` (`origin`).

## Core Rules for AI & Human Collaborators

1. **Never commit runtime data or secrets:**
   - `STUDIO_DATA_DIR` (`engines.yaml`, `models.yaml`, `engine_setup_defaults.yaml`, `engine-definitions/`, `proxy-accounting.sqlite3`, API keys, and local model paths) lives **outside** this repository.
   - Never hardcode host-specific absolute paths, usernames, or credentials in source files or tests.
2. **Unified Engine Plugin Contract:**
   - All engine help parsers and launch compilers are registered in [`backend/engines/plugins.py`](backend/engines/plugins.py).
   - To add a new engine:
     1. Add its YAML definition in `backend/engines/definitions/<engine>.yaml` (or in `STUDIO_DATA_DIR/config/engine-definitions/` for host-local experimental engines).
     2. Add a help parser in `backend/engines/scan/<engine>_help.py` and register it in `plugins.py` (`get_help_parsers()` and `SUPPORTED_PARSER_NAMES`).
     3. Add a launch compiler in `backend/proxy/<engine>_launch.py` and register it in `plugins.py` (`get_launch_compiler()` and `SUPPORTED_LAUNCH_FAMILIES`).
   - Do **not** add parallel `if/elif` dispatch chains in `schema.py`, `scanner.py`, or `profile_launch.py`.
3. **Multi-GPU Detection:**
   - [`backend/gpu_detector.py`](backend/gpu_detector.py) merges both NVIDIA (`pynvml`/`nvidia-smi`) and AMD (`/sys/class/drm` / `rocm-smi` / `lspci`) GPUs in `_collect_gpu_list()` and `_collect_gpu_info()`. Preserve multi-vendor support when editing hardware detection.

## Verification Commands

Run from the repository root using the project `.venv`:

```bash
# Backend engine & API tests
.venv/bin/python -m pytest -q \
  backend/tests/test_engine_definition_extensions.py \
  backend/tests/test_koboldcpp_launch.py \
  backend/tests/test_koboldcpp_scan.py \
  backend/tests/test_moe4all_launch.py \
  backend/tests/test_strata_launch.py \
  backend/tests/test_engine_setup_defaults.py

# Full backend suite
.venv/bin/python -m pytest -q backend/tests

# Frontend tests
npm test
```
