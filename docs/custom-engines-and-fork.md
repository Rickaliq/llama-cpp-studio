# Custom Engines and Fork Additions

This document summarizes the extensions maintained on `custom-fork` in `Rickaliq/llama-cpp-studio` beyond upstream `lapy/llama-cpp-studio`.

## 1. Unified Engine Plugin Registry

[`backend/engines/plugins.py`](../backend/engines/plugins.py) provides a single registration point for custom engine help parsers and launch compilers:

- `get_help_parsers()` / `SUPPORTED_PARSER_NAMES` — consumed by `backend/engines/scan/scanner.py` and `backend/engines/definitions/schema.py`.
- `get_launch_compiler(family)` / `SUPPORTED_LAUNCH_FAMILIES` — consumed by `backend/proxy/profile_launch.py` and `backend/engines/definitions/schema.py`.

## 2. Added Custom Engines

| Engine | ID | Launch Family | Help Parser | Notes |
| --- | --- | --- | --- | --- |
| KoboldCpp | `koboldcpp` | `koboldcpp_gguf_server` | `koboldcpp_help` | Constrained local GGUF server (`--skiplauncher`, `--quantkv`, `--usevulkan`). |
| MoE4All | `moe4all` | `infr_gguf_server` | `infr_help` | `infr serve` OpenAI-compatible GGUF server with allowlisted `--set` keys. |
| Strata | `strata` | `profile_bundle_http_server` | `strata_profile` | Prepared-bundle profile runtime. |
| Maya-IK2 | `maya_ik2` | `maya_ik2_gguf_server` | `maya_ik2_help` | Additive definition support for `maya-ik2-serve` (Inkling-2-Mini). |

## 3. Multi-GPU Detection

[`backend/gpu_detector.py`](../backend/gpu_detector.py) merges NVIDIA (`pynvml` / `nvidia-smi`) and AMD (`/sys/class/drm` / `rocm-smi` / `lspci`) devices in `_collect_gpu_list()` and `_collect_gpu_info()` so mixed-vendor hosts report all GPUs and combined VRAM in `/api/gpu-list` and `/api/gpu-info`.

## 4. Additional Fork Capabilities

- **Engine Setup Defaults:** `backend/routes/engine_setup_defaults.py` (`GET`/`PUT /api/engine-setup-defaults/{engine}`) for per-engine default seeding.
- **Local GGUF Import & Autoconfig:** `backend/services/model_autoconfig.py` and by-reference local GGUF import routes.
