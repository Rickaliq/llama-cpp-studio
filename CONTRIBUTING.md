# Contributing to `Rickaliq/llama-cpp-studio`

Thank you for contributing to this fork of `llama-cpp-studio`!

## Repository & Branching

- **`upstream`**: `https://github.com/lapy/llama-cpp-studio.git` (`main`)
- **`origin`**: `https://github.com/Rickaliq/llama-cpp-studio.git` (`custom-fork`)

To sync with upstream:

```bash
git fetch upstream
git checkout custom-fork
git merge upstream/main   # or git rebase upstream/main
```

## Adding a Custom Engine

1. **Definition (`backend/engines/definitions/<engine>.yaml`):** Declare `id`, `label`, `runtime_kind`, `discovery.parser`, and `runtime_profiles.default.launch_family`.
2. **Help Parser (`backend/engines/scan/<engine>_help.py`):** Parse `--help` output into Studio parameter sections, filtering to safe local-server controls.
3. **Launch Compiler (`backend/proxy/<engine>_launch.py`):** Compile Studio's structured `LaunchSpec` for your engine's CLI contract.
4. **Register in [`backend/engines/plugins.py`](backend/engines/plugins.py):** Add your parser and launch family to `get_help_parsers()`, `get_launch_compiler()`, `SUPPORTED_PARSER_NAMES`, and `SUPPORTED_LAUNCH_FAMILIES`.
5. **Tests:** Add unit tests under `backend/tests/` covering definition validation, help parsing, and launch compilation.

See [`docs/custom-engines-and-fork.md`](docs/custom-engines-and-fork.md) for details.

## Pull Request Checklist

- [ ] No local filesystem paths, API keys, or `STUDIO_DATA_DIR` files included.
- [ ] Backend tests pass (`.venv/bin/python -m pytest -q backend/tests`).
- [ ] Frontend tests pass (`npm test`) if UI files changed.
- [ ] Documentation updated if new engines or endpoints were added.
