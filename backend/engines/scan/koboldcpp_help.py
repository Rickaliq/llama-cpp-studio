"""KoboldCpp discovery limited to reviewed model-serving controls."""

from __future__ import annotations

from backend.engines.scan.help_parsers import parse_llama_help_to_sections


# Exclude orchestration, remote-access, credential, file-loading, and
# non-text-model features from Studio's generic model settings form.
SUPPORTED_FLAGS = frozenset(
    {
        "--autofit",
        "--autofitpadding",
        "--batchsize",
        "--blasthreads",
        "--contextsize",
        "--defaultgenamt",
        "--n-gpu-layers",
        "--n-gpu-layers-draft",
        "--main-gpu",
        "--nommq",
        "--noflashattention",
        "--noshift",
        "--noswa",
        "--parallelrequests",
        "--quantkv",
        "--threads",
        "--tensor_split",
        "--ubatchsize",
        "--usecpu",
        "--usemmap",
        "--usemlock",
        "--usevulkan",
    }
)


def parse_koboldcpp_help_to_sections(text: str, engine: str) -> list[dict]:
    """Reuse llama.cpp help parsing, then expose only safe server controls.

    KoboldCpp implements a broad desktop/application CLI. Studio manages only
    one local text-model process, so features like config-file loading, GUI or
    browser launch, credentials, tunnels, RPC serving, and unrelated media
    models stay outside the generic configuration surface.
    """
    sections = parse_llama_help_to_sections(text, engine)
    filtered = []
    for section in sections:
        params = [
            param
            for param in section.get("params", [])
            if str(param.get("primary_flag") or "") in SUPPORTED_FLAGS
        ]
        if params:
            filtered.append({**section, "params": params})
    return filtered


__all__ = ["SUPPORTED_FLAGS", "parse_koboldcpp_help_to_sections"]
