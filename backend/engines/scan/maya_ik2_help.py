"""Maya-IK2 discovery limited to reviewed local-server controls."""

from __future__ import annotations

from backend.engines.scan.help_parsers import parse_llama_help_to_sections


SUPPORTED_FLAGS = frozenset(
    {
        "--cache-gb",
        "--chunk",
        "--draft",
        "--max-context",
        "--parallel",
        "--threads",
    }
)


def parse_maya_ik2_help_to_sections(text: str, engine: str) -> list[dict]:
    """Reuse llama.cpp help parsing, then expose only reviewed maya-ik2 controls."""
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


__all__ = ["SUPPORTED_FLAGS", "parse_maya_ik2_help_to_sections"]
