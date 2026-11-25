"""Unified plugin registry for Studio engine help parsers and launch compilers."""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet


HelpParser = Callable[[str, str], list[dict]]
LaunchCompiler = Callable[..., Any]


def get_help_parsers() -> Dict[str, HelpParser]:
    """Return the unified mapping of discovery.parser name -> parser callable."""
    from backend.engines.scan.help_parsers import (
        lmdeploy_params_to_sections,
        parse_infr_serve_help_to_sections,
        parse_llama_help_to_sections,
        parse_lmdeploy_api_server_help,
        parse_sglang_diffusion_help,
        parse_sglang_launch_server_help,
        parse_vllm_serve_help,
        sglang_params_to_sections,
        vllm_params_to_sections,
    )
    from backend.engines.scan.koboldcpp_help import parse_koboldcpp_help_to_sections
    from backend.engines.scan.maya_ik2_help import parse_maya_ik2_help_to_sections

    return {
        "infr_help": parse_infr_serve_help_to_sections,
        "koboldcpp_help": parse_koboldcpp_help_to_sections,
        "llama_help": parse_llama_help_to_sections,
        "maya_ik2_help": parse_maya_ik2_help_to_sections,
        "lmdeploy_help": lambda text, engine: lmdeploy_params_to_sections(
            parse_lmdeploy_api_server_help(text)
        ),
        "vllm_help": lambda text, engine: vllm_params_to_sections(parse_vllm_serve_help(text)),
        "sglang_help": lambda text, engine: sglang_params_to_sections(
            parse_sglang_launch_server_help(text)
        ),
        "sglang_diffusion_help": lambda text, engine: sglang_params_to_sections(
            parse_sglang_diffusion_help(text)
        ),
    }


def get_launch_compiler(family: str) -> LaunchCompiler | None:
    """Return the launch compiler for a declared runtime_profile.launch_family."""
    if family == "infr_gguf_server":
        from backend.proxy.infr_launch import compile_infr_gguf_launch
        return compile_infr_gguf_launch
    if family == "profile_bundle_http_server":
        from backend.proxy.profile_bundle_launch import compile_profile_bundle_launch
        return compile_profile_bundle_launch
    if family == "koboldcpp_gguf_server":
        from backend.proxy.koboldcpp_launch import compile_koboldcpp_gguf_launch
        return compile_koboldcpp_gguf_launch
    if family == "maya_ik2_gguf_server":
        from backend.proxy.maya_ik2_launch import compile_maya_ik2_gguf_launch
        return compile_maya_ik2_gguf_launch
    return None


SUPPORTED_PARSER_NAMES: FrozenSet[str] = frozenset({
    "llama_help",
    "koboldcpp_help",
    "maya_ik2_help",
    "infr_help",
    "vllm_help",
    "sglang_help",
    "sglang_diffusion_help",
    "lmdeploy_help",
})

SUPPORTED_LAUNCH_FAMILIES: FrozenSet[str] = frozenset({
    "gguf_llama_server",
    "infr_gguf_server",
    "profile_bundle_http_server",
    "koboldcpp_gguf_server",
    "maya_ik2_gguf_server",
})
