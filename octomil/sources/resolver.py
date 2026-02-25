"""Unified model resolution across Ollama, HuggingFace, and Kaggle.

Resolves human-friendly model names to local file paths, downloading
from the best available source if needed.

Supports:
- Explicit sources: ``hf:org/model``, ``ollama:name:tag``, ``kaggle:org/model``
- Known aliases: ``phi-4-mini``, ``gemma-1b``, ``llama-3.2-3b``
- Direct HuggingFace repo IDs: ``microsoft/Phi-4-mini-instruct``
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import click

from .base import SourceResult
from .huggingface import HuggingFaceSource
from .kaggle import KaggleSource
from .ollama import OllamaSource

logger = logging.getLogger(__name__)

# ── Known model aliases ──────────────────────────────────────────────────────
# Maps common short names to source-specific identifiers.

_MODEL_ALIASES: Dict[str, Dict[str, str]] = {
    # ── Phi family ────────────────────────────────────────────────────────
    "phi-4-mini": {"hf": "microsoft/Phi-4-mini-instruct", "ollama": "phi4-mini"},
    "phi-mini": {"hf": "microsoft/Phi-3.5-mini-instruct", "ollama": "phi3.5"},
    "phi-4": {"hf": "microsoft/phi-4", "ollama": "phi4"},
    "phi4": {"hf": "microsoft/phi-4", "ollama": "phi4"},
    "phi4-mini": {"hf": "microsoft/Phi-4-mini-instruct", "ollama": "phi4-mini"},
    # ── Gemma 3 family ────────────────────────────────────────────────────
    "gemma-1b": {"hf": "google/gemma-3-1b-it", "ollama": "gemma3:1b"},
    "gemma-3b": {"hf": "google/gemma-3-4b-it", "ollama": "gemma3:4b"},
    "gemma-4b": {"hf": "google/gemma-3-4b-it", "ollama": "gemma3:4b"},
    "gemma-12b": {"hf": "google/gemma-3-12b-it", "ollama": "gemma3:12b"},
    "gemma-27b": {"hf": "google/gemma-3-27b-it", "ollama": "gemma3:27b"},
    # ── Gemma 2 family ────────────────────────────────────────────────────
    "gemma2-2b": {"hf": "google/gemma-2-2b-it", "ollama": "gemma2:2b"},
    "gemma2-9b": {"hf": "google/gemma-2-9b-it", "ollama": "gemma2:9b"},
    "gemma2-27b": {"hf": "google/gemma-2-27b-it", "ollama": "gemma2:27b"},
    "gemma2": {"hf": "google/gemma-2-9b-it", "ollama": "gemma2:9b"},
    # ── Llama 3.2 family ──────────────────────────────────────────────────
    "llama-1b": {"hf": "meta-llama/Llama-3.2-1B-Instruct", "ollama": "llama3.2:1b"},
    "llama-3b": {"hf": "meta-llama/Llama-3.2-3B-Instruct", "ollama": "llama3.2:3b"},
    "llama-3.2-1b": {"hf": "meta-llama/Llama-3.2-1B-Instruct", "ollama": "llama3.2:1b"},
    "llama-3.2-3b": {"hf": "meta-llama/Llama-3.2-3B-Instruct", "ollama": "llama3.2:3b"},
    "llama3.2-1b": {"hf": "meta-llama/Llama-3.2-1B-Instruct", "ollama": "llama3.2:1b"},
    "llama3.2-3b": {"hf": "meta-llama/Llama-3.2-3B-Instruct", "ollama": "llama3.2:3b"},
    # ── Llama 3.1 family ──────────────────────────────────────────────────
    "llama-8b": {"hf": "meta-llama/Meta-Llama-3.1-8B-Instruct", "ollama": "llama3.1:8b"},
    "llama-3.1-8b": {"hf": "meta-llama/Meta-Llama-3.1-8B-Instruct", "ollama": "llama3.1:8b"},
    "llama-3.1-70b": {"hf": "meta-llama/Meta-Llama-3.1-70B-Instruct", "ollama": "llama3.1:70b"},
    "llama3.1": {"hf": "meta-llama/Meta-Llama-3.1-8B-Instruct", "ollama": "llama3.1:8b"},
    "llama3.1-8b": {"hf": "meta-llama/Meta-Llama-3.1-8B-Instruct", "ollama": "llama3.1:8b"},
    "llama3.1-70b": {"hf": "meta-llama/Meta-Llama-3.1-70B-Instruct", "ollama": "llama3.1:70b"},
    # ── Llama 3.3 family ──────────────────────────────────────────────────
    "llama-3.3-70b": {"hf": "meta-llama/Llama-3.3-70B-Instruct", "ollama": "llama3.3:70b"},
    "llama3.3": {"hf": "meta-llama/Llama-3.3-70B-Instruct", "ollama": "llama3.3:70b"},
    "llama3.3-70b": {"hf": "meta-llama/Llama-3.3-70B-Instruct", "ollama": "llama3.3:70b"},
    # ── Qwen 3 family ────────────────────────────────────────────────────
    "qwen3": {"hf": "Qwen/Qwen3-4B", "ollama": "qwen3:4b"},
    "qwen3-0.6b": {"hf": "Qwen/Qwen3-0.6B", "ollama": "qwen3:0.6b"},
    "qwen3-1b": {"hf": "Qwen/Qwen3-1.7B", "ollama": "qwen3:1.7b"},
    "qwen3-1.7b": {"hf": "Qwen/Qwen3-1.7B", "ollama": "qwen3:1.7b"},
    "qwen3-4b": {"hf": "Qwen/Qwen3-4B", "ollama": "qwen3:4b"},
    "qwen3-8b": {"hf": "Qwen/Qwen3-8B", "ollama": "qwen3:8b"},
    "qwen3-14b": {"hf": "Qwen/Qwen3-14B", "ollama": "qwen3:14b"},
    "qwen3-32b": {"hf": "Qwen/Qwen3-32B", "ollama": "qwen3:32b"},
    # ── Qwen 2.5 family ──────────────────────────────────────────────────
    "qwen-1.5b": {"hf": "Qwen/Qwen2.5-1.5B-Instruct", "ollama": "qwen2.5:1.5b"},
    "qwen-3b": {"hf": "Qwen/Qwen2.5-3B-Instruct", "ollama": "qwen2.5:3b"},
    "qwen-7b": {"hf": "Qwen/Qwen2.5-7B-Instruct", "ollama": "qwen2.5:7b"},
    "qwen2.5-coder-1.5b": {"hf": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "ollama": "qwen2.5-coder:1.5b"},
    "qwen2.5-coder-7b": {"hf": "Qwen/Qwen2.5-Coder-7B-Instruct", "ollama": "qwen2.5-coder:7b"},
    "qwen2.5-coder": {"hf": "Qwen/Qwen2.5-Coder-7B-Instruct", "ollama": "qwen2.5-coder:7b"},
    # ── DeepSeek R1 family ────────────────────────────────────────────────
    "deepseek-r1": {"hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "ollama": "deepseek-r1:7b"},
    "deepseek-r1-1.5b": {"hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ollama": "deepseek-r1:1.5b"},
    "deepseek-r1-7b": {"hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "ollama": "deepseek-r1:7b"},
    "deepseek-r1-8b": {"hf": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "ollama": "deepseek-r1:8b"},
    "deepseek-r1-14b": {"hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "ollama": "deepseek-r1:14b"},
    "deepseek-r1-32b": {"hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "ollama": "deepseek-r1:32b"},
    "deepseek-r1-70b": {"hf": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "ollama": "deepseek-r1:70b"},
    # ── DeepSeek other ────────────────────────────────────────────────────
    "deepseek-v3": {"hf": "deepseek-ai/DeepSeek-V3", "ollama": "deepseek-v3"},
    "deepseek-coder": {"hf": "deepseek-ai/deepseek-coder-6.7b-instruct", "ollama": "deepseek-coder:6.7b"},
    "deepseek-coder-v2": {"hf": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "ollama": "deepseek-coder-v2"},
    # ── Mistral family ────────────────────────────────────────────────────
    "mistral-7b": {"hf": "mistralai/Mistral-7B-Instruct-v0.3", "ollama": "mistral"},
    "mistral-nemo": {"hf": "mistralai/Mistral-Nemo-Instruct-2407", "ollama": "mistral-nemo"},
    "mistral-nemo-12b": {"hf": "mistralai/Mistral-Nemo-Instruct-2407", "ollama": "mistral-nemo"},
    "mistral-small": {"hf": "mistralai/Mistral-Small-24B-Instruct-2501", "ollama": "mistral-small"},
    "mistral-small-24b": {"hf": "mistralai/Mistral-Small-24B-Instruct-2501", "ollama": "mistral-small"},
    "mixtral": {"hf": "mistralai/Mixtral-8x7B-Instruct-v0.1", "ollama": "mixtral"},
    "mixtral-8x7b": {"hf": "mistralai/Mixtral-8x7B-Instruct-v0.1", "ollama": "mixtral"},
    "mixtral-8x22b": {"hf": "mistralai/Mixtral-8x22B-Instruct-v0.1", "ollama": "mixtral:8x22b"},
    "codestral": {"hf": "mistralai/Codestral-22B-v0.1", "ollama": "codestral"},
    "mathstral": {"hf": "mistralai/mathstral-7B-v0.1", "ollama": "mathstral"},
    "mathstral-7b": {"hf": "mistralai/mathstral-7B-v0.1", "ollama": "mathstral"},
    # ── CodeLlama family ──────────────────────────────────────────────────
    "codellama": {"hf": "meta-llama/CodeLlama-7b-Instruct-hf", "ollama": "codellama:7b"},
    "codellama-7b": {"hf": "meta-llama/CodeLlama-7b-Instruct-hf", "ollama": "codellama:7b"},
    "codellama-13b": {"hf": "meta-llama/CodeLlama-13b-Instruct-hf", "ollama": "codellama:13b"},
    "codellama-34b": {"hf": "meta-llama/CodeLlama-34b-Instruct-hf", "ollama": "codellama:34b"},
    # ── StarCoder2 family ─────────────────────────────────────────────────
    "starcoder2": {"hf": "bigcode/starcoder2-15b-instruct-v0.1", "ollama": "starcoder2:15b"},
    "starcoder2-3b": {"hf": "bigcode/starcoder2-3b", "ollama": "starcoder2:3b"},
    "starcoder2-7b": {"hf": "bigcode/starcoder2-7b", "ollama": "starcoder2:7b"},
    "starcoder2-15b": {"hf": "bigcode/starcoder2-15b-instruct-v0.1", "ollama": "starcoder2:15b"},
    # ── Google code models ────────────────────────────────────────────────
    "codegemma": {"hf": "google/codegemma-7b-it", "ollama": "codegemma:7b"},
    "codegemma-7b": {"hf": "google/codegemma-7b-it", "ollama": "codegemma:7b"},
    # ── Yi family ─────────────────────────────────────────────────────────
    "yi": {"hf": "01-ai/Yi-1.5-6B-Chat", "ollama": "yi:6b"},
    "yi-6b": {"hf": "01-ai/Yi-1.5-6B-Chat", "ollama": "yi:6b"},
    "yi-9b": {"hf": "01-ai/Yi-1.5-9B-Chat", "ollama": "yi:9b"},
    "yi-34b": {"hf": "01-ai/Yi-1.5-34B-Chat", "ollama": "yi:34b"},
    # ── Falcon 3 family ───────────────────────────────────────────────────
    "falcon3": {"hf": "tiiuae/Falcon3-7B-Instruct", "ollama": "falcon3:7b"},
    "falcon3-1b": {"hf": "tiiuae/Falcon3-1B-Instruct", "ollama": "falcon3:1b"},
    "falcon3-7b": {"hf": "tiiuae/Falcon3-7B-Instruct", "ollama": "falcon3:7b"},
    "falcon3-10b": {"hf": "tiiuae/Falcon3-10B-Instruct", "ollama": "falcon3:10b"},
    # ── Cohere Command-R ──────────────────────────────────────────────────
    "command-r": {"hf": "CohereForAI/c4ai-command-r-v01", "ollama": "command-r"},
    "command-r-plus": {"hf": "CohereForAI/c4ai-command-r-plus", "ollama": "command-r-plus"},
    # ── SmolLM ────────────────────────────────────────────────────────────
    "smollm-360m": {"hf": "HuggingFaceTB/SmolLM2-360M-Instruct", "ollama": "smollm2:360m"},
    "smollm-1.7b": {"hf": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "ollama": "smollm2:1.7b"},
    "smollm2-1.7b": {"hf": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "ollama": "smollm2:1.7b"},
    # ── Other popular models ──────────────────────────────────────────────
    "tinyllama": {"hf": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "ollama": "tinyllama"},
    "internlm2": {"hf": "internlm/internlm2_5-7b-chat", "ollama": "internlm2:7b"},
    "internlm2-7b": {"hf": "internlm/internlm2_5-7b-chat", "ollama": "internlm2:7b"},
    "stable-code": {"hf": "stabilityai/stable-code-instruct-3b", "ollama": "stable-code:3b"},
    "stable-code-3b": {"hf": "stabilityai/stable-code-instruct-3b", "ollama": "stable-code:3b"},
    "glm4": {"hf": "THUDM/glm-4-9b-chat", "ollama": "glm4:9b"},
    "glm4-9b": {"hf": "THUDM/glm-4-9b-chat", "ollama": "glm4:9b"},
    "granite-3.2": {"hf": "ibm-granite/granite-3.2-8b-instruct", "ollama": "granite3.2-moe"},
    "granite-3.2-2b": {"hf": "ibm-granite/granite-3.2-2b-instruct", "ollama": "granite3.2-moe:1b"},
    "granite-3.2-8b": {"hf": "ibm-granite/granite-3.2-8b-instruct", "ollama": "granite3.2-moe"},
    "olmo2": {"hf": "allenai/OLMo-2-1124-7B-Instruct", "ollama": "olmo2:7b"},
    "olmo2-7b": {"hf": "allenai/OLMo-2-1124-7B-Instruct", "ollama": "olmo2:7b"},
    "zephyr": {"hf": "HuggingFaceH4/zephyr-7b-beta", "ollama": "zephyr"},
    "zephyr-7b": {"hf": "HuggingFaceH4/zephyr-7b-beta", "ollama": "zephyr"},
    "solar": {"hf": "upstage/SOLAR-10.7B-Instruct-v1.0", "ollama": "solar"},
    "solar-10.7b": {"hf": "upstage/SOLAR-10.7B-Instruct-v1.0", "ollama": "solar"},
    # ── Community fine-tunes ──────────────────────────────────────────────
    "openhermes": {"hf": "teknium/OpenHermes-2.5-Mistral-7B", "ollama": "openhermes"},
    "dolphin-mistral": {"hf": "cognitivecomputations/dolphin-2.6-mistral-7b", "ollama": "dolphin-mistral"},
    "neural-chat": {"hf": "Intel/neural-chat-7b-v3-3", "ollama": "neural-chat"},
    "nous-hermes2-mixtral": {"hf": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", "ollama": "nous-hermes2-mixtral"},
    "orca2": {"hf": "microsoft/Orca-2-7b", "ollama": "orca2:7b"},
    "vicuna": {"hf": "lmsys/vicuna-7b-v1.5", "ollama": "vicuna:7b"},
    "wizardcoder": {"hf": "WizardLMTeam/WizardCoder-Python-7B-V1.0", "ollama": "wizardcoder:7b"},
    "stablelm2": {"hf": "stabilityai/stablelm-2-zephyr-1_6b", "ollama": "stablelm2:1.6b"},
    # ── Whisper (HF only) ─────────────────────────────────────────────────
    "whisper-tiny": {"hf": "openai/whisper-tiny"},
    "whisper-base": {"hf": "openai/whisper-base"},
    "whisper-small": {"hf": "openai/whisper-small"},
    "whisper-medium": {"hf": "openai/whisper-medium"},
    "whisper-large-v3": {"hf": "openai/whisper-large-v3"},
}

_ollama = OllamaSource()
_hf = HuggingFaceSource()
_kaggle = KaggleSource()


def _parse_explicit_source(name: str) -> Optional[tuple[str, str]]:
    """Parse ``hf:org/model`` → ``("hf", "org/model")``. Returns None if no prefix."""
    for prefix in ("hf:", "huggingface:", "ollama:", "kaggle:"):
        if name.startswith(prefix):
            source = prefix.rstrip(":")
            if source == "huggingface":
                source = "hf"
            return source, name[len(prefix):]
    return None


_SOURCE_HINTS: Dict[str, str] = {
    "ollama": "Install Ollama from https://ollama.com",
    "hf": "Install the Python package: pip install huggingface_hub",
    "kaggle": "Install the Kaggle CLI: pip install kaggle",
}


def _try_source(source: str, ref: str) -> Optional[SourceResult]:
    """Attempt a single source. Returns None on failure (with a logged hint)."""
    backends = {"ollama": _ollama, "hf": _hf, "kaggle": _kaggle}
    backend = backends.get(source)
    if backend is None:
        return None
    try:
        if not backend.is_available():
            hint = _SOURCE_HINTS.get(source, "")
            logger.debug("Source %s unavailable for %s", source, ref)
            click.echo(f"  {source} backend unavailable. {hint}", err=True)
            return None
        return backend.resolve(ref)
    except Exception as exc:
        logger.debug("Source %s failed for %s: %s", source, ref, exc)
    return None


def resolve_and_download(name: str) -> str:
    """Resolve a model name and return a local file path.

    Parameters
    ----------
    name:
        Model name, alias, or explicit source reference.

    Returns
    -------
    str
        Local path to the downloaded model.

    Raises
    ------
    RuntimeError
        If the model cannot be resolved from any source.
    """
    # ── Explicit source ───────────────────────────────────────────────────
    explicit = _parse_explicit_source(name)
    if explicit:
        source, ref = explicit
        click.echo(f"  Downloading from {source}: {ref}")
        result = _try_source(source, ref)
        if result:
            if result.cached:
                click.echo(f"  Using cache: {result.path}")
            return result.path
        raise RuntimeError(f"Could not download '{ref}' from {source}")

    # ── Alias lookup ──────────────────────────────────────────────────────
    aliases = _MODEL_ALIASES.get(name)
    if aliases:
        # Try Ollama first (fastest — local cache), then HuggingFace
        if "ollama" in aliases:
            click.echo(f"  Checking Ollama for {aliases['ollama']}...")
            result = _try_source("ollama", aliases["ollama"])
            if result:
                if result.cached:
                    click.echo(f"  Using Ollama cache")
                else:
                    click.echo(f"  Downloaded from Ollama")
                return result.path

        if "hf" in aliases:
            click.echo(f"  Downloading from HuggingFace: {aliases['hf']}")
            result = _try_source("hf", aliases["hf"])
            if result:
                return result.path

        if "kaggle" in aliases:
            result = _try_source("kaggle", aliases["kaggle"])
            if result:
                return result.path

        raise RuntimeError(
            f"Could not download '{name}' from any source. "
            f"Tried: {', '.join(aliases.keys())}"
        )

    # ── Direct HuggingFace repo ID (org/model format) ─────────────────────
    if "/" in name:
        click.echo(f"  Downloading from HuggingFace: {name}")
        result = _try_source("hf", name)
        if result:
            return result.path
        raise RuntimeError(f"Could not download '{name}' from HuggingFace")

    # ── Unknown ───────────────────────────────────────────────────────────
    known = ", ".join(sorted(_MODEL_ALIASES.keys()))
    raise RuntimeError(
        f"Unknown model: '{name}'\n"
        f"  Known models: {known}\n"
        f"  Or use: hf:<org>/<model>, ollama:<name>, kaggle:<path>"
    )


def resolve_hf_repo(name: str) -> Optional[str]:
    """Resolve a model name to a HuggingFace repo ID without downloading.

    Returns the HF repo string (e.g. ``microsoft/Phi-4-mini-instruct``) or
    ``None`` if the name is not a known alias or explicit HF reference.
    """
    # Explicit hf: prefix
    explicit = _parse_explicit_source(name)
    if explicit:
        source, ref = explicit
        return ref if source == "hf" else None

    # Alias lookup
    aliases = _MODEL_ALIASES.get(name)
    if aliases and "hf" in aliases:
        return aliases["hf"]

    # Direct org/model format
    if "/" in name:
        return name

    return None
