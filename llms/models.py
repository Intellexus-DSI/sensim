"""LLM model factory utilities.

This module keeps provider selection centralized and uses LangChain's `init_chat_model`
to construct chat models across providers in a uniform way.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline

import torch
import gc

import os
import yaml
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# SUPPORTED_MODELS: Dict[str, Dict[str, str]] = {
#     "gpt-4o-mini": {"provider": "openai", "api_key_env": "OPENAI_API_KEY"},
#     "claude-3-haiku-20240307": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
#     "gemini-2.5-flash": {"provider": "google_genai", "api_key_env": "GOOGLE_API_KEY"},
#     "gemini-3-flash-preview": {"provider": "google_genai", "api_key_env": "GOOGLE_API_KEY"},
# }

_DEFAULT_MODELS_CONFIG = str(Path(__file__).resolve().parent.parent / "supported_models_config")


def get_supported_models(models_config_folder=_DEFAULT_MODELS_CONFIG):
    # Get all yaml files in the folder
    yaml_files = list(Path(models_config_folder).glob("*.yaml")) + list(Path(models_config_folder).glob("*.yml"))

    supported_models: Dict[str, Dict[str, str]] = {}

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)

            # Check if 'model' property exists and is in llms_committee
            if config and 'model' in config:
                config['config_path'] = str(yaml_file)  # Add the path to the config for reference
                supported_models[config['model']] = config

        except Exception as e:
            print(f"Error loading {yaml_file}: {e}")

    return supported_models

SUPPORTED_MODELS = get_supported_models()  # Load supported models from YAML config files

def _set_api_key_from_dict(keys: Dict[str, Any], env_var: str) -> None:
    """Set required API key env var from a provided keys dict."""
    if env_var not in keys:
        raise KeyError(f"'{env_var}' not found in keys file")
    os.environ[env_var] = str(keys[env_var])


def _maybe_create_rate_limiter(
        use_rate_limiter: bool = False,
        requests_per_second: float = 1,
        check_every_n_seconds: int = 3,
) -> Optional[InMemoryRateLimiter]:
    if not use_rate_limiter:
        return None
    return InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=check_every_n_seconds,
        max_bucket_size=10,
    )


def make_chat_model(
        model_name: str,
        temperature: float,
        keys: Dict[str, Any],
        use_rate_limiter: bool = False,
        requests_per_second: float = 0.7,
        check_every_n_seconds: int = 10,
) -> object:
    """Create a chat model instance.

    Args:
        model_name: One of SUPPORTED_MODELS keys.
        temperature: Sampling temperature for the provider.
        keys: Mapping containing API key env var names (e.g. OPENAI_API_KEY).
        use_rate_limiter: Whether to enable a client-side limiter.
        requests_per_second: RPS for the limiter (if enabled).
        check_every_n_seconds: Limiter check interval.

    Returns:
        A LangChain chat model object (provider dependent).
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {list(SUPPORTED_MODELS.keys())}")

    cfg = SUPPORTED_MODELS[model_name]
    provider = cfg["provider"]
    api_key_env = cfg["api_key_env"]

    _set_api_key_from_dict(keys, api_key_env)

    rate_limiter = _maybe_create_rate_limiter(use_rate_limiter, requests_per_second, check_every_n_seconds)
    LOGGER.info("Initializing chat model provider=%s model=%s temp=%.2f rate_limiter=%s", provider,
                model_name,
                temperature, bool(rate_limiter))

    try:
        # kwargs: Dict[str, Any] = {}
        # if provider == "google_genai":
        #     # langchain_google_genai defaults to v1beta; switch to stable v1
        #     # so that current model aliases (gemini-2.0-flash etc.) resolve correctly.
        #     from langchain_google_genai import ChatGoogleGenerativeAI
        #     from google.genai.types import HttpOptions
        #     return ChatGoogleGenerativeAI(
        #         model=model_name,
        #         temperature=temperature,
        #         rate_limiter=rate_limiter,
        #         http_options=HttpOptions(api_version="v1"),
        #     )
        return init_chat_model(
            model=f"{provider}:{model_name}",
            temperature=temperature,
            rate_limiter=rate_limiter,
            # **kwargs,
        )
    except Exception as e:
        LOGGER.error("Error initializing chat model: %s", e)
        raise


def make_local_chat_model(
        model_name: str,
        temperature: float,
        # keys: Dict[str, Any],
        # use_rate_limiter: bool = False,
        # requests_per_second: float = 0.7,
        # check_every_n_seconds: int = 10,
) -> object:

    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    print(f"Tokenizer length: {len(tokenizer)}")

    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    LOGGER.info("Initialized local chat model %s with temperature %.2f", model_name, temperature)

    return llm
