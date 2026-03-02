"""
Prompt language management for RAGAnything.

Enables switching prompt templates between languages at runtime.
Addresses GitHub issue #85 — prompt language support.

Usage::

    from raganything.prompt_manager import set_prompt_language, get_prompt_language

    # Switch all prompts to Chinese
    set_prompt_language("zh")

    # Switch back to English (default)
    set_prompt_language("en")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from raganything.prompt import PROMPTS

logger = logging.getLogger(__name__)

# Store the original English prompts as the canonical fallback
_ENGLISH_PROMPTS: Dict[str, Any] = dict(PROMPTS)

# Registry of available prompt languages
_PROMPT_LANGUAGES: Dict[str, Dict[str, Any]] = {
    "en": _ENGLISH_PROMPTS,
}

# Current active language
_current_language: str = "en"


def _lazy_load_language(lang: str) -> Dict[str, Any]:
    """Lazily load prompt templates for a language."""
    if lang == "zh":
        from raganything.prompts_zh import PROMPTS_ZH

        return PROMPTS_ZH
    return {}


def register_prompt_language(language_code: str, prompts: Dict[str, Any]) -> None:
    """Register a new set of prompt templates for a language.

    Args:
        language_code: ISO 639-1 language code (e.g., "zh", "ja", "ko").
        prompts: Dictionary of prompt templates, using the same keys as
                 :data:`raganything.prompt.PROMPTS`.

    Example::

        from raganything.prompt_manager import register_prompt_language

        my_prompts = {"IMAGE_ANALYSIS_SYSTEM": "...in Japanese..."}
        register_prompt_language("ja", my_prompts)
    """
    _PROMPT_LANGUAGES[language_code] = prompts
    logger.info(
        f"Registered prompt language '{language_code}' with {len(prompts)} templates"
    )


def set_prompt_language(language: str) -> None:
    """Switch the active prompt language.

    This replaces the global ``PROMPTS`` dictionary entries with the
    corresponding language templates.  Any keys missing in the target
    language fall back to English.

    Args:
        language: Language code (e.g., "en", "zh").

    Raises:
        ValueError: If the language is not registered and cannot be
                    loaded automatically.
    """
    global _current_language

    lang = language.strip().lower()

    # Try lazy-loading if not already registered
    if lang not in _PROMPT_LANGUAGES:
        loaded = _lazy_load_language(lang)
        if loaded:
            _PROMPT_LANGUAGES[lang] = loaded
        else:
            available = ", ".join(sorted(_PROMPT_LANGUAGES.keys()))
            raise ValueError(
                f"Unknown prompt language '{language}'. "
                f"Available: {available}. "
                f"Register new languages with register_prompt_language()."
            )

    target_prompts = _PROMPT_LANGUAGES[lang]

    # Replace PROMPTS entries, falling back to English for missing keys
    for key in _ENGLISH_PROMPTS:
        if key in target_prompts:
            PROMPTS[key] = target_prompts[key]
        else:
            PROMPTS[key] = _ENGLISH_PROMPTS[key]

    _current_language = lang
    logger.info(f"Prompt language set to '{lang}'")


def get_prompt_language() -> str:
    """Return the currently active prompt language code."""
    return _current_language


def reset_prompts() -> None:
    """Reset all prompts back to the default English templates."""
    global _current_language
    for key, value in _ENGLISH_PROMPTS.items():
        PROMPTS[key] = value
    _current_language = "en"
    logger.info("Prompts reset to English defaults")


def get_available_languages() -> list[str]:
    """Return a list of all registered language codes.

    Note: Languages that can be lazily loaded (like 'zh') may not appear
    here until they are first used or explicitly registered.
    """
    # Include known lazy-loadable languages
    all_langs = set(_PROMPT_LANGUAGES.keys()) | {"zh"}
    return sorted(all_langs)
