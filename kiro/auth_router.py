# -*- coding: utf-8 -*-

"""
Multi-account auth routing.

Routes requests to different KiroAuthManager instances based on the requested model.
When MULTI_ACCOUNT_ROUTING is enabled, haiku models use a separate account
from everything else (primary account).
"""

from loguru import logger

from kiro.auth import KiroAuthManager
from kiro.config import HAIKU_MODEL_PATTERNS


def is_haiku_model(model: str) -> bool:
    """Check if the requested model should use the haiku account."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in HAIKU_MODEL_PATTERNS)


def resolve_auth_manager(
    model: str,
    primary_auth: KiroAuthManager,
    haiku_auth: KiroAuthManager,
) -> KiroAuthManager:
    """
    Pick the right auth manager based on the requested model.

    Args:
        model: The model name from the request
        primary_auth: Auth manager for the primary (opus/sonnet/etc.) account
        haiku_auth: Auth manager for the haiku account

    Returns:
        The appropriate KiroAuthManager for this request
    """
    if is_haiku_model(model):
        logger.info(f"Multi-account routing: model={model} → haiku account")
        return haiku_auth
    logger.info(f"Multi-account routing: model={model} → primary account")
    return primary_auth
