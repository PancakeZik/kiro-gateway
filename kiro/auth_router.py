# -*- coding: utf-8 -*-

"""
Multi-account auth routing.

Routes requests to different KiroAuthManager instances based on the requested model.
When MULTI_ACCOUNT_ROUTING is enabled, opus models use a dedicated account
and everything else uses the secondary account.
"""

from loguru import logger

from kiro.auth import KiroAuthManager
from kiro.config import PRIMARY_MODEL_PATTERNS


def is_primary_model(model: str) -> bool:
    """Check if the requested model should use the primary (opus) account."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in PRIMARY_MODEL_PATTERNS)


def resolve_auth_manager(
    model: str,
    primary_auth: KiroAuthManager,
    secondary_auth: KiroAuthManager,
) -> KiroAuthManager:
    """
    Pick the right auth manager based on the requested model.

    Args:
        model: The model name from the request
        primary_auth: Auth manager for the primary (opus) account
        secondary_auth: Auth manager for the secondary (everything else) account

    Returns:
        The appropriate KiroAuthManager for this request
    """
    if is_primary_model(model):
        logger.info(f"Multi-account routing: model={model} → primary account")
        return primary_auth
    logger.info(f"Multi-account routing: model={model} → secondary account")
    return secondary_auth
