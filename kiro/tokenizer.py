# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Module for fast token counting.

Uses tiktoken (OpenAI's Rust library) for approximate
token counting. The cl100k_base encoding is close to Claude tokenization.

Note: This is an approximate count, as the exact Claude tokenizer
is not public. Anthropic does not publish their tokenizer,
so tiktoken with a correction coefficient is used.

The correction coefficient CLAUDE_CORRECTION_FACTOR = 1.15 is based on
empirical observations: Claude tokenizes text approximately 15%
more than GPT-4 (cl100k_base). This is due to differences in BPE vocabularies.
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger

# Lazy loading of tiktoken to speed up import
_encoding = None

# Correction coefficient for Claude models
# Claude tokenizes text approximately 15% more than GPT-4 (cl100k_base)
# This is an empirical value based on comparison with context_usage from API
CLAUDE_CORRECTION_FACTOR = 1.15

# Estimated tokens per image for Claude models.
# Claude counts images by pixel dimensions: tokens = (width * height) / 750.
# A max-size image (1568x1568) costs ~3,277 tokens. We use 3,000 as a
# conservative estimate since we don't know actual dimensions at this layer.
IMAGE_TOKEN_ESTIMATE = 3000

# Regex to match base64 image data in serialized Kiro JSON payloads.
# Matches "bytes": "..." where the value is a long base64 string.
_BASE64_BYTES_PATTERN = re.compile(r'"bytes"\s*:\s*"[A-Za-z0-9+/=]{100,}"')


def _get_encoding():
    """
    Lazy initialization of tokenizer.

    Uses cl100k_base - encoding for GPT-4/ChatGPT,
    which is close enough to Claude tokenization.

    Returns:
        tiktoken.Encoding or None if tiktoken is unavailable
    """
    global _encoding
    if _encoding is None:
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("cl100k_base")
            logger.debug("[Tokenizer] Initialized tiktoken with cl100k_base encoding")
        except ImportError:
            logger.warning(
                "[Tokenizer] tiktoken not installed. "
                "Token counting will use fallback estimation. "
                "Install with: pip install tiktoken"
            )
            _encoding = False  # Marker that import failed
        except Exception as e:
            logger.error(f"[Tokenizer] Failed to initialize tiktoken: {e}")
            _encoding = False
    return _encoding if _encoding else None


def count_tokens(text: str, apply_claude_correction: bool = True) -> int:
    """
    Counts the number of tokens in text.

    Args:
        text: Text to count tokens for
        apply_claude_correction: Apply correction coefficient for Claude (default True)

    Returns:
        Number of tokens (approximate, with Claude correction)
    """
    if not text:
        return 0

    encoding = _get_encoding()
    if encoding:
        try:
            base_tokens = len(encoding.encode(text))
            if apply_claude_correction:
                return int(base_tokens * CLAUDE_CORRECTION_FACTOR)
            return base_tokens
        except Exception as e:
            logger.warning(f"[Tokenizer] Error encoding text: {e}")

    # Fallback: rough estimate ~4 characters per token for English,
    # ~2-3 characters for other languages (taking average ~3.5)
    # For Claude we add correction
    base_estimate = len(text) // 4 + 1
    if apply_claude_correction:
        return int(base_estimate * CLAUDE_CORRECTION_FACTOR)
    return base_estimate


def count_payload_tokens(
    payload_text: str, apply_claude_correction: bool = True
) -> int:
    """
    Counts tokens in a serialized Kiro JSON payload, handling images correctly.

    Base64 image data in the payload is stripped before tokenization and replaced
    with a fixed per-image token estimate (IMAGE_TOKEN_ESTIMATE). This prevents
    massive over-counting — a single screenshot's base64 can be 200KB+ which
    tiktoken would count as ~50k-200k tokens, but Claude actually counts images
    by pixel dimensions (~1,600 tokens max for a large image).

    Args:
        payload_text: Serialized JSON payload as string
        apply_claude_correction: Apply correction coefficient for Claude

    Returns:
        Approximate number of tokens
    """
    if not payload_text:
        return 0

    # Count and strip base64 image data
    image_count = len(_BASE64_BYTES_PATTERN.findall(payload_text))
    if image_count > 0:
        stripped = _BASE64_BYTES_PATTERN.sub('"bytes": "<image>"', payload_text)
        logger.debug(
            f"[Tokenizer] Stripped {image_count} base64 image(s) from payload for token counting"
        )
    else:
        stripped = payload_text

    text_tokens = count_tokens(
        stripped, apply_claude_correction=apply_claude_correction
    )
    image_tokens = image_count * IMAGE_TOKEN_ESTIMATE

    return text_tokens + image_tokens


def count_message_tokens(
    messages: List[Dict[str, Any]], apply_claude_correction: bool = True
) -> int:
    """
    Counts tokens in a list of chat messages.

    Accounts for OpenAI/Claude message structure:
    - role: ~1 token
    - content: text tokens
    - Service tokens between messages: ~3-4 tokens

    Args:
        messages: List of messages in OpenAI format
        apply_claude_correction: Apply correction coefficient for Claude

    Returns:
        Approximate number of tokens (with Claude correction)
    """
    if not messages:
        return 0

    total_tokens = 0

    for message in messages:
        # Base tokens per message (role, delimiters)
        total_tokens += 4  # ~4 tokens for service information

        # Role tokens (without correction, these are short strings)
        role = message.get("role", "")
        total_tokens += count_tokens(role, apply_claude_correction=False)

        # Content tokens
        content = message.get("content")
        if content:
            if isinstance(content, str):
                total_tokens += count_tokens(content, apply_claude_correction=False)
            elif isinstance(content, list):
                # Multimodal content (text + images)
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total_tokens += count_tokens(
                                item.get("text", ""), apply_claude_correction=False
                            )
                        elif item.get("type") == "image_url":
                            # Images take ~85-170 tokens depending on size
                            total_tokens += 100  # Average estimate

        # tool_calls tokens (if present)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                total_tokens += 4  # Service tokens
                func = tc.get("function", {})
                total_tokens += count_tokens(
                    func.get("name", ""), apply_claude_correction=False
                )
                total_tokens += count_tokens(
                    func.get("arguments", ""), apply_claude_correction=False
                )

        # tool_call_id tokens (for tool responses)
        if message.get("tool_call_id"):
            total_tokens += count_tokens(
                message["tool_call_id"], apply_claude_correction=False
            )

    # Final service tokens
    total_tokens += 3

    # Apply correction to total count
    if apply_claude_correction:
        return int(total_tokens * CLAUDE_CORRECTION_FACTOR)
    return total_tokens


def count_tools_tokens(
    tools: Optional[List[Dict[str, Any]]], apply_claude_correction: bool = True
) -> int:
    """
    Counts tokens in tool definitions.

    Args:
        tools: List of tools in OpenAI format
        apply_claude_correction: Apply correction coefficient for Claude

    Returns:
        Approximate number of tokens (with Claude correction)
    """
    if not tools:
        return 0

    total_tokens = 0

    for tool in tools:
        total_tokens += 4  # Service tokens

        if tool.get("type") == "function":
            func = tool.get("function", {})

            # Function name
            total_tokens += count_tokens(
                func.get("name", ""), apply_claude_correction=False
            )

            # Function description
            total_tokens += count_tokens(
                func.get("description", ""), apply_claude_correction=False
            )

            # Parameters (JSON schema)
            params = func.get("parameters")
            if params:
                import json

                params_str = json.dumps(params, ensure_ascii=False)
                total_tokens += count_tokens(params_str, apply_claude_correction=False)

    # Apply correction to total count
    if apply_claude_correction:
        return int(total_tokens * CLAUDE_CORRECTION_FACTOR)
    return total_tokens


def estimate_request_tokens(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, int]:
    """
    Estimates total number of tokens in request.

    Args:
        messages: List of messages
        tools: List of tools (optional)
        system_prompt: System prompt (optional, if not in messages)

    Returns:
        Dictionary with token breakdown:
        - messages_tokens: message tokens
        - tools_tokens: tool tokens
        - system_tokens: system prompt tokens
        - total_tokens: total count
    """
    messages_tokens = count_message_tokens(messages)
    tools_tokens = count_tools_tokens(tools)
    system_tokens = count_tokens(system_prompt) if system_prompt else 0

    return {
        "messages_tokens": messages_tokens,
        "tools_tokens": tools_tokens,
        "system_tokens": system_tokens,
        "total_tokens": messages_tokens + tools_tokens + system_tokens,
    }
