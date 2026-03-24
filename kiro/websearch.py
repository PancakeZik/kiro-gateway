# -*- coding: utf-8 -*-

"""
Web Search via Kiro MCP endpoint.

Detects web_search server tools in Anthropic requests, performs the search
via Kiro's MCP endpoint, and injects results back into the conversation.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _is_server_web_search_tool(tool: Dict[str, Any]) -> bool:
    """Check if a tool dict is an Anthropic server-side web_search tool.

    Server-side tools have type like "web_search_20250305" and no input_schema.
    Custom tools named "web_search" with an input_schema are NOT matched.
    """
    tool_type = tool.get("type", "")
    if tool_type.startswith("web_search"):
        return True
    # Fallback: name-based detection only if there's no input_schema
    # (server tools don't define one, custom tools do)
    name = tool.get("name", "")
    if name == "web_search" and not tool.get("input_schema"):
        return True
    return False


def find_web_search_tool(tools: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Find and return the web_search server tool from the tools list, or None."""
    if not tools:
        return None
    for tool in tools:
        if _is_server_web_search_tool(tool):
            return tool
    return None


def extract_web_search_tools(
    tools: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split tools into (web_search_tool, remaining_tools).

    Returns the web_search server tool separately so it can be handled
    via MCP, and the remaining tools for normal Kiro conversion.
    """
    ws_tool = None
    remaining = []
    for tool in tools:
        if ws_tool is None and _is_server_web_search_tool(tool):
            ws_tool = tool
        else:
            remaining.append(tool)
    return ws_tool, remaining


def extract_search_query(messages: List[Dict[str, Any]]) -> str:
    """Extract the search query from the last user message.

    Claude Code sends web_search requests with a single user message like:
      "Perform a web search for the query: <actual query>"
    """
    if not messages:
        return ""

    # Walk messages in reverse to find the last user message
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    break
        else:
            continue

        # Strip the known prefix
        prefix = "Perform a web search for the query: "
        if text.startswith(prefix):
            text = text[len(prefix):]
        return text.strip()

    return ""


# ---------------------------------------------------------------------------
# MCP Client
# ---------------------------------------------------------------------------

def _build_mcp_request(query: str) -> Tuple[str, str, Dict[str, Any]]:
    """Build an MCP tools/call request for web_search.

    Returns (request_id, tool_use_id, request_body).
    """
    random22 = uuid.uuid4().hex[:22]
    ts = int(time.time() * 1000)
    random8 = uuid.uuid4().hex[:8]

    request_id = f"web_search_tooluse_{random22}_{ts}_{random8}"
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:32]}"

    body = {
        "id": request_id,
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "web_search",
            "arguments": {
                "query": query,
                "_meta": {
                    "_isValid": True,
                    "_activePath": ["query"],
                    "_completedPaths": [["query"]],
                },
            },
        },
    }
    return request_id, tool_use_id, body


async def mcp_web_search(
    query: str,
    api_host: str,
    headers: Dict[str, str],
    http_client: Optional[httpx.AsyncClient] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Call Kiro MCP web_search and return (tool_use_id, parsed_results).

    Args:
        query: Search query string
        api_host: Kiro API host (e.g. https://q.us-east-1.amazonaws.com)
        headers: Auth headers (must include Authorization: Bearer ...)
        http_client: Optional shared httpx.AsyncClient

    Returns:
        (tool_use_id, results_dict) where results_dict has {results: [...], ...}
        or (tool_use_id, None) on failure.
    """
    _, tool_use_id, request_body = _build_mcp_request(query)
    mcp_url = f"{api_host}/mcp"

    logger.info(f"[websearch] MCP call: query='{query[:80]}' -> {mcp_url}")

    try:
        if http_client:
            resp = await http_client.post(
                mcp_url, json=request_body, headers=headers, timeout=30.0
            )
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    mcp_url, json=request_body, headers=headers, timeout=30.0
                )

        if resp.status_code != 200:
            logger.error(f"[websearch] MCP returned {resp.status_code}: {resp.text[:300]}")
            return tool_use_id, None

        data = resp.json()

        # Parse results from MCP response
        result = data.get("result")
        if not result or not result.get("content"):
            logger.warning("[websearch] MCP response has no result content")
            return tool_use_id, None

        content_text = result["content"][0].get("text", "")
        results = json.loads(content_text)
        result_count = len(results.get("results", []))
        logger.info(f"[websearch] Got {result_count} results for '{query[:50]}'")
        return tool_use_id, results

    except Exception as e:
        logger.error(f"[websearch] MCP call failed: {e}")
        return tool_use_id, None


# ---------------------------------------------------------------------------
# Payload injection (Anthropic format)
# ---------------------------------------------------------------------------

def inject_search_into_messages(
    messages: List[Dict[str, Any]],
    tool_use_id: str,
    query: str,
    results: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Append assistant tool_use + user tool_result messages with search results.

    This makes the model see the search results as context in the conversation.
    """
    messages = list(messages)  # shallow copy

    # Assistant message: "I used web_search"
    messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": "web_search",
                "input": {"query": query},
            }
        ],
    })

    # User message: tool_result with search results + guidance
    result_text = _format_result_text(results)
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    guidance = (
        f"<search_guidance>\n"
        f"Current date: {now.strftime('%B %d, %Y')} ({now.strftime('%A')})\n\n"
        f"Evaluate the search results above. If they are mostly spam, "
        f"unrelated, or missing information about the query topic, "
        f"use the web_search tool again with a refined query.\n"
        f"</search_guidance>"
    )

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_text,
            },
            {
                "type": "text",
                "text": guidance,
            },
        ],
    })

    return messages


def _format_result_text(results: Optional[Dict[str, Any]]) -> str:
    """Format search results as text for the tool_result content."""
    if not results or not results.get("results"):
        return "No search results found."

    items = results["results"]
    text = f"Found {len(items)} search result(s):\n\n"
    text += json.dumps(items, indent=2, ensure_ascii=False)
    return text


# ---------------------------------------------------------------------------
# Response injection (SSE stream prefix)
# ---------------------------------------------------------------------------

def build_search_indicator_events(
    tool_use_id: str,
    query: str,
    results: Optional[Dict[str, Any]],
    start_index: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Build content blocks for server_tool_use + web_search_tool_result.

    These get prepended to the response so Claude Code shows
    "Searched N times" in the UI.

    Returns (content_blocks, next_index).
    """
    blocks = []

    # server_tool_use block
    blocks.append({
        "type": "server_tool_use",
        "id": tool_use_id,
        "name": "web_search",
        "input": {"query": query},
    })

    # web_search_tool_result block
    search_content = []
    if results and results.get("results"):
        for r in results["results"]:
            search_content.append({
                "type": "web_search_result",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "encrypted_content": r.get("snippet", ""),
                "page_age": None,
            })

    blocks.append({
        "type": "web_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": search_content,
    })

    next_index = start_index + len(blocks)
    return blocks, next_index
