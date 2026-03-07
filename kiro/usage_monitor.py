# -*- coding: utf-8 -*-

"""
Usage monitor for Kiro Gateway.

Polls the GetUsageLimits API to track account credit consumption.
Logs colored warnings at escalating thresholds (50%, 80%, 90%, 95%).
Uses adaptive polling: checks more frequently as usage climbs.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx
from loguru import logger

from kiro.auth import KiroAuthManager, AuthType


# ANSI color codes for direct stderr output
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


# Adaptive polling thresholds: (usage_pct, check_every_n_requests)
# As usage climbs, we check more often
ADAPTIVE_THRESHOLDS = [
    (0.95, 5),    # >95%: every 5 requests
    (0.90, 10),   # >90%: every 10 requests
    (0.80, 15),   # >80%: every 15 requests
    (0.50, 25),   # >50%: every 25 requests
    (0.0,  50),   # <50%: every 50 requests
]

# Warning thresholds: (pct, label, color)
WARNING_LEVELS = [
    (0.95, "CRITICAL", RED),
    (0.90, "HIGH",     RED),
    (0.80, "WARNING",  YELLOW),
    (0.50, "MODERATE", YELLOW),
]

# Minimum interval between polls (seconds) to avoid hammering the API
MIN_POLL_INTERVAL = 60


@dataclass
class UsageInfo:
    """Cached usage data for a single account."""
    credits_used: float = 0.0
    credits_limit: float = 0.0
    usage_pct: float = 0.0
    plan_name: str = "Unknown"
    next_reset: Optional[str] = None
    days_until_reset: Optional[int] = None
    overage_enabled: bool = False
    last_checked: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class AccountMonitor:
    """Tracks polling state for a single account."""
    name: str
    auth_manager: KiroAuthManager
    usage: UsageInfo = field(default_factory=UsageInfo)
    request_count: int = 0
    request_count_at_last_check: int = 0
    last_warned_level: float = 0.0  # highest pct level we've warned about
    disabled: bool = False  # stop polling after permanent errors (e.g., no subscription)


async def fetch_usage_limits(auth_manager: KiroAuthManager) -> UsageInfo:
    """
    Calls GetUsageLimits on the CodeWhisperer API.

    Returns a UsageInfo with the parsed response, or with error set on failure.
    """
    info = UsageInfo()

    try:
        token = await auth_manager.get_access_token()

        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "x-amz-target": "AmazonCodeWhispererService.GetUsageLimits",
            "Authorization": f"Bearer {token}",
        }

        body: dict = {}
        if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
            body["profileArn"] = auth_manager.profile_arn

        api_host = auth_manager.api_host

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{api_host}/",
                headers=headers,
                json=body,
            )

            if resp.status_code != 200:
                # Friendly message for common errors
                try:
                    err_data = resp.json()
                    reason = err_data.get("reason") or err_data.get("message") or ""
                    if "FEATURE_NOT_SUPPORTED" in reason:
                        info.error = "no usage tracking (account has no Kiro subscription)"
                    else:
                        info.error = f"HTTP {resp.status_code}: {reason or resp.text[:200]}"
                except Exception:
                    info.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return info

            data = resp.json()

        # Debug: log raw response to help tune parsing
        logger.debug(f"GetUsageLimits raw response: {data}")

        # Parse subscription info
        sub = data.get("subscriptionInfo") or {}
        info.plan_name = sub.get("subscriptionTitle") or sub.get("subscriptionName") or sub.get("type") or "Unknown"

        # Parse usage breakdown — look for the aggregate or first entry
        breakdown = data.get("usageBreakdown") or {}
        breakdown_list = data.get("usageBreakdownList") or []

        # Try aggregate breakdown first, then sum from list
        if breakdown:
            info.credits_used = _extract_credits_used(breakdown)
            info.credits_limit = _extract_credits_limit(breakdown)
        elif breakdown_list:
            for entry in breakdown_list:
                info.credits_used += _extract_credits_used(entry)
                info.credits_limit = max(info.credits_limit, _extract_credits_limit(entry))

        # Fallback: parse from limits list
        if info.credits_limit == 0:
            for limit_entry in data.get("limits", []):
                used = limit_entry.get("used") or limit_entry.get("currentUsage") or 0
                total = limit_entry.get("limit") or limit_entry.get("maxUsage") or 0
                if isinstance(used, (int, float)) and isinstance(total, (int, float)):
                    info.credits_used = max(info.credits_used, float(used))
                    info.credits_limit = max(info.credits_limit, float(total))

        # Calculate percentage
        if info.credits_limit > 0:
            info.usage_pct = info.credits_used / info.credits_limit
        else:
            info.usage_pct = 0.0

        # Reset date
        reset_ts = data.get("nextDateReset")
        if reset_ts:
            try:
                reset_dt = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
                info.next_reset = reset_dt.strftime("%Y-%m-%d")
                # Calculate days ourselves since API sometimes returns 0
                days = (reset_dt - datetime.now(timezone.utc)).days
                info.days_until_reset = max(0, days)
            except Exception:
                info.next_reset = str(reset_ts)

        # Fallback to API value if we didn't calculate
        if info.days_until_reset is None:
            info.days_until_reset = data.get("daysUntilReset")

        # Overage
        overage = data.get("overageConfiguration") or {}
        overage_status = overage.get("overageStatus") or overage.get("status") or ""
        info.overage_enabled = overage_status.lower() in ("enabled", "on", "active")

        info.last_checked = datetime.now(timezone.utc)

    except Exception as e:
        info.error = str(e)

    return info


def _extract_credits_used(breakdown: dict) -> float:
    """Extract credits used from a usage breakdown entry."""
    for key in ("currentUsageWithPrecision", "currentUsage", "creditsUsed", "used", "totalUsage"):
        val = breakdown.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _extract_credits_limit(breakdown: dict) -> float:
    """Extract credits limit from a usage breakdown entry."""
    for key in ("usageLimitWithPrecision", "usageLimit", "creditsLimit", "limit", "maxUsage", "totalLimit", "includedCredits"):
        val = breakdown.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _get_poll_interval(usage_pct: float) -> int:
    """Return how many requests between polls based on current usage."""
    for threshold_pct, interval in ADAPTIVE_THRESHOLDS:
        if usage_pct >= threshold_pct:
            return interval
    return ADAPTIVE_THRESHOLDS[-1][1]


def _format_usage_log(account: AccountMonitor) -> str:
    """Format a colored usage log line for stderr."""
    u = account.usage

    if u.error:
        return f"{DIM}📊 Usage [{account.name}]: ⚠ check failed — {u.error}{RESET}"

    pct = u.usage_pct * 100
    bar_width = 30
    filled = int(bar_width * u.usage_pct)
    empty = bar_width - filled

    # Pick color based on usage
    if pct >= 90:
        bar_color = RED
    elif pct >= 80:
        bar_color = YELLOW
    elif pct >= 50:
        bar_color = CYAN
    else:
        bar_color = GREEN

    bar = f"{bar_color}{'█' * filled}{DIM}{'░' * empty}{RESET}"

    reset_info = ""
    if u.next_reset:
        reset_info = f" | resets {u.next_reset}"
        if u.days_until_reset is not None:
            reset_info += f" ({u.days_until_reset}d)"

    overage = ""
    if u.overage_enabled:
        overage = f" | {YELLOW}overages ON{RESET}"
    else:
        overage = f" | overages off"

    return (
        f"{MAGENTA}{BOLD}📊 Usage [{account.name}]{RESET} "
        f"{bar} "
        f"{bar_color}{BOLD}{pct:.1f}%{RESET} "
        f"({u.credits_used:.1f}/{u.credits_limit:.0f} credits)"
        f"{DIM} | {u.plan_name}{reset_info}{overage}{RESET}"
    )


def _format_warning(account: AccountMonitor, level_label: str, color: str) -> str:
    """Format a colored warning line."""
    u = account.usage
    pct = u.usage_pct * 100
    remaining = u.credits_limit - u.credits_used

    return (
        f"{color}{BOLD}⚠️  USAGE {level_label} [{account.name}]: "
        f"{pct:.1f}% used — {remaining:.1f} credits remaining{RESET}"
    )


def log_usage(account: AccountMonitor) -> None:
    """Log usage status and any warnings to stderr via loguru."""
    import sys

    # Always print the status bar
    print(_format_usage_log(account), file=sys.stderr)

    # Check if we need to warn at a new level
    u = account.usage
    if u.error:
        return

    for threshold, label, color in WARNING_LEVELS:
        if u.usage_pct >= threshold and threshold > account.last_warned_level:
            print(_format_warning(account, label, color), file=sys.stderr)
            account.last_warned_level = threshold
            break


async def _monitor_loop(accounts: list[AccountMonitor], stop_event: asyncio.Event) -> None:
    """
    Background loop that checks usage based on request count changes.

    Wakes up every MIN_POLL_INTERVAL seconds and checks if enough requests
    have happened since the last poll to warrant a new usage check.
    """
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=MIN_POLL_INTERVAL)
            break  # stop_event was set
        except asyncio.TimeoutError:
            pass  # normal wake-up

        for acct in accounts:
            if acct.disabled:
                continue
            requests_since = acct.request_count - acct.request_count_at_last_check
            poll_interval = _get_poll_interval(acct.usage.usage_pct)

            if requests_since >= poll_interval:
                acct.usage = await fetch_usage_limits(acct.auth_manager)
                acct.request_count_at_last_check = acct.request_count
                log_usage(acct)


class UsageMonitor:
    """
    Manages background usage monitoring for one or more Kiro accounts.

    Usage:
        monitor = UsageMonitor()
        monitor.add_account("primary", auth_manager_primary)
        monitor.add_account("haiku", auth_manager_haiku)
        await monitor.start()   # initial check + background loop
        ...
        monitor.increment("primary")  # call on each request
        ...
        await monitor.stop()
    """

    def __init__(self):
        self._accounts: dict[str, AccountMonitor] = {}
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def add_account(self, name: str, auth_manager: KiroAuthManager) -> None:
        """Register an account to monitor."""
        self._accounts[name] = AccountMonitor(name=name, auth_manager=auth_manager)

    def increment(self, account_name: str) -> None:
        """Increment request counter for an account. Call from route handlers."""
        acct = self._accounts.get(account_name)
        if acct:
            acct.request_count += 1

    def get_usage(self, account_name: str) -> Optional[UsageInfo]:
        """Get cached usage info for an account."""
        acct = self._accounts.get(account_name)
        return acct.usage if acct else None

    async def start(self) -> None:
        """Run initial usage check for all accounts, then start background loop."""
        import sys

        print(f"\n{MAGENTA}{BOLD}📊 Checking account usage...{RESET}", file=sys.stderr)

        for acct in self._accounts.values():
            acct.usage = await fetch_usage_limits(acct.auth_manager)
            # Disable polling for accounts with permanent errors (no subscription)
            if acct.usage.error and "no usage tracking" in (acct.usage.error or ""):
                acct.disabled = True
            log_usage(acct)

        print(file=sys.stderr)

        # Start background monitor
        account_list = list(self._accounts.values())
        self._task = asyncio.create_task(
            _monitor_loop(account_list, self._stop_event)
        )

    async def stop(self) -> None:
        """Stop the background monitor."""
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
