# Kiro Gateway: Endpoint Migration Guide (May 15, 2026)

## What's Changing

AWS is retiring the legacy `q.{region}.amazonaws.com` endpoint on May 15, 2026. All Kiro API traffic must move to new `kiro.dev` endpoints:

| Old Endpoint | New Endpoint | Purpose |
|---|---|---|
| `q.{region}.amazonaws.com` | `runtime.{region}.kiro.dev` | Inference, streaming |
| (new) | `management.{region}.kiro.dev` | Configuration, lifecycle |
| (new) | `telemetry.{region}.kiro.dev` | Metrics, observability |

Supported regions: `us-east-1`, `eu-central-1`.

Auth/SSO endpoints (`oidc.{region}.amazonaws.com`, `prod.{region}.auth.desktop.kiro.dev`) are NOT changing.

## Gateway Changes Required

### 1. Update endpoint templates in `kiro/config.py`

```python
# Before
KIRO_API_HOST_TEMPLATE: str = "https://q.{region}.amazonaws.com"
KIRO_Q_HOST_TEMPLATE: str = "https://q.{region}.amazonaws.com"

# After
KIRO_API_HOST_TEMPLATE: str = "https://runtime.{region}.kiro.dev"
KIRO_Q_HOST_TEMPLATE: str = "https://runtime.{region}.kiro.dev"
```

### 2. Send profileArn for all auth types

The new endpoint requires `profileArn` in the request payload for ALL auth types, including AWS SSO OIDC (kiro-cli / IDC). The old endpoint did not enforce this for SSO users.

In `kiro/routes_anthropic.py` and `kiro/routes_openai.py`, the current logic skips profileArn for SSO:

```python
# Current (only sends for Kiro Desktop)
profile_arn_for_payload = ""
if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
    profile_arn_for_payload = auth_manager.profile_arn
```

This must be updated to send profileArn for all auth types when available:

```python
# New (send for all auth types)
profile_arn_for_payload = auth_manager.profile_arn or ""
```

**Important:** SSO/IDC users currently don't have a profileArn stored in the kiro-cli SQLite database. You must update kiro-cli to a version that provides it. Without a valid profileArn, the new endpoint returns:

```json
{"message": "profileArn is required for this request.", "reason": null}
```

### 3. Update kiro-cli

Before switching endpoints, update kiro-cli to the latest version. Newer versions are expected to store a `profile_arn` in the SQLite database (`~/.local/share/kiro-cli/data.sqlite3`).

### 4. Firewall / network allowlist

If your network has outbound firewall rules, add `*.kiro.dev` (or the specific subdomains listed above).

## What We Tested (April 21, 2026)

Using a live IDC token and the gateway's real payload format:

| Endpoint | Without profileArn | With profileArn |
|---|---|---|
| `q.us-east-1.amazonaws.com` | **200 OK** (works) | 403 (breaks SSO) |
| `runtime.us-east-1.kiro.dev` | 400 "profileArn is required" | 403 "bearer token invalid"* |

\* The 403 with profileArn on the new endpoint was because we used a profileArn from a different account. With the correct profileArn (from an updated kiro-cli), this should return 200.

**Key finding:** The old endpoint works for SSO users WITHOUT profileArn and BREAKS if you send one. The new endpoint REQUIRES profileArn. This means you cannot incrementally migrate — the endpoint switch and profileArn change must happen together.

## Migration Checklist

- [ ] Update kiro-cli to latest version (must provide profileArn for SSO users)
- [ ] Verify profileArn is present in SQLite: `sqlite3 ~/.local/share/kiro-cli/data.sqlite3 "SELECT value FROM auth_kv WHERE key='kirocli:odic:token'" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('profile_arn', 'MISSING'))"`
- [ ] Update `KIRO_API_HOST_TEMPLATE` and `KIRO_Q_HOST_TEMPLATE` in `kiro/config.py`
- [ ] Update profileArn logic in route handlers to send for all auth types
- [ ] Update firewall rules to allow `*.kiro.dev`
- [ ] Restart gateway and test
- [ ] Revert if broken: `git checkout kiro/config.py kiro/routes_anthropic.py kiro/routes_openai.py && restart gateway`

## Usage Implications

The profileArn requirement likely enables per-subscription usage tracking. SSO/IDC users who previously had unmetered access (because the old endpoint couldn't identify their subscription without profileArn) will likely have usage limits enforced after migration.

## References

- AWS Health Event: endpoint deactivation notice (April 2026)
- Kiro firewall docs: https://kiro.dev/docs/privacy-and-security/firewalls/
- Kiro CLI firewall docs: https://kiro.dev/docs/cli/privacy-and-security/firewalls/#core-urls
