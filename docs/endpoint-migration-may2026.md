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

### 2. Fetch and send profileArn for all auth types

The new endpoint requires `profileArn` in the request payload for ALL auth types. The profileArn is NOT stored in kiro-cli's SQLite DB — it must be fetched at runtime via the management API:

```python
# On startup: fetch profileArn from management API
POST https://management.{region}.kiro.dev/listAvailableProfiles
Authorization: Bearer <access_token>
Body: {}
# Returns: {"profiles": [{"arn": "arn:aws:codewhisperer:..."}]}
```

In `kiro/routes_anthropic.py` and `kiro/routes_openai.py`, update to always include the fetched profileArn:

```python
# New (always send fetched profileArn)
profile_arn_for_payload = auth_manager.profile_arn or ""
```

### 3. Update model fetching endpoint

The `ListAvailableModels` API moved from `q.amazonaws.com` to `management.kiro.dev` and now requires profileArn:

```python
# Before
GET https://q.{region}.amazonaws.com/ListAvailableModels?origin=AI_EDITOR

# After
GET https://management.{region}.kiro.dev/ListAvailableModels?origin=AI_EDITOR&profileArn=<arn>
```

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

## What We Tested (May 6, 2026)

### Discovery: profileArn can be fetched via management API

The kiro-cli (v2.2.1) still does NOT store profileArn in the SQLite database. However, we discovered that the profileArn can be obtained at runtime via a management API call:

```bash
# Fetch profileArn (requires Bearer token from SSO OIDC auth)
POST https://management.us-east-1.kiro.dev/listAvailableProfiles
Authorization: Bearer <access_token>
Content-Type: application/json
Body: {}

# Response:
{
  "profiles": [
    {
      "arn": "arn:aws:codewhisperer:us-east-1:XXXXXXXXX:profile/XXXXXXXXXXX",
      "profileName": "KiroProfile-us-east-1",
      "profileType": "Q_DEVELOPER",
      "status": "ACTIVE"
    }
  ]
}
```

### Confirmed: runtime.kiro.dev works with fetched profileArn

```bash
# Inference (streaming) — returns 200 with event-stream
POST https://runtime.us-east-1.kiro.dev/generateAssistantResponse
Authorization: Bearer <access_token>
Content-Type: application/json
Body: { "conversationState": {...}, "profileArn": "<arn from above>" }
```

### Discovery: ListAvailableModels moved to management endpoint

The old endpoint served models from `q.amazonaws.com/ListAvailableModels`. On the new infrastructure:

```bash
# Models list (requires profileArn as query param)
GET https://management.us-east-1.kiro.dev/ListAvailableModels?origin=AI_EDITOR&profileArn=<arn>
Authorization: Bearer <access_token>

# Without profileArn → 400 "Invalid profileArn"
# On runtime.kiro.dev → 404 (not available there)
```

### Additional management APIs discovered

```bash
# Profile details
POST https://management.us-east-1.kiro.dev/getProfile
Body: { "profileArn": "<arn>" }

# ListModels (different from ListAvailableModels) — requires SigV4, not Bearer
# Returns 403 with Bearer token
```

### New endpoint mapping (complete)

| Operation | Old Endpoint | New Endpoint |
|---|---|---|
| Inference (streaming) | `q.{region}.amazonaws.com/generateAssistantResponse` | `runtime.{region}.kiro.dev/generateAssistantResponse` |
| List models | `q.{region}.amazonaws.com/ListAvailableModels` | `management.{region}.kiro.dev/ListAvailableModels` |
| List profiles | N/A | `management.{region}.kiro.dev/listAvailableProfiles` |
| Get profile | N/A | `management.{region}.kiro.dev/getProfile` |

### Models available (May 6, 2026)

| Model | Rate Multiplier | Max Input | Prompt Caching |
|---|---|---|---|
| auto | 1.0x | 1M | Yes (4 checkpoints, 1024 min) |
| claude-opus-4.6 | 2.2x | 1M | Yes (4 checkpoints, 4096 min) |
| claude-sonnet-4.6 | 1.3x | 1M | Yes (4 checkpoints, 1024 min) |
| claude-opus-4.5 | 2.2x | 200k | Yes (4 checkpoints, 4096 min) |
| claude-sonnet-4.5 | 1.3x | 200k | Yes (4 checkpoints, 1024 min) |
| claude-sonnet-4 | 1.3x | 200k | Yes (4 checkpoints, 1024 min) |
| claude-haiku-4.5 | 0.4x | 200k | Yes (4 checkpoints, 4096 min) |
| deepseek-3.2 | 0.25x | 164k | No |
| minimax-m2.5 | 0.25x | 196k | No |
| minimax-m2.1 | 0.15x | 196k | No |
| glm-5 | 0.5x | 200k | No |
| qwen3-coder-next | 0.05x | 256k | No |

All models output max 64k tokens.

## Migration Checklist

- [ ] Update `KIRO_API_HOST_TEMPLATE` in `kiro/config.py` → `https://runtime.{region}.kiro.dev`
- [ ] Update `KIRO_Q_HOST_TEMPLATE` in `kiro/config.py` → `https://management.{region}.kiro.dev`
- [ ] Add `fetch_profile_arn()` to auth manager: call `management.kiro.dev/listAvailableProfiles` on startup, cache the ARN
- [ ] Update profileArn logic in route handlers to ALWAYS include the fetched profileArn (not just for KIRO_DESKTOP)
- [ ] Update firewall rules to allow `*.kiro.dev`
- [ ] Restart gateway and test
- [ ] Revert if broken: `git checkout kiro/config.py kiro/routes_anthropic.py kiro/routes_openai.py && restart gateway`

## Usage Implications

The profileArn requirement enables per-subscription usage tracking. SSO/IDC users who previously had unmetered access (because the old endpoint couldn't identify their subscription without profileArn) will have usage limits enforced after migration. The old endpoint without profileArn does NOT count against quota.

## References

- AWS Health Event: endpoint deactivation notice (April 2026)
- Kiro firewall docs: https://kiro.dev/docs/privacy-and-security/firewalls/
- Kiro CLI firewall docs: https://kiro.dev/docs/cli/privacy-and-security/firewalls/#core-urls
