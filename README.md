# Octomil Python SDK

Python SDK for federated orchestration and device runtime participation.

## Enterprise Runtime Auth (required for device-side/runtime use)

Do not embed org API keys in distributed clients. Use backend-issued bootstrap tokens and short-lived device credentials.

**Server endpoints**
- `POST /api/v1/device-auth/bootstrap`
- `POST /api/v1/device-auth/refresh`
- `POST /api/v1/device-auth/revoke`

**Default lifetimes**
- Access token: 15 minutes (configurable, max 60 minutes)
- Refresh token: 30 days (rotated on refresh)

## Installation

```bash
pip install octomil-sdk
```

## Quick Start (Enterprise Runtime Auth)

```python
from octomil import DeviceAuthClient, FederatedClient

auth = DeviceAuthClient(
    base_url="https://api.octomil.io",
    org_id="org_123",
    device_identifier="python-runtime-001",
)

# One-time bootstrap with backend-issued token
await auth.bootstrap(bootstrap_bearer_token=backend_bootstrap_token)

client = FederatedClient(
    auth_token_provider=lambda: auth.get_access_token_sync(),
    org_id="org_123",
)

device_id = client.register()
```

## Token lifecycle

- `DeviceAuthClient.get_access_token()` auto-refreshes near expiry.
- `DeviceAuthClient.revoke()` invalidates the session.
- Use `keyring` for system keychain/keyring storage.

## Docs

- https://docs.octomil.io/sdks/python
- https://docs.octomil.io/reference/api-endpoints
