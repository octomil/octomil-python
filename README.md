# EdgeML Python SDK

Federated Learning orchestration and client SDK for Python.

## Installation

```bash
pip install edgeml-sdk
```

## Quick Start

```python
from edgeml import FederatedClient

client = FederatedClient(
    auth_token_provider=lambda: "<short-lived-device-token>",
    org_id="org_123",
)

# Register this runtime as a device participant
device_id = client.register()
```

## Documentation

https://docs.edgeml.io/sdks/python

## Runtime Device Auth

Use backend-issued short-lived device tokens instead of embedding org API keys in clients.

```python
from edgeml import DeviceAuthClient

auth = DeviceAuthClient(
    base_url="https://api.edgeml.io",
    org_id="org_123",
    device_identifier="device-abc",
)

# 1) Bootstrap once with a backend-issued bootstrap bearer token.
await auth.bootstrap(bootstrap_bearer_token="token_from_backend")

# 2) Get access token for device API calls (auto-refreshes near expiry).
access_token = await auth.get_access_token()
```
