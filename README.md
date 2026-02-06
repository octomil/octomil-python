# Octomil Python SDK

Federated Learning orchestration and client SDK for Python.

## Installation

```bash
pip install octomil-sdk
```

## Quick Start

```python
from octomil import Octomil

client = Octomil(api_key="your-api-key")

# Create rollout
rollout = client.rollouts.create(
    model_id="model-123",
    version="2.0.0",
    rollout_percentage=10
)
```

## Documentation

https://docs.octomil.io/sdks/python

## Runtime Device Auth

Use backend-issued short-lived device tokens instead of embedding org API keys in clients.

```python
from octomil import DeviceAuthClient

auth = DeviceAuthClient(
    base_url="https://api.octomil.io",
    org_id="org_123",
    device_identifier="device-abc",
)

# 1) Bootstrap once with a backend-issued bootstrap bearer token.
await auth.bootstrap(bootstrap_bearer_token="token_from_backend")

# 2) Get access token for device API calls (auto-refreshes near expiry).
access_token = await auth.get_access_token()
```
