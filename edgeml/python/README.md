# EdgeML Python SDK

Python SDK for federated orchestration and device runtime participation.

## Enterprise Runtime Auth

Use backend-issued bootstrap tokens and short-lived device credentials for runtime clients.

```python
from edgeml import DeviceAuthClient

auth = DeviceAuthClient(
    base_url="https://api.edgeml.io",
    org_id="org_123",
    device_identifier="python-runtime-001",
)

await auth.bootstrap(bootstrap_bearer_token=backend_bootstrap_token)
access_token = await auth.get_access_token()
```

Pass an auth token provider into `FederatedClient`:

```python
from edgeml import FederatedClient

client = FederatedClient(
    auth_token_provider=lambda: auth.get_access_token_sync(),
    org_id="org_123",
)
```
