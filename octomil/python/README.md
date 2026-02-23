# Octomil Python SDK

Python SDK for federated orchestration and device runtime participation.

## Enterprise Runtime Auth

Use backend-issued bootstrap tokens and short-lived device credentials for runtime clients.

```python
from octomil import DeviceAuthClient

auth = DeviceAuthClient(
    base_url="https://api.octomil.com",
    org_id="org_123",
    device_identifier="python-runtime-001",
)

await auth.bootstrap(bootstrap_bearer_token=backend_bootstrap_token)
access_token = await auth.get_access_token()
```

Pass an auth token provider into `FederatedClient`:

````python
from octomil import FederatedClient

client = FederatedClient(
    auth_token_provider=lambda: auth.get_access_token_sync(),
    org_id="org_123",
)

Control-plane management is also available:

```python
from octomil import Octomil

edge = Octomil(
    auth_token_provider=lambda: auth.get_access_token_sync(),
    org_id="org_123",
)

# Model + version management
model = edge.registry.ensure_model(
    name="keyboard-next-word",
    framework="pytorch",
    use_case="text_classification",
)
edge.registry.publish_version(model["id"], "1.0.0")

# Rollout / deployment
edge.rollouts.create(
    model_id=model["id"],
    version="1.0.0",
    rollout_percentage=10,
    target_percentage=100,
    increment_step=10,
    start_immediately=True,
)

# Experiments + analytics
exp = edge.experiments.create(
    name="keyboard-ab-v1-v2",
    model_id=model["id"],
    control_version="1.0.0",
    treatment_version="1.1.0",
)
edge.experiments.start(exp["id"])
stats = edge.experiments.get_analytics(exp["id"])
````

```

```
