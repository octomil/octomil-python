# Deployment Routing

## How it works

When an `OctomilClient` fetches desired state via `control.get_desired_state()`, deployment routing policies are automatically applied to inference requests made through `client.responses` and `client.chat`.

> **Scope:** Automatic routing applies to `responses.create()`, `responses.stream()`, `chat.create()`, `chat.stream()`, and any code that goes through `OctomilResponses` (including `WorkflowRunner` and `ToolRunner`). It does **not** apply to `client.text`, `client.audio`, or other namespaces that resolve runtimes directly.

Each deployment carries:

- `routing_mode`: `auto`, `local_only`, or `cloud_only`
- `routing_preference`: `local`, `balanced`, or `quality` (within `auto` mode)

These map to SDK routing behavior:

| Mode         | Preference | Effect                                                                          |
| ------------ | ---------- | ------------------------------------------------------------------------------- |
| `local_only` | —          | All inference runs on-device. Requests that can't be handled locally will fail. |
| `cloud_only` | —          | All inference runs in the cloud.                                                |
| `auto`       | `local`    | Prefer on-device; fall back to cloud if local is unavailable.                   |
| `auto`       | `balanced` | Route to whichever is fastest (on-device or cloud).                             |
| `auto`       | `quality`  | Prefer cloud for highest quality; use local as fallback.                        |

## Automatic routing (recommended)

After `control.get_desired_state()`, the client resolves routing automatically from the model name:

```python
import octomil
from octomil.auth import OrgApiKeyAuth

client = octomil.OctomilClient(auth=OrgApiKeyAuth(api_key="edg_...", org_id="org_123"))

# Register and fetch desired state — routing policies are applied automatically
client.control.register()
client.control.get_desired_state()

# Requests route according to the deployment's policy — no metadata needed
result = client.chat.create(model="my-model", messages=[{"role": "user", "content": "Hello"}])
```

## Per-request override

Override routing for a specific request using metadata:

```python
from octomil.responses.types import ResponseRequest, text_input

result = await client.responses.create(ResponseRequest(
    model="my-model",
    input=[text_input("Hello")],
    metadata={"routing.policy": "local_only"},
))
```

## Multi-deployment disambiguation

When the same model appears under multiple deployments with different routing policies, the model name alone is ambiguous. Pass `deployment_id` in metadata:

```python
result = await client.responses.create(ResponseRequest(
    model="shared-model",
    input=[text_input("Hello")],
    metadata={"deployment_id": "dep_private"},
))
```

When routing policies are identical across deployments of the same model, auto-resolution works normally.

## Agent sessions

`AgentSession` is standalone and does not inherit routing by default. Use `client.agent_session()` to create a session pre-wired with the client's routing:

```python
session = client.agent_session()
result = await session.run("deployment_advisor", "Deploy phi-mini to iOS staging")
```

Or pass `responses=client.responses` manually:

```python
from octomil.agents.session import AgentSession

session = AgentSession(
    base_url="https://api.octomil.com",
    auth_token="...",
    responses=client.responses,
)
```

## Resolution order

1. Explicit `routing.policy` in request metadata
2. Explicit `deployment_id` in request metadata -> per-deployment policy
3. Model name -> deployment lookup -> per-deployment policy (automatic)
4. Global default (first policy from desired state)
