"""Example: automatic deployment routing with OctomilClient.

After fetching desired state, inference requests are automatically
routed according to each deployment's routing policy.
"""

import asyncio

import octomil
from octomil.auth import OrgApiKeyAuth
from octomil.responses.types import ResponseRequest, text_input

# Initialize client
client = octomil.OctomilClient(
    auth=OrgApiKeyAuth(api_key="edg_your_key", org_id="org_your_org"),
)

# Register device and fetch desired state.
# This automatically applies routing policies from your deployments.
client.control.register()
entries = client.control.get_desired_state()
print(f"Loaded {len(entries)} model(s) from desired state")

# All inference calls now route automatically.
# If "chat-model" is deployed with routing_preference="quality",
# requests will prefer cloud inference for best quality.
result = client.chat.create(
    model="chat-model",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(result.message["content"])

# Override routing for a single request:
override_result = asyncio.run(
    client.responses.create(
        ResponseRequest(
            model="chat-model",
            input=[text_input("What is 2+2?")],
            metadata={"routing.policy": "local_only"},  # force on-device
        )
    )
)
output = override_result.output[0]
if hasattr(output, "text"):
    print(output.text)
