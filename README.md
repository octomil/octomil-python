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
