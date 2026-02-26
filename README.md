<p align="center">
  <strong>Octomil</strong><br>
  On-device AI inference. Deploy, route, observe.
</p>

<p align="center">
  <a href="https://github.com/octomil/octomil-python/actions/workflows/ci.yml"><img src="https://github.com/octomil/octomil-python/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/octomil-sdk/"><img src="https://img.shields.io/pypi/v/octomil-sdk.svg" alt="PyPI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT"></a>
</p>

---

## Install

```bash
pip install octomil-sdk
```

## Quick Start

Serve a model locally with an OpenAI-compatible API:

```bash
octomil serve gemma-1b
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "gemma-1b", "messages": [{"role": "user", "content": "Hello"}]}'
```

## SDK Usage

```python
from octomil import Octomil

client = Octomil(api_key="oct_...", org_id="org_123")

# Register a model
model = client.registry.ensure_model(name="sentiment", framework="pytorch")

# Gradual rollout
client.rollouts.create(model_id=model["id"], version="1.0.0", rollout_percentage=10)

# A/B test
client.experiments.create(name="v1-vs-v2", model_id=model["id"],
                          control_version="1.0.0", treatment_version="1.1.0")
```

## CLI

| Command                     |                                            |
| --------------------------- | ------------------------------------------ |
| `octomil serve <model>`     | Local inference server (OpenAI-compatible) |
| `octomil pull <model>`      | Download a model                           |
| `octomil push <file>`       | Upload a model                             |
| `octomil deploy <model>`    | Deploy to devices                          |
| `octomil convert <file>`    | Convert to CoreML / TFLite                 |
| `octomil check <file>`      | Validate a model                           |
| `octomil scan <path>`       | Security scan                              |
| `octomil benchmark <model>` | Latency benchmarks                         |
| `octomil login`             | Authenticate                               |

## Federated Learning (Enterprise)

Train across devices without centralizing data:

```python
from octomil import DeviceAuthClient, FederatedClient

auth = DeviceAuthClient(base_url="https://api.octomil.com", org_id="org_123",
                        device_identifier="runtime-001")
await auth.bootstrap(bootstrap_bearer_token=token)

client = FederatedClient(auth_token_provider=lambda: auth.get_access_token_sync(),
                         org_id="org_123")
client.register()

assignment = client.get_round_assignment()
if assignment:
    client.participate_in_round(round_id=assignment["round_id"],
                                local_train_fn=my_train_fn)
```

## Documentation

[docs.octomil.com/sdks/python](https://docs.octomil.com/sdks/python)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
