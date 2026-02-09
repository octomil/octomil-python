# EdgeML Python SDK

[![CI](https://github.com/edgeml-ai/edgeml-python/actions/workflows/ci.yml/badge.svg)](https://github.com/edgeml-ai/edgeml-python/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/edgeml-ai/edgeml-python/branch/main/graph/badge.svg)](https://codecov.io/gh/edgeml-ai/edgeml-python)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=edgeml-ai_edgeml-python&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=edgeml-ai_edgeml-python)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=edgeml-ai_edgeml-python&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=edgeml-ai_edgeml-python)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/edgeml-ai/edgeml-python/badge)](https://scorecard.dev/viewer/?uri=github.com/edgeml-ai/edgeml-python)
[![CII Best Practices](https://www.bestpractices.dev/projects/11914/badge)](https://www.bestpractices.dev/projects/11914)
[![CodeQL](https://github.com/edgeml-ai/edgeml-python/actions/workflows/codeql.yml/badge.svg)](https://github.com/edgeml-ai/edgeml-python/actions/workflows/codeql.yml)
[![Python Version](https://img.shields.io/pypi/pyversions/edgeml-sdk.svg)](https://pypi.org/project/edgeml-sdk/)
[![PyPI version](https://badge.fury.io/py/edgeml-sdk.svg)](https://badge.fury.io/py/edgeml-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Enterprise-grade Python SDK for federated learning orchestration and secure device runtime participation.

## Overview

The EdgeML Python SDK enables privacy-preserving federated learning for production environments. Built with enterprise security requirements in mind, it provides secure device authentication, automated token management, and comprehensive control-plane APIs for model orchestration.

### Key Features

- **🔒 Enterprise Security**: Token-based authentication with automatic refresh and secure keyring storage
- **🚀 Production Ready**: Comprehensive error handling, logging, and monitoring capabilities
- **📊 Model Management**: Full control-plane APIs for model registry, rollouts, and experiments
- **🔄 Federated Learning**: Round-based training with automatic model synchronization and delta computation
- **🎯 Personalization**: Ditto and FedPer strategies for device-specific model adaptation
- **🛡️ Privacy Filters**: Configurable filter pipeline (gradient clipping, noise, sparsification, quantization)
- **✅ Type Safe**: Complete type hints for enhanced IDE support and code quality
- **📈 Observable**: Built-in metrics, logging, and health check endpoints

### Security & Privacy

- ✅ **Code Coverage**: >80% test coverage
- ✅ **Static Analysis**: SonarCloud quality gates enforced
- ✅ **Security Scanning**: Bandit and Safety checks on every commit
- ✅ **No Data Exfiltration**: All training happens on-device
- ✅ **Zero-Knowledge Server**: Server never sees raw training data

## Installation

```bash
pip install edgeml-sdk
```

### Optional Dependencies

```bash
# For cloud storage support
pip install edgeml-sdk[s3]      # AWS S3
pip install edgeml-sdk[gcs]     # Google Cloud Storage
pip install edgeml-sdk[azure]   # Azure Blob Storage
pip install edgeml-sdk[all]     # All cloud providers

# For deep learning frameworks
pip install edgeml-sdk[torch]   # PyTorch support

# For secure token storage
pip install edgeml-sdk[auth]    # System keyring integration

# For development
pip install edgeml-sdk[dev]     # Testing and linting tools
```

## Quick Start

### Enterprise Runtime Authentication

For production deployments, use secure token-based authentication:

```python
from edgeml import DeviceAuthClient, FederatedClient

# Initialize device auth client
auth = DeviceAuthClient(
    base_url="https://api.edgeml.io",
    org_id="org_123",
    device_identifier="python-runtime-001",
)

# One-time bootstrap with backend-issued token
await auth.bootstrap(bootstrap_bearer_token=backend_bootstrap_token)

# Create federated client with automatic token refresh
client = FederatedClient(
    auth_token_provider=lambda: auth.get_access_token_sync(),
    org_id="org_123",
)

# Register device and participate in federated rounds
device_id = client.register()

# Check for an active training round
assignment = client.get_round_assignment()
if assignment:
    result = client.participate_in_round(
        round_id=assignment["round_id"],
        local_train_fn=my_train_fn,  # your training loop
    )
    # participate_in_round handles the full lifecycle:
    # fetch config -> pull model -> train -> compute delta -> apply filters -> upload
```

### Control Plane APIs

For model management and orchestration:

```python
from edgeml import EdgeML

edge = EdgeML(
    auth_token_provider=lambda: auth.get_access_token_sync(),
    org_id="org_123",
)

# Model registry operations
model = edge.registry.ensure_model(
    name="sentiment-model",
    framework="pytorch",
    use_case="nlp",
)

# Deploy with gradual rollout
edge.rollouts.create(
    model_id=model["id"],
    version="1.0.0",
    rollout_percentage=10,
    target_percentage=100,
    increment_step=10,
)

# A/B testing
experiment = edge.experiments.create(
    name="v1-vs-v2",
    model_id=model["id"],
    control_version="1.0.0",
    treatment_version="1.1.0",
)
```

## Architecture

The Python SDK provides enterprise-grade device authentication and federated learning capabilities:

### Token Lifecycle

- **Access Token**: 15 minutes (configurable, max 60 minutes)
- **Refresh Token**: 30 days (automatically rotated on refresh)
- **Storage**: System keyring (production) or in-memory (development)

The SDK automatically handles token refresh before expiration, ensuring uninterrupted API access.

## Round-Based Training

The SDK manages the full lifecycle of federated training rounds:

```python
from edgeml import FederatedClient

client = FederatedClient(
    auth_token_provider=auth_provider,
    org_id="org_123",
)
client.register()

# Poll for round assignments
assignment = client.get_round_assignment()
if assignment:
    # Full lifecycle: fetch config, pull model, train, compute delta, apply filters, upload
    result = client.participate_in_round(
        round_id=assignment["round_id"],
        local_train_fn=my_train_fn,
    )
```

### State Dict Utilities

Compute weight deltas between model states:

```python
from edgeml.federated import compute_state_dict_delta

delta = compute_state_dict_delta(base_state=global_weights, updated_state=trained_weights)
```

## Personalization

The SDK supports personalization strategies that adapt models to individual devices while still contributing to the global federation.

### Personalized Model State

```python
# Fetch the device's personalized model
personal_state = client.get_personalized_model()

# Upload a personalized update with metrics
client.upload_personalized_update(
    weights=updated_personal_weights,
    metrics={"loss": 0.12, "accuracy": 0.95},
)
```

### Ditto Strategy

Ditto trains a global model and a personal model simultaneously. The personal model uses proximal regularization to stay close to the global model:

```python
result = client.train_ditto(
    global_model=global_weights,
    personal_model=personal_weights,
    local_train_fn=my_train_fn,
    lambda_ditto=0.1,  # proximal regularization strength
)
# result contains updated global and personal model states
```

### FedPer Strategy

FedPer splits the model into body (shared) and head (personal) layers. Only body layers are uploaded to the server:

```python
result = client.train_fedper(
    model=model_weights,
    head_layers=["classifier.weight", "classifier.bias"],
    local_train_fn=my_train_fn,
)
# Only body layers (everything except head_layers) are sent to the server
```

## Filter Pipeline

Privacy-preserving filters are applied automatically during round participation. You can also use them directly:

```python
from edgeml.federated import apply_filters

filters = [
    {"type": "gradient_clip", "max_norm": 1.0},
    {"type": "gaussian_noise", "sigma": 0.01},
    {"type": "norm_validation", "max_norm": 10.0},
    {"type": "sparsification", "top_k": 0.1},
    {"type": "quantization", "bits": 8},
]

filtered_delta = apply_filters(delta=model_delta, filters=filters)
```

Supported filters:
- **gradient_clip**: Clip gradient norms to a maximum value
- **gaussian_noise**: Add calibrated Gaussian noise for differential privacy
- **norm_validation**: Reject updates that exceed a norm threshold
- **sparsification**: Keep only the top-k% of weight updates
- **quantization**: Reduce weight precision to save bandwidth

### Privacy Budget

Query the remaining privacy budget for a federation:

```python
budget = client.get_privacy_budget(federation_id="fed_456")
# Returns current epsilon spent, remaining budget, etc.
```

## Configuration

### Environment Variables

```bash
# API Configuration
export EDGEML_API_BASE="https://api.edgeml.io/api/v1"
export EDGEML_ORG_ID="your-org-id"

# Device Configuration
export EDGEML_DEVICE_ID="unique-device-identifier"
export EDGEML_PLATFORM="python"

# Token Storage (optional)
export EDGEML_TOKEN_STORAGE="keyring"  # or "memory"

# Logging
export EDGEML_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

### Advanced Configuration

```python
from edgeml import FederatedClient

client = FederatedClient(
    auth_token_provider=auth_provider,
    org_id="org_123",
    api_base="https://api.edgeml.io/api/v1",
    device_identifier="custom-id",
    platform="python",
    timeout=30.0,  # API timeout in seconds
)
```

## Best Practices

### Security

1. **Never embed API keys**: Always use token-based authentication for distributed clients
2. **Use keyring storage**: Enable secure token storage in production
3. **Rotate credentials**: Implement regular token rotation policies
4. **Monitor auth failures**: Set up alerts for authentication anomalies

### Performance

1. **Batch operations**: Use bulk APIs when processing multiple models
2. **Cache tokens**: Leverage automatic token caching and refresh
3. **Async operations**: Use async clients for concurrent requests
4. **Connection pooling**: Reuse client instances across requests

### Reliability

1. **Handle errors gracefully**: Implement retry logic with exponential backoff
2. **Monitor metrics**: Track success rates and latencies
3. **Health checks**: Implement periodic connectivity tests
4. **Fallback strategies**: Design degradation paths for API failures

## Testing

```bash
# Run all tests with coverage
pytest --cov=edgeml --cov-report=term-missing

# Run specific test suite
pytest tests/test_device_auth.py -v

# Run with different Python versions
tox
```

## Documentation

For full SDK documentation, see [https://docs.edgeml.io/sdks/python](https://docs.edgeml.io/sdks/python)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/edgeml-ai/edgeml-python.git
cd edgeml-python

# Install in development mode
pip install -e edgeml/python[dev]

# Run tests
pytest

# Run linters
ruff check .
mypy edgeml/python/edgeml
```

## Privacy Statement

### Data Collection Disclosure

The Python SDK collects minimal device information for runtime identification:

**Device Identification** (collected at registration):
- **Device Identifier**: Custom identifier you provide, or auto-generated UUID
- **Platform**: Always set to "python"
- **OS Version**: Operating system information
- **Organization ID**: Your EdgeML organization ID

**Model Information** (during training):
- Model ID and version being trained
- Training metrics (loss, accuracy, sample count)
- Compressed model weight deltas (gradients)

### No System Permissions Required

The Python SDK:
- ✅ Runs in standard Python environments (no special permissions)
- ✅ Does not access system hardware information (battery, sensors, etc.)
- ✅ Does not require root/admin privileges
- ✅ Only makes HTTPS API calls to EdgeML servers

### Why This Data is Collected

- **Device Tracking**: Distinguish training updates from different runtimes
- **Model Versioning**: Ensure devices train on correct model versions
- **Analytics**: Aggregate training performance across your fleet

### What Data is NOT Collected

**Important**: All training happens locally. The SDK never collects or transmits:
- ❌ Personal information or user data
- ❌ Training datasets or raw input data
- ❌ File system contents
- ❌ Environment variables or secrets
- ❌ System hardware specs (CPU, RAM, disk)
- ❌ Network activity outside EdgeML API calls

Only model gradients (mathematical weight updates) are uploaded to the server.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and feature requests, please use the [GitHub issue tracker](https://github.com/edgeml-ai/edgeml-python/issues).

For questions: support@edgeml.io

---

<p align="center">
  <strong>Built with ❤️ by the EdgeML Team</strong>
</p>
