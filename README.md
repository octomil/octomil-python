# EdgeML Python SDK

Official Python SDK for the EdgeML federated learning platform.

## Features

- ✅ **Automatic Device Registration** - Collects and sends complete hardware metadata
- ✅ **Real-Time Monitoring** - Tracks battery level, network type, and system constraints
- ✅ **Stable Device IDs** - Hardware-based identifiers prevent duplicate registrations
- ✅ **Cross-Platform** - Works on macOS, Linux, and Windows
- ✅ **Privacy-First** - All training happens on-device

## Installation

```bash
pip install edgeml psutil
```

## Quick Start

```python
from edgeml import DeviceInfo

# Collect device information
device = DeviceInfo()

# Get registration payload
registration_data = device.to_registration_dict()
print(f"Device ID: {device.device_id}")

# Get current metadata (battery, network)
metadata = device.update_metadata()
print(f"Battery: {metadata['battery_level']}%")
print(f"Network: {metadata['network_type']}")
```

## Device Information Collected

### Hardware
- Manufacturer (e.g., "Apple", "Dell")
- Model (e.g., "MacBookPro18,1")
- CPU Architecture (e.g., "arm64", "x86_64")
- Total Memory (MB)
- Available Storage (MB)
- GPU Available (boolean)

### Runtime Constraints
- Battery Level (0-100%)
- Network Type (wifi, cellular, ethernet)

### System Info
- Platform (darwin, linux, windows)
- OS Version
- Python Version
- Timezone

## Integration

```python
import httpx
from edgeml import DeviceInfo

async def register_device(api_key: str, org_id: str, base_url: str):
    device = DeviceInfo()

    # Prepare registration data
    data = device.to_registration_dict()
    data["org_id"] = org_id
    data["sdk_version"] = "1.0.0"

    # Send to EdgeML API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/api/v1/devices/register",
            json=data,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

async def send_heartbeat(device_id: str, api_key: str, base_url: str):
    device = DeviceInfo()

    # Get updated metadata
    metadata = device.update_metadata()

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{base_url}/api/v1/devices/{device_id}/heartbeat",
            json={"metadata": metadata},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()
```

## Dependencies

- **psutil** (optional but recommended) - For battery and system info
- **httpx** - For API communication

If `psutil` is not installed, the SDK will gracefully fall back to defaults:
- Battery: None
- Memory/Storage: None
- GPU detection: Limited

## License

MIT
