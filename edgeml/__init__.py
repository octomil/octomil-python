"""
EdgeML Python SDK

Official Python SDK for the EdgeML federated learning platform.

Provides:
- Automatic device registration with hardware metadata
- Real-time battery and network monitoring
- Federated training client
- Model versioning and rollout support

Installation:
    pip install edgeml

Basic Usage:
    from edgeml import EdgeMLClient

    client = EdgeMLClient(
        api_key="edg_your_key_here",
        base_url="https://api.edgeml.io"
    )

    # Device automatically registers with full metadata
    await client.register()

    # Send periodic heartbeats with updated battery/network status
    await client.send_heartbeat()
"""

from .device_info import DeviceInfo

__version__ = "1.0.0"
__all__ = ["DeviceInfo"]
