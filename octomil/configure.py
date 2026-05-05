"""Module-level configure() for silent device registration.

Provides a simple entrypoint for client-side SDKs to register a device
with the Octomil platform without blocking the calling thread.

Usage::

    import octomil
    from octomil.auth_config import PublishableKeyAuth
    from octomil.monitoring_config import MonitoringConfig

    octomil.configure(
        auth=PublishableKeyAuth(key="oct_pub_live_abc123"),
        monitoring=MonitoringConfig(enabled=True),
    )
"""

from __future__ import annotations

import logging
import platform as _platform
import random
import threading
import time
from typing import Optional

import httpx

from .auth_config import AnonymousAuth, BootstrapTokenAuth, DeviceAuthConfig, PublishableKeyAuth
from .device_context import DeviceContext, RegistrationState, TokenState
from .device_info import DeviceInfo, get_battery_level, is_charging
from .monitoring_config import MonitoringConfig

logger = logging.getLogger(__name__)

__all__ = ["configure", "get_device_context"]

_DEFAULT_BASE_URL = "https://api.octomil.com/api/v1"

# Module-level singleton — populated by configure()
_device_context: Optional[DeviceContext] = None
_registration_thread: Optional[threading.Thread] = None
_heartbeat_thread: Optional[threading.Thread] = None
_heartbeat_stop: threading.Event = threading.Event()


def get_device_context() -> Optional[DeviceContext]:
    """Return the current DeviceContext, or None if configure() has not been called."""
    return _device_context


def configure(
    auth: Optional[DeviceAuthConfig] = None,
    monitoring: Optional[MonitoringConfig] = None,
    base_url: Optional[str] = None,
) -> DeviceContext:
    """Configure the SDK for silent device registration.

    Populates a :class:`DeviceContext` immediately and, when ``auth``
    is provided, starts a background thread that registers the device
    with the Octomil API.  Registration failure never blocks the caller.

    Args:
        auth: Device authentication configuration. One of
            :class:`PublishableKeyAuth`, :class:`BootstrapTokenAuth`,
            or :class:`AnonymousAuth`.
        monitoring: Optional monitoring configuration controlling
            heartbeat behaviour.
        base_url: Override the default Octomil API base URL.

    Returns:
        The :class:`DeviceContext` that tracks registration state.
    """
    global _device_context, _registration_thread, _heartbeat_thread, _heartbeat_stop  # noqa: PLW0603

    effective_base = base_url or _DEFAULT_BASE_URL
    mon = monitoring or MonitoringConfig()

    ctx = DeviceContext()
    # Populate org_id / app_id from auth config if available
    if isinstance(auth, AnonymousAuth):
        ctx.app_id = auth.app_id

    _device_context = ctx

    if auth is not None and _should_auto_register(auth):
        _registration_thread = threading.Thread(
            target=_background_register,
            args=(ctx, auth, effective_base),
            daemon=True,
            name="octomil-device-register",
        )
        _registration_thread.start()

    # Start heartbeat if monitoring is enabled
    if mon.enabled:
        _heartbeat_stop.clear()
        _heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(ctx, effective_base, mon.heartbeat_interval_seconds),
            daemon=True,
            name="octomil-heartbeat",
        )
        _heartbeat_thread.start()

    return ctx


def _should_auto_register(auth: DeviceAuthConfig) -> bool:
    """Determine whether the auth config should trigger auto-registration."""
    if isinstance(auth, PublishableKeyAuth):
        return True
    if isinstance(auth, BootstrapTokenAuth):
        return True
    if isinstance(auth, AnonymousAuth):
        return True
    return False


def _background_register(
    ctx: DeviceContext,
    auth: DeviceAuthConfig,
    base_url: str,
    *,
    max_retries: int = 5,
) -> None:
    """Register the device with exponential backoff + jitter. Never raises."""
    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            _do_register(ctx, auth, base_url)
            ctx.registration_state = RegistrationState.REGISTERED
            logger.info(
                "Device registered: installation_id=%s server_device_id=%s",
                ctx.installation_id,
                ctx.server_device_id,
            )
            return
        except Exception:
            logger.debug("Device registration attempt %d failed", attempt + 1, exc_info=True)
            delay = base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
            time.sleep(delay)

    ctx.registration_state = RegistrationState.FAILED
    logger.warning("Device registration failed after %d attempts", max_retries)


def _do_register(
    ctx: DeviceContext,
    auth: DeviceAuthConfig,
    base_url: str,
) -> None:
    """Execute a single registration attempt against the API."""
    url = f"{base_url.rstrip('/')}/devices/register"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if isinstance(auth, PublishableKeyAuth):
        headers["Authorization"] = f"Bearer {auth.key}"
    elif isinstance(auth, BootstrapTokenAuth):
        headers["Authorization"] = f"Bearer {auth.token}"

    hw = DeviceInfo().collect_device_info()
    payload: dict[str, object] = {
        "device_identifier": ctx.installation_id,
        "installation_id": ctx.installation_id,
        "platform": "python",
        "sdk_version": _get_sdk_version(),
        "os_version": f"{_platform.system()} {_platform.release()}",
        "manufacturer": hw.get("manufacturer"),
        "model": hw.get("model"),
        "cpu_architecture": hw.get("cpu_architecture"),
        "gpu_available": hw.get("gpu_available"),
        "total_memory_mb": hw.get("total_memory_mb"),
        "available_storage_mb": hw.get("available_storage_mb"),
        "battery_pct": get_battery_level(),
        "charging": is_charging(),
    }
    if ctx.org_id:
        payload["org_id"] = ctx.org_id
    if ctx.app_id:
        payload["app_id"] = ctx.app_id

    with httpx.Client(timeout=10.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    ctx.server_device_id = data.get("id") or data.get("device_id")
    ctx.org_id = data.get("org_id") or ctx.org_id

    # If the server returns a token, populate token state
    access_token = data.get("access_token")
    if access_token:
        expires_at = data.get("expires_at")
        ctx.token_state = TokenState(
            access_token=access_token,
            expires_at=float(expires_at) if expires_at is not None else None,
        )


def _heartbeat_loop(
    ctx: DeviceContext,
    base_url: str,
    interval_seconds: int,
) -> None:
    """Background heartbeat loop. Sends heartbeat pings at the configured interval."""
    while not _heartbeat_stop.wait(timeout=interval_seconds):
        if ctx.registration_state != RegistrationState.REGISTERED:
            continue
        if not ctx.server_device_id:
            continue
        try:
            url = f"{base_url.rstrip('/')}/devices/{ctx.server_device_id}/heartbeat"
            headers: dict[str, str] = {"Content-Type": "application/json"}
            auth_headers = ctx.auth_headers()
            if auth_headers:
                headers.update(auth_headers)
            payload = {
                "sdk_version": _get_sdk_version(),
                "platform": "python",
            }
            with httpx.Client(timeout=5.0) as client:
                client.post(url, json=payload, headers=headers)
        except Exception:
            logger.debug("Heartbeat failed", exc_info=True)


def _get_sdk_version() -> str:
    try:
        from octomil import __version__

        return __version__
    except ImportError:
        return "0.0.0"
