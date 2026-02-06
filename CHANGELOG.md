# Changelog

## 1.1.0 - 2026-02-06
- Added `DeviceAuthClient` for short-lived device token bootstrap/refresh/revoke.
- Added secure token persistence via system keyring (`keyring`).
- Improved telemetry defaults in `DeviceInfo`:
  - network type detection now inspects active interfaces.
  - locale/region are detected from system locale (no hardcoded `en_US`/`US`).
- Updated docs to recommend backend-issued short-lived device tokens.
