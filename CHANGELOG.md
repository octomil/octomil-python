# Changelog

## Unreleased

- Added `DeviceAuthClient` runtime auth helper for device token bootstrap, refresh, and revoke flows.
- Added optional `auth` extra with `keyring` secure storage dependency.
## 2.3.0 (2026-02-26)

### Features

- migrate TelemetryReporter to v2 OTLP envelope format
- instrument Client.push, import_from_hf, rollback with v2 funnel events
- instrument FederatedClient with v2 funnel events
- add programmatic inference API and update push snippets

### Fixes

- strip variant suffix in resolve_model_id (#126)
- centralize __version__ and add release automation (#127)
- replace PAT with GitHub App token for cross-repo dispatch (#129)
- add pyproject.toml to Knope versioned files (#130)
- sync homebrew formula and test_cli to 2.2.0
- auto-install shell completions on login
- add phi-3.5-mini and variant aliases
- update knope.toml to v0.22+ config format
