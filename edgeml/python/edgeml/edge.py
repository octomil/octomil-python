from __future__ import annotations

from typing import Callable, Optional

from .api_client import _ApiClient
from .control_plane import ExperimentsAPI, RolloutsAPI
from .federated_client import FederatedClient
from .federation import Federation
from .registry import ModelRegistry


class EdgeML:
    """Unified high-level client exposing registry, rollouts, experiments and federation."""

    def __init__(
        self,
        auth_token_provider: Callable[[], str],
        org_id: str = "default",
        api_base: str = "https://api.edgeml.io/api/v1",
    ):
        self.api = _ApiClient(auth_token_provider=auth_token_provider, api_base=api_base)
        self.org_id = org_id
        self.registry = ModelRegistry(auth_token_provider, org_id=org_id, api_base=api_base)
        self.rollouts = RolloutsAPI(self.api)
        self.experiments = ExperimentsAPI(self.api, org_id=org_id)
        self.federation = Federation(auth_token_provider, org_id=org_id, api_base=api_base)

    def client(self, device_identifier: Optional[str] = None, platform: str = "python") -> FederatedClient:
        return FederatedClient(
            auth_token_provider=self.api.auth_token_provider,
            org_id=self.org_id,
            api_base=self.api.api_base,
            device_identifier=device_identifier,
            platform=platform,
        )
