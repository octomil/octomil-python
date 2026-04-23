import os
import unittest
from unittest.mock import patch

from octomil.edge import Octomil
from octomil.federated_client import FederatedClient


class OctomilTests(unittest.TestCase):
    def test_init(self):
        # Mock all the components that make HTTP calls during init
        with patch("octomil.edge._ApiClient"), patch("octomil.edge.ModelRegistry"), patch("octomil.edge.Federation"):
            client = Octomil(
                auth_token_provider=lambda: "token123",
                org_id="org_1",
                api_base="https://api.test.com",
            )
            self.assertEqual(client.org_id, "org_1")

    def test_client_factory_with_device_identifier(self):
        with patch("octomil.edge._ApiClient"), patch("octomil.edge.ModelRegistry"), patch("octomil.edge.Federation"):
            edge = Octomil(auth_token_provider=lambda: "token123", org_id="org_1")
            fed_client = edge.client(device_identifier="device_123", platform="ios")
            self.assertIsInstance(fed_client, FederatedClient)
            self.assertEqual(fed_client.device_identifier, "device_123")
            self.assertEqual(fed_client.platform, "ios")

    def test_client_factory_without_device_identifier(self):
        with patch("octomil.edge._ApiClient"), patch("octomil.edge.ModelRegistry"), patch("octomil.edge.Federation"):
            edge = Octomil(auth_token_provider=lambda: "token123", org_id="org_1")
            fed_client = edge.client()
            self.assertIsInstance(fed_client, FederatedClient)
            self.assertIsNotNone(fed_client.device_identifier)  # Generated automatically
            self.assertEqual(fed_client.platform, "python")

    def test_from_env_delegates_to_unified_facade(self):
        from octomil.facade import Octomil as FacadeOctomil

        with patch.dict(
            os.environ,
            {
                "OCTOMIL_SERVER_KEY": "oct_sk_test_123",
                "OCTOMIL_ORG_ID": "org_1",
            },
            clear=True,
        ):
            client = Octomil.from_env()

        self.assertIsInstance(client, FacadeOctomil)

    def test_hosted_from_env_delegates_to_unified_facade(self):
        from octomil.facade import Octomil as FacadeOctomil

        with patch.dict(
            os.environ,
            {
                "OCTOMIL_SERVER_KEY": "oct_sk_test_123",
                "OCTOMIL_ORG_ID": "org_1",
            },
            clear=True,
        ):
            client = Octomil.hosted_from_env()

        self.assertIsInstance(client, FacadeOctomil)
        self.assertFalse(client.planner_enabled)


if __name__ == "__main__":
    unittest.main()
