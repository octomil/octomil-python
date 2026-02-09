import unittest
from unittest.mock import patch

from edgeml.edge import EdgeML
from edgeml.federated_client import FederatedClient


class EdgeMLTests(unittest.TestCase):
    def test_init(self):
        # Mock all the components that make HTTP calls during init
        with patch("edgeml.edge._ApiClient"), \
             patch("edgeml.edge.ModelRegistry"), \
             patch("edgeml.edge.Federation"):
            client = EdgeML(auth_token_provider=lambda: "token123", org_id="org_1", api_base="https://api.test.com")
            self.assertEqual(client.org_id, "org_1")

    def test_client_factory_with_device_identifier(self):
        with patch("edgeml.edge._ApiClient"), \
             patch("edgeml.edge.ModelRegistry"), \
             patch("edgeml.edge.Federation"):
            edge = EdgeML(auth_token_provider=lambda: "token123", org_id="org_1")
            fed_client = edge.client(device_identifier="device_123", platform="ios")
            self.assertIsInstance(fed_client, FederatedClient)
            self.assertEqual(fed_client.device_identifier, "device_123")
            self.assertEqual(fed_client.platform, "ios")

    def test_client_factory_without_device_identifier(self):
        with patch("edgeml.edge._ApiClient"), \
             patch("edgeml.edge.ModelRegistry"), \
             patch("edgeml.edge.Federation"):
            edge = EdgeML(auth_token_provider=lambda: "token123", org_id="org_1")
            fed_client = edge.client()
            self.assertIsInstance(fed_client, FederatedClient)
            self.assertIsNotNone(fed_client.device_identifier)  # Generated automatically
            self.assertEqual(fed_client.platform, "python")


if __name__ == "__main__":
    unittest.main()
