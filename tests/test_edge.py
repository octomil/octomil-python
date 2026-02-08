import unittest

from octomil.edge import Octomil
from octomil.federated_client import FederatedClient


class OctomilTests(unittest.TestCase):
    def test_init(self):
        client = Octomil(auth_token_provider=lambda: "token123", org_id="org_1", api_base="https://api.test.com")
        self.assertEqual(client.org_id, "org_1")
        self.assertIsNotNone(client.api)
        self.assertIsNotNone(client.registry)
        self.assertIsNotNone(client.rollouts)
        self.assertIsNotNone(client.experiments)
        self.assertIsNotNone(client.federation)

    def test_client_factory_with_device_identifier(self):
        edge = Octomil(auth_token_provider=lambda: "token123", org_id="org_1")
        fed_client = edge.client(device_identifier="device_123", platform="ios")
        self.assertIsInstance(fed_client, FederatedClient)
        self.assertEqual(fed_client.device_identifier, "device_123")
        self.assertEqual(fed_client.platform, "ios")

    def test_client_factory_without_device_identifier(self):
        edge = Octomil(auth_token_provider=lambda: "token123", org_id="org_1")
        fed_client = edge.client()
        self.assertIsInstance(fed_client, FederatedClient)
        self.assertIsNotNone(fed_client.device_identifier)  # Generated automatically
        self.assertEqual(fed_client.platform, "python")


if __name__ == "__main__":
    unittest.main()
