"""Tests for octomil.auth_config — publishable key auth configuration."""

import unittest

from octomil.auth_config import AnonymousAuth, BootstrapTokenAuth, PublishableKeyAuth
from octomil.errors import OctomilError, OctomilErrorCode


class TestPublishableKeyAuth(unittest.TestCase):
    def test_valid_live_key(self):
        auth = PublishableKeyAuth(key="oct_pub_live_abc123xyz")
        self.assertEqual(auth.key, "oct_pub_live_abc123xyz")

    def test_valid_test_key(self):
        auth = PublishableKeyAuth(key="oct_pub_test_mykey456")
        self.assertEqual(auth.key, "oct_pub_test_mykey456")

    def test_invalid_prefix_raises(self):
        with self.assertRaises(OctomilError) as ctx:
            PublishableKeyAuth(key="oct_pub_badprefix")
        self.assertEqual(ctx.exception.code, OctomilErrorCode.INVALID_API_KEY)
        self.assertIn("oct_pub_test_", str(ctx.exception))
        self.assertIn("oct_pub_live_", str(ctx.exception))

    def test_bare_oct_pub_prefix_rejected(self):
        with self.assertRaises(OctomilError):
            PublishableKeyAuth(key="oct_pub_something")

    def test_empty_key_rejected(self):
        with self.assertRaises(OctomilError):
            PublishableKeyAuth(key="")

    def test_random_string_rejected(self):
        with self.assertRaises(OctomilError):
            PublishableKeyAuth(key="sk_live_something")

    def test_frozen(self):
        auth = PublishableKeyAuth(key="oct_pub_live_frozen")
        with self.assertRaises(AttributeError):
            auth.key = "changed"  # type: ignore[misc]


class TestBootstrapTokenAuth(unittest.TestCase):
    def test_creation(self):
        auth = BootstrapTokenAuth(token="jwt.token.here")
        self.assertEqual(auth.token, "jwt.token.here")

    def test_frozen(self):
        auth = BootstrapTokenAuth(token="jwt")
        with self.assertRaises(AttributeError):
            auth.token = "changed"  # type: ignore[misc]


class TestAnonymousAuth(unittest.TestCase):
    def test_creation(self):
        auth = AnonymousAuth(app_id="my-app")
        self.assertEqual(auth.app_id, "my-app")

    def test_frozen(self):
        auth = AnonymousAuth(app_id="app")
        with self.assertRaises(AttributeError):
            auth.app_id = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
