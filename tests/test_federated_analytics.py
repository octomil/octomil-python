import unittest

from edgeml.control_plane import FederatedAnalyticsAPI


class _StubApi:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None):
        self.calls.append(("get", path, params))
        return {"path": path, "params": params}

    def post(self, path, payload=None):
        self.calls.append(("post", path, payload))
        return {"path": path, "payload": payload}


FED_ID = "fed_abc"


class FederatedAnalyticsApiTests(unittest.TestCase):
    def _make(self):
        api = _StubApi()
        analytics = FederatedAnalyticsAPI(api, FED_ID)
        return api, analytics

    # -- descriptive --------------------------------------------------------

    def test_descriptive_default_params(self):
        api, analytics = self._make()
        analytics.descriptive("accuracy")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, f"/federations/{FED_ID}/analytics/descriptive")
        self.assertEqual(payload["variable"], "accuracy")
        self.assertEqual(payload["group_by"], "device_group")
        self.assertTrue(payload["include_percentiles"])
        self.assertNotIn("group_ids", payload)
        self.assertNotIn("filters", payload)

    def test_descriptive_with_all_params(self):
        api, analytics = self._make()
        filters = {"start_time": "2025-01-01T00:00:00", "device_platform": "ios"}
        analytics.descriptive(
            "loss",
            group_by="federation_member",
            group_ids=["org-1", "org-2"],
            include_percentiles=False,
            filters=filters,
        )
        _, _, payload = api.calls[-1]
        self.assertEqual(payload["variable"], "loss")
        self.assertEqual(payload["group_by"], "federation_member")
        self.assertEqual(payload["group_ids"], ["org-1", "org-2"])
        self.assertFalse(payload["include_percentiles"])
        self.assertEqual(payload["filters"], filters)

    # -- t_test -------------------------------------------------------------

    def test_t_test_default_confidence(self):
        api, analytics = self._make()
        analytics.t_test("loss", "group-a", "group-b")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, f"/federations/{FED_ID}/analytics/t-test")
        self.assertEqual(payload["variable"], "loss")
        self.assertEqual(payload["group_a"], "group-a")
        self.assertEqual(payload["group_b"], "group-b")
        self.assertEqual(payload["confidence_level"], 0.95)
        self.assertNotIn("filters", payload)

    def test_t_test_custom_confidence_and_filters(self):
        api, analytics = self._make()
        filters = {"min_sample_count": 10}
        analytics.t_test("loss", "a", "b", confidence_level=0.99, filters=filters)
        _, _, payload = api.calls[-1]
        self.assertEqual(payload["confidence_level"], 0.99)
        self.assertEqual(payload["filters"], filters)

    # -- chi_square ---------------------------------------------------------

    def test_chi_square_default_params(self):
        api, analytics = self._make()
        analytics.chi_square("feature_x", "feature_y")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, f"/federations/{FED_ID}/analytics/chi-square")
        self.assertEqual(payload["variable_1"], "feature_x")
        self.assertEqual(payload["variable_2"], "feature_y")
        self.assertEqual(payload["confidence_level"], 0.95)
        self.assertNotIn("group_ids", payload)
        self.assertNotIn("filters", payload)

    def test_chi_square_with_groups_and_filters(self):
        api, analytics = self._make()
        filters = {"device_platform": "android"}
        analytics.chi_square(
            "v1", "v2",
            group_ids=["g1", "g2"],
            confidence_level=0.90,
            filters=filters,
        )
        _, _, payload = api.calls[-1]
        self.assertEqual(payload["group_ids"], ["g1", "g2"])
        self.assertEqual(payload["confidence_level"], 0.90)
        self.assertEqual(payload["filters"], filters)

    # -- anova --------------------------------------------------------------

    def test_anova_default_params(self):
        api, analytics = self._make()
        analytics.anova("latency_ms")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, f"/federations/{FED_ID}/analytics/anova")
        self.assertEqual(payload["variable"], "latency_ms")
        self.assertEqual(payload["group_by"], "device_group")
        self.assertEqual(payload["confidence_level"], 0.95)
        self.assertTrue(payload["post_hoc"])
        self.assertNotIn("group_ids", payload)
        self.assertNotIn("filters", payload)

    def test_anova_with_all_params(self):
        api, analytics = self._make()
        filters = {"end_time": "2025-06-01T00:00:00"}
        analytics.anova(
            "latency_ms",
            group_by="federation_member",
            group_ids=["org-1", "org-2", "org-3"],
            confidence_level=0.99,
            post_hoc=False,
            filters=filters,
        )
        _, _, payload = api.calls[-1]
        self.assertEqual(payload["group_by"], "federation_member")
        self.assertEqual(payload["group_ids"], ["org-1", "org-2", "org-3"])
        self.assertEqual(payload["confidence_level"], 0.99)
        self.assertFalse(payload["post_hoc"])
        self.assertEqual(payload["filters"], filters)

    # -- list_queries -------------------------------------------------------

    def test_list_queries_default_params(self):
        api, analytics = self._make()
        analytics.list_queries()
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, f"/federations/{FED_ID}/analytics/queries")
        self.assertEqual(params["limit"], 50)
        self.assertEqual(params["offset"], 0)

    def test_list_queries_custom_params(self):
        api, analytics = self._make()
        analytics.list_queries(limit=10, offset=20)
        _, _, params = api.calls[-1]
        self.assertEqual(params["limit"], 10)
        self.assertEqual(params["offset"], 20)

    # -- get_query ----------------------------------------------------------

    def test_get_query(self):
        api, analytics = self._make()
        analytics.get_query("query_456")
        method, path, _ = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, f"/federations/{FED_ID}/analytics/queries/query_456")

    # -- response pass-through ----------------------------------------------

    def test_post_returns_api_response(self):
        api, analytics = self._make()
        result = analytics.descriptive("accuracy")
        self.assertIn("path", result)
        self.assertIn("payload", result)

    def test_get_returns_api_response(self):
        api, analytics = self._make()
        result = analytics.get_query("q1")
        self.assertIn("path", result)


if __name__ == "__main__":
    unittest.main()
