"""Tests for the extensible filter pipeline (octomil.filters).

Covers:
- DeltaFilter abstract base class
- FilterRegistry (register, get, unregister, list, validation)
- Built-in filters via class API
- Custom user-defined filters
- Data-kind routing
- Audit trail
- Pipeline runner with mixed dict configs and DeltaFilter instances
- Backward compatibility via federated_client.apply_filters wrapper
"""

import unittest
from typing import List

from octomil.filters import (
    DataKind,
    DeltaFilter,
    FilterRegistry,
    FilterResult,
    GaussianNoiseFilter,
    GradientClipFilter,
    NormValidationFilter,
    QuantizationFilter,
    SparsificationFilter,
    apply_filters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DoubleFilter(DeltaFilter):
    """Test filter that doubles all tensor values."""

    supported_data_kinds = [DataKind.WEIGHT_DIFF]

    def process(self, delta, config=None):
        import torch

        return {k: v * 2 if torch.is_tensor(v) else v for k, v in delta.items()}


class _MetricsOnlyFilter(DeltaFilter):
    """Test filter that only applies to METRICS data kind."""

    supported_data_kinds = [DataKind.METRICS]

    def process(self, delta, config=None):
        return {k: v for k, v in delta.items()}


class _AnyKindFilter(DeltaFilter):
    """Test filter that applies to ANY data kind."""

    supported_data_kinds = [DataKind.ANY]

    def process(self, delta, config=None):
        return delta


class _NoneReturnFilter(DeltaFilter):
    """Test filter that returns None (no change)."""

    def process(self, delta, config=None):
        return None


class _ConfigReadingFilter(DeltaFilter):
    """Test filter that reads a value from config and scales tensors."""

    def process(self, delta, config=None):
        import torch

        factor = float((config or {}).get("factor", 1.0))
        return {k: v * factor if torch.is_tensor(v) else v for k, v in delta.items()}


# ---------------------------------------------------------------------------
# DeltaFilter ABC
# ---------------------------------------------------------------------------


class DeltaFilterBaseTests(unittest.TestCase):
    """Tests for DeltaFilter abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """DeltaFilter cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            DeltaFilter()

    def test_name_defaults_to_class_name(self):
        f = _DoubleFilter()
        self.assertEqual(f.name, "_DoubleFilter")

    def test_supported_data_kinds_on_subclass(self):
        f = _DoubleFilter()
        self.assertEqual(f.supported_data_kinds, [DataKind.WEIGHT_DIFF])

    def test_supported_data_kinds_default_none(self):
        f = _NoneReturnFilter()
        self.assertIsNone(f.supported_data_kinds)


# ---------------------------------------------------------------------------
# FilterRegistry
# ---------------------------------------------------------------------------


class FilterRegistryTests(unittest.TestCase):
    """Tests for FilterRegistry."""

    def setUp(self):
        # Clean up any test registrations after each test
        self._to_cleanup: List[str] = []

    def tearDown(self):
        for name in self._to_cleanup:
            FilterRegistry.unregister(name)

    def _register(self, name, cls):
        FilterRegistry.register(name, cls)
        self._to_cleanup.append(name)

    def test_builtin_filters_registered(self):
        """All 5 built-in filters should be pre-registered."""
        names = FilterRegistry.list_filters()
        for expected in [
            "gradient_clip",
            "gaussian_noise",
            "norm_validation",
            "sparsification",
            "quantization",
        ]:
            self.assertIn(expected, names)

    def test_get_returns_class(self):
        cls = FilterRegistry.get("gradient_clip")
        self.assertIs(cls, GradientClipFilter)

    def test_get_unknown_returns_none(self):
        self.assertIsNone(FilterRegistry.get("nonexistent_filter"))

    def test_register_custom_filter(self):
        self._register("double_test", _DoubleFilter)
        self.assertIs(FilterRegistry.get("double_test"), _DoubleFilter)

    def test_register_non_subclass_raises(self):
        with self.assertRaises(ValueError) as ctx:
            FilterRegistry.register("bad", str)
        self.assertIn("DeltaFilter subclass", str(ctx.exception))

    def test_register_non_class_raises(self):
        with self.assertRaises(ValueError):
            FilterRegistry.register("bad", lambda: None)

    def test_unregister_existing(self):
        self._register("temp_filter", _DoubleFilter)
        result = FilterRegistry.unregister("temp_filter")
        self.assertTrue(result)
        self.assertIsNone(FilterRegistry.get("temp_filter"))
        self._to_cleanup.remove("temp_filter")

    def test_unregister_nonexistent(self):
        result = FilterRegistry.unregister("never_registered")
        self.assertFalse(result)

    def test_list_filters_sorted(self):
        names = FilterRegistry.list_filters()
        self.assertEqual(names, sorted(names))

    def test_register_overwrites(self):
        """Registering under the same name overwrites the previous class."""
        self._register("overwrite_test", _DoubleFilter)
        self.assertIs(FilterRegistry.get("overwrite_test"), _DoubleFilter)

        FilterRegistry.register("overwrite_test", _MetricsOnlyFilter)
        self.assertIs(FilterRegistry.get("overwrite_test"), _MetricsOnlyFilter)


# ---------------------------------------------------------------------------
# FilterResult
# ---------------------------------------------------------------------------


class FilterResultTests(unittest.TestCase):
    """Tests for FilterResult dataclass."""

    def test_default_audit_trail_empty(self):
        r = FilterResult(delta={"w": 1})
        self.assertEqual(r.audit_trail, [])

    def test_custom_audit_trail(self):
        trail = [("GradientClipFilter", "weight_diff")]
        r = FilterResult(delta={"w": 1}, audit_trail=trail)
        self.assertEqual(len(r.audit_trail), 1)
        self.assertEqual(r.audit_trail[0][0], "GradientClipFilter")


# ---------------------------------------------------------------------------
# DataKind
# ---------------------------------------------------------------------------


class DataKindTests(unittest.TestCase):
    """Tests for DataKind enum."""

    def test_values(self):
        self.assertEqual(DataKind.WEIGHTS.value, "weights")
        self.assertEqual(DataKind.WEIGHT_DIFF.value, "weight_diff")
        self.assertEqual(DataKind.METRICS.value, "metrics")
        self.assertEqual(DataKind.ANY.value, "any")

    def test_string_comparison(self):
        """DataKind is a str enum, so it can be compared with strings."""
        self.assertEqual(DataKind.WEIGHTS, "weights")


# ---------------------------------------------------------------------------
# Built-in filters via class API
# ---------------------------------------------------------------------------


class BuiltInFilterClassTests(unittest.TestCase):
    """Test built-in filter classes directly (not through dict config)."""

    def test_gradient_clip_as_instance(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        f = GradientClipFilter(max_norm=1.0)
        delta = {"w": torch.tensor([3.0, 4.0])}
        result = f.process(delta)

        clipped_norm = torch.norm(result["w"].float().flatten(), dim=0)
        self.assertAlmostEqual(clipped_norm.item(), 1.0, places=4)

    def test_gradient_clip_config_overrides_init(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        f = GradientClipFilter(max_norm=100.0)
        delta = {"w": torch.tensor([3.0, 4.0])}
        result = f.process(delta, config={"max_norm": 1.0})

        clipped_norm = torch.norm(result["w"].float().flatten(), dim=0)
        self.assertAlmostEqual(clipped_norm.item(), 1.0, places=4)

    def test_gaussian_noise_as_instance(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        torch.manual_seed(42)
        f = GaussianNoiseFilter(stddev=1.0)
        delta = {"w": torch.zeros(100)}
        result = f.process(delta)

        self.assertGreater(torch.abs(result["w"]).sum().item(), 0)

    def test_norm_validation_as_instance(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        f = NormValidationFilter(max_norm=1.0)
        delta = {
            "small": torch.tensor([0.1, 0.2]),
            "large": torch.tensor([100.0, 200.0]),
        }
        result = f.process(delta)

        self.assertIn("small", result)
        self.assertNotIn("large", result)

    def test_sparsification_as_instance(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        f = SparsificationFilter(top_k_percent=40.0)
        delta = {"w": torch.tensor([0.01, 0.5, 0.02, 0.9, 0.01])}
        result = f.process(delta)

        non_zero = (result["w"] != 0).sum().item()
        self.assertEqual(non_zero, 2)

    def test_quantization_as_instance(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        f = QuantizationFilter(bits=2)
        delta = {"w": torch.tensor([0.0, 0.33, 0.66, 1.0])}
        result = f.process(delta)

        unique_vals = result["w"].unique()
        self.assertLessEqual(len(unique_vals), 4)

    def test_builtin_supported_data_kinds(self):
        """All built-in filters support WEIGHT_DIFF and WEIGHTS."""
        for cls in [
            GradientClipFilter,
            GaussianNoiseFilter,
            NormValidationFilter,
            SparsificationFilter,
            QuantizationFilter,
        ]:
            f = cls()
            self.assertIn(DataKind.WEIGHT_DIFF, f.supported_data_kinds)
            self.assertIn(DataKind.WEIGHTS, f.supported_data_kinds)


# ---------------------------------------------------------------------------
# Data-kind routing
# ---------------------------------------------------------------------------


class DataKindRoutingTests(unittest.TestCase):
    """Test that filters are skipped when data kind doesn't match."""

    def test_filter_skipped_for_wrong_data_kind(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([1.0, 2.0])}
        f = _DoubleFilter()  # only supports WEIGHT_DIFF

        # Apply with METRICS data kind -- should skip
        result = apply_filters(delta, [f], data_kind=DataKind.METRICS)

        torch.testing.assert_close(result.delta["w"], delta["w"])
        self.assertEqual(len(result.audit_trail), 0)

    def test_filter_applied_for_matching_data_kind(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([1.0, 2.0])}
        f = _DoubleFilter()  # supports WEIGHT_DIFF

        result = apply_filters(delta, [f], data_kind=DataKind.WEIGHT_DIFF)

        torch.testing.assert_close(result.delta["w"], torch.tensor([2.0, 4.0]))
        self.assertEqual(len(result.audit_trail), 1)

    def test_any_kind_filter_always_applied(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([1.0])}
        f = _AnyKindFilter()

        for kind in [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF, DataKind.METRICS]:
            result = apply_filters(delta, [f], data_kind=kind)
            self.assertEqual(len(result.audit_trail), 1)

    def test_none_supported_kinds_means_all(self):
        """Filter with supported_data_kinds=None applies to all kinds."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([1.0])}
        f = _NoneReturnFilter()  # supported_data_kinds is None

        # None return means no change, but the filter IS invoked (no skip)
        # Since it returns None, it won't appear in audit trail
        result = apply_filters(delta, [f], data_kind=DataKind.METRICS)
        # The filter was invoked but returned None so no audit entry
        self.assertEqual(len(result.audit_trail), 0)


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


class AuditTrailTests(unittest.TestCase):
    """Test audit trail recording."""

    def test_audit_trail_records_applied_filters(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([3.0, 4.0])}
        result = apply_filters(
            delta,
            [
                {"type": "gradient_clip", "max_norm": 1.0},
                {"type": "gaussian_noise", "stddev": 0.01},
            ],
        )

        self.assertEqual(len(result.audit_trail), 2)
        self.assertEqual(result.audit_trail[0][0], "GradientClipFilter")
        self.assertEqual(result.audit_trail[0][1], "weight_diff")
        self.assertEqual(result.audit_trail[1][0], "GaussianNoiseFilter")

    def test_audit_trail_excludes_skipped_filters(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([1.0])}
        result = apply_filters(
            delta,
            [
                {"type": "unknown_filter"},
                {"type": "gradient_clip", "max_norm": 10.0},
            ],
        )

        self.assertEqual(len(result.audit_trail), 1)
        self.assertEqual(result.audit_trail[0][0], "GradientClipFilter")

    def test_audit_trail_excludes_none_return(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([1.0])}
        result = apply_filters(delta, [_NoneReturnFilter()])

        self.assertEqual(len(result.audit_trail), 0)

    def test_audit_trail_with_custom_data_kind(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([3.0, 4.0])}
        result = apply_filters(
            delta,
            [{"type": "gradient_clip", "max_norm": 1.0}],
            data_kind=DataKind.WEIGHTS,
        )

        self.assertEqual(len(result.audit_trail), 1)
        self.assertEqual(result.audit_trail[0][1], "weights")


# ---------------------------------------------------------------------------
# Pipeline runner: mixed dict + DeltaFilter instances
# ---------------------------------------------------------------------------


class MixedPipelineTests(unittest.TestCase):
    """Test pipeline with mix of dict configs and DeltaFilter instances."""

    def test_dict_and_instance_mixed(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([3.0, 4.0])}
        result = apply_filters(
            delta,
            [
                {"type": "gradient_clip", "max_norm": 1.0},  # dict config
                _DoubleFilter(),  # instance
            ],
        )

        # After clip to norm 1.0, then doubled -> norm should be ~2.0
        clipped_norm = torch.norm(result.delta["w"].float().flatten(), dim=0)
        self.assertAlmostEqual(clipped_norm.item(), 2.0, places=3)
        self.assertEqual(len(result.audit_trail), 2)

    def test_instance_with_config_passthrough(self):
        """DeltaFilter instances receive config=None."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([1.0])}
        f = _DoubleFilter()
        result = apply_filters(delta, [f])

        torch.testing.assert_close(result.delta["w"], torch.tensor([2.0]))

    def test_invalid_entry_skipped(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([1.0, 2.0])}
        result = apply_filters(
            delta,
            [
                42,  # invalid
                "not_a_filter",  # invalid
                {"type": "gradient_clip", "max_norm": 10.0},  # valid
            ],
        )

        self.assertEqual(len(result.audit_trail), 1)

    def test_empty_pipeline(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([1.0, 2.0])}
        result = apply_filters(delta, [])

        torch.testing.assert_close(result.delta["w"], delta["w"])
        self.assertEqual(len(result.audit_trail), 0)

    def test_pipeline_does_not_mutate_original(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        original = torch.tensor([3.0, 4.0])
        delta = {"w": original}
        apply_filters(delta, [{"type": "gradient_clip", "max_norm": 0.1}])

        torch.testing.assert_close(delta["w"], torch.tensor([3.0, 4.0]))


# ---------------------------------------------------------------------------
# Custom filter via registry
# ---------------------------------------------------------------------------


class CustomFilterRegistrationTests(unittest.TestCase):
    """Test user-defined custom filter registration and usage."""

    def setUp(self):
        self._to_cleanup: List[str] = []

    def tearDown(self):
        for name in self._to_cleanup:
            FilterRegistry.unregister(name)

    def test_custom_filter_used_via_dict_config(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        FilterRegistry.register("config_scale", _ConfigReadingFilter)
        self._to_cleanup.append("config_scale")

        delta = {"w": torch.tensor([2.0, 3.0])}
        result = apply_filters(
            delta,
            [
                {"type": "config_scale", "factor": 3.0},
            ],
        )

        torch.testing.assert_close(result.delta["w"], torch.tensor([6.0, 9.0]))
        self.assertEqual(result.audit_trail[0][0], "_ConfigReadingFilter")

    def test_custom_filter_used_as_instance(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([5.0])}
        result = apply_filters(delta, [_DoubleFilter()])

        torch.testing.assert_close(result.delta["w"], torch.tensor([10.0]))

    def test_custom_filter_in_pipeline_with_builtins(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        FilterRegistry.register("double", _DoubleFilter)
        self._to_cleanup.append("double")

        delta = {"w": torch.tensor([3.0, 4.0])}
        result = apply_filters(
            delta,
            [
                {"type": "gradient_clip", "max_norm": 1.0},
                {"type": "double"},
            ],
        )

        # After clip to norm 1.0, then doubled -> norm ~2.0
        clipped_norm = torch.norm(result.delta["w"].float().flatten(), dim=0)
        self.assertAlmostEqual(clipped_norm.item(), 2.0, places=3)
        self.assertEqual(len(result.audit_trail), 2)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class BackwardCompatibilityTests(unittest.TestCase):
    """Test backward-compatible apply_filters in federated_client."""

    def test_wrapper_returns_dict(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from octomil.federated_client import apply_filters as compat_apply_filters

        delta = {"w": torch.tensor([3.0, 4.0])}
        result = compat_apply_filters(
            delta, [{"type": "gradient_clip", "max_norm": 1.0}]
        )

        # Should return a plain dict, not FilterResult
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, FilterResult)
        self.assertIn("w", result)

    def test_wrapper_same_behavior_as_old(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from octomil.federated_client import apply_filters as compat_apply_filters

        delta = {"w": torch.tensor([3.0, 4.0])}
        result = compat_apply_filters(
            delta, [{"type": "gradient_clip", "max_norm": 1.0}]
        )

        clipped_norm = torch.norm(result["w"].float().flatten(), dim=0)
        self.assertAlmostEqual(clipped_norm.item(), 1.0, places=4)

    def test_wrapper_non_tensor_preserved(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from octomil.federated_client import apply_filters as compat_apply_filters

        delta = {"w": torch.tensor([3.0, 4.0]), "metadata": "some_string"}
        result = compat_apply_filters(
            delta, [{"type": "gradient_clip", "max_norm": 1.0}]
        )

        self.assertEqual(result["metadata"], "some_string")

    def test_wrapper_empty_filters(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from octomil.federated_client import apply_filters as compat_apply_filters

        delta = {"w": torch.tensor([1.0, 2.0])}
        result = compat_apply_filters(delta, [])

        torch.testing.assert_close(result["w"], delta["w"])


# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------


class ModuleExportTests(unittest.TestCase):
    """Verify that filter types are accessible from octomil package."""

    def test_filters_accessible_from_package(self):
        from octomil import DataKind, DeltaFilter, FilterRegistry, FilterResult

        self.assertIsNotNone(DataKind)
        self.assertIsNotNone(DeltaFilter)
        self.assertIsNotNone(FilterRegistry)
        self.assertIsNotNone(FilterResult)

    def test_apply_filters_accessible_from_federated_client(self):
        from octomil.federated_client import apply_filters

        self.assertTrue(callable(apply_filters))

    def test_filter_registry_accessible_from_federated_client(self):
        """FilterRegistry re-exported from federated_client for backward compat."""
        from octomil.federated_client import FilterRegistry, FilterResult

        self.assertIsNotNone(FilterRegistry)
        self.assertIsNotNone(FilterResult)


if __name__ == "__main__":
    unittest.main()
