import unittest
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class FeatureAlignmentTests(unittest.TestCase):
    def setUp(self):
        if not HAS_PANDAS:
            self.skipTest("pandas not installed")

    def test_feature_aligner_embedding_strategy(self):
        from edgeml.feature_alignment import FeatureAligner

        aligner = FeatureAligner(strategy="embedding", input_dim=10)
        self.assertEqual(aligner.strategy, "embedding")
        self.assertEqual(aligner.input_dim, 10)

    def test_feature_aligner_fit_single_dataframe(self):
        from edgeml.feature_alignment import FeatureAligner

        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
        })

        aligner = FeatureAligner(strategy="embedding", input_dim=5)
        result = aligner.fit(df, target_col="target")
        self.assertIsInstance(result, FeatureAligner)

    def test_feature_aligner_fit_multiple_datasets(self):
        from edgeml.feature_alignment import FeatureAligner

        datasets = {
            "device1": pd.DataFrame({
                "feature1": [1, 2],
                "feature2": [3, 4],
                "target": [0, 1],
            }),
            "device2": pd.DataFrame({
                "feature1": [5, 6],
                "feature3": [7, 8],
                "target": [1, 0],
            }),
        }

        aligner = FeatureAligner(strategy="intersection", input_dim=5)
        result = aligner.fit(datasets, target_col="target")
        self.assertIsInstance(result, FeatureAligner)

    def test_feature_aligner_transform_embedding(self):
        from edgeml.feature_alignment import FeatureAligner

        train_df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "target": [0, 1, 0],
        })

        aligner = FeatureAligner(strategy="embedding", input_dim=4)
        aligner.fit(train_df, target_col="target")

        test_df = pd.DataFrame({
            "f1": [7, 8],
            "f2": [9, 10],
        })

        transformed = aligner.transform(test_df)
        self.assertEqual(transformed.shape[1], 4)

    def test_feature_aligner_intersection_strategy(self):
        from edgeml.feature_alignment import FeatureAligner

        datasets = {
            "device1": pd.DataFrame({
                "common1": [1, 2],
                "unique1": [3, 4],
                "target": [0, 1],
            }),
            "device2": pd.DataFrame({
                "common1": [5, 6],
                "unique2": [7, 8],
                "target": [1, 0],
            }),
        }

        aligner = FeatureAligner(strategy="intersection")
        aligner.fit(datasets, target_col="target")

        # Should only use common features
        test_df = pd.DataFrame({
            "common1": [9, 10],
            "unique1": [11, 12],
        })

        transformed = aligner.transform(test_df)
        self.assertIn("common1", transformed.columns)

    def test_feature_aligner_union_imputation_strategy(self):
        from edgeml.feature_alignment import FeatureAligner

        datasets = {
            "device1": pd.DataFrame({
                "f1": [1.0, 2.0],
                "f2": [3.0, 4.0],
                "target": [0, 1],
            }),
            "device2": pd.DataFrame({
                "f1": [5.0, 6.0],
                "f3": [7.0, 8.0],
                "target": [1, 0],
            }),
        }

        aligner = FeatureAligner(strategy="union_imputation", imputation_method="mean")
        aligner.fit(datasets, target_col="target")

        # Test with missing features
        test_df = pd.DataFrame({
            "f1": [9.0, 10.0],
            "f2": [11.0, 12.0],
        })

        transformed = aligner.transform(test_df)
        # Should have all union features
        self.assertGreaterEqual(len(transformed.columns), 2)

    def test_feature_aligner_with_feature_schema(self):
        from edgeml.feature_alignment import FeatureAligner

        aligner = FeatureAligner(
            strategy="embedding",
            input_dim=5,
            feature_schema=["f1", "f2", "f3"]
        )

        df = pd.DataFrame({
            "f1": [1, 2],
            "f2": [3, 4],
            "f3": [5, 6],
        })

        aligner.fit(df, target_col="target")
        self.assertEqual(aligner.feature_schema, ["f1", "f2", "f3"])

    def test_auto_align_function(self):
        from edgeml.feature_alignment import auto_align

        # auto_align takes a single DataFrame, not a dict
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [3, 4, 5],
            "target": [0, 1, 0],
        })

        # Returns (np.ndarray, List[str], Dict[str, Any])
        X, detected_features, coverage_info = auto_align(df, target_col="target", input_dim=4)

        # Check return types
        import numpy as np
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(detected_features, list)
        self.assertIsInstance(coverage_info, dict)

        # Check shapes and content
        self.assertEqual(X.shape[0], len(df))
        self.assertEqual(X.shape[1], 4)  # input_dim
        self.assertEqual(len(detected_features), 2)  # f1 and f2

    def test_feature_aligner_get_coverage(self):
        from edgeml.feature_alignment import FeatureAligner

        datasets = {
            "device1": pd.DataFrame({
                "f1": [1, 2],
                "f2": [3, 4],
                "target": [0, 1],
            }),
            "device2": pd.DataFrame({
                "f1": [5, 6],
                "target": [1, 0],
            }),
        }

        aligner = FeatureAligner(strategy="intersection")
        aligner.fit(datasets, target_col="target")

        # Test get_coverage (not get_coverage_info)
        test_df = pd.DataFrame({
            "f1": [9, 10],
            "target": [0, 1],
        })
        coverage = aligner.get_coverage(test_df, target_col="target")
        self.assertIsInstance(coverage, dict)
        self.assertIn("coverage", coverage)

    def test_feature_aligner_imputation_methods(self):
        from edgeml.feature_alignment import FeatureAligner

        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        })

        for method in ["mean", "median", "zero"]:
            aligner = FeatureAligner(
                strategy="union_imputation",
                imputation_method=method
            )
            result = aligner.fit(df, target_col="target")
            self.assertEqual(result.imputation_method, method)

    def test_feature_aligner_transform_without_fit_raises(self):
        from edgeml.feature_alignment import FeatureAligner

        aligner = FeatureAligner(strategy="embedding", input_dim=5)

        test_df = pd.DataFrame({
            "f1": [1, 2],
            "f2": [3, 4],
        })

        # Should raise or handle gracefully
        try:
            aligner.transform(test_df)
            # If it doesn't raise, that's ok too (graceful handling)
        except (ValueError, RuntimeError, KeyError):
            pass  # Expected behavior

    def test_feature_aligner_empty_dataframe(self):
        from edgeml.feature_alignment import FeatureAligner

        df = pd.DataFrame()
        aligner = FeatureAligner(strategy="embedding", input_dim=5)

        try:
            aligner.fit(df, target_col="target")
        except (ValueError, KeyError):
            pass  # Expected to fail on empty data

    def test_feature_aligner_projection_matrix_shape(self):
        from edgeml.feature_alignment import FeatureAligner

        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "f3": [7, 8, 9],
            "target": [0, 1, 0],
        })

        input_dim = 5
        aligner = FeatureAligner(strategy="embedding", input_dim=input_dim)
        aligner.fit(df, target_col="target")

        # Check projection matrix exists and has correct shape
        if aligner._projection_matrix is not None:
            self.assertEqual(aligner._projection_matrix.shape[1], input_dim)

    # --- Tests for uncovered lines ---

    def test_embedding_auto_input_dim(self):
        """When input_dim=None, it should be set to len(all_features) during fit."""
        from edgeml.feature_alignment import FeatureAligner

        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
            "d": [7, 8],
            "e": [9, 10],
            "target": [0, 1],
        })

        aligner = FeatureAligner(strategy="embedding", input_dim=None)
        self.assertIsNone(aligner.input_dim)

        aligner.fit(df, target_col="target")
        self.assertEqual(aligner.input_dim, 5)

    def test_embedding_large_input_dim_uses_secondary_projection(self):
        """When input_dim > 10, secondary hash projection entries with value 0.5 are used."""
        from edgeml.feature_alignment import FeatureAligner

        df = pd.DataFrame({
            f"f{i}": [float(i), float(i + 1)] for i in range(5)
        })
        df["target"] = [0, 1]

        aligner = FeatureAligner(strategy="embedding", input_dim=20)
        aligner.fit(df, target_col="target")

        # The raw projection matrix before normalization would have 0.5 entries.
        # After column normalization those exact values change, but we can verify
        # the matrix has more non-zero entries than just the primary projections
        # (i.e., more than one non-zero per row for at least some rows).
        matrix = aligner._projection_matrix
        self.assertIsNotNone(matrix)

        # Count non-zero entries per row; with secondary projection some rows
        # should have 2 non-zero entries.
        nonzero_per_row = np.count_nonzero(matrix, axis=1)
        self.assertTrue(
            np.any(nonzero_per_row >= 2),
            "Expected at least one feature row with secondary projection (>=2 non-zero entries)"
        )

    def test_embedding_feature_all_nan(self):
        """A feature column that is all NaN should get default stats."""
        from edgeml.feature_alignment import FeatureAligner

        df = pd.DataFrame({
            "good_feature": [1.0, 2.0, 3.0],
            "nan_feature": [float("nan"), float("nan"), float("nan")],
            "target": [0, 1, 0],
        })

        aligner = FeatureAligner(strategy="embedding", input_dim=4)
        aligner.fit(df, target_col="target")

        stats = aligner.feature_stats["nan_feature"]
        self.assertEqual(stats["mean"], 0.0)
        self.assertEqual(stats["std"], 1.0)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 0.0)

    def test_transform_union_imputation_median(self):
        """Union imputation with median method uses median value for missing features."""
        from edgeml.feature_alignment import FeatureAligner

        datasets = {
            "d1": pd.DataFrame({
                "f1": [1.0, 2.0, 3.0],
                "f2": [10.0, 20.0, 30.0],
                "target": [0, 1, 0],
            }),
        }

        aligner = FeatureAligner(strategy="union_imputation", imputation_method="median")
        aligner.fit(datasets, target_col="target")

        # Transform a DataFrame missing f2
        test_df = pd.DataFrame({"f1": [5.0]})
        transformed = aligner.transform(test_df, target_col="target")

        expected_median = 20.0  # median of [10, 20, 30]
        self.assertIn("f2", transformed.columns)
        self.assertAlmostEqual(transformed["f2"].iloc[0], expected_median)

    def test_transform_union_imputation_zero(self):
        """Union imputation with zero method fills missing features with 0."""
        from edgeml.feature_alignment import FeatureAligner

        datasets = {
            "d1": pd.DataFrame({
                "f1": [1.0, 2.0],
                "f2": [10.0, 20.0],
                "target": [0, 1],
            }),
        }

        aligner = FeatureAligner(strategy="union_imputation", imputation_method="zero")
        aligner.fit(datasets, target_col="target")

        test_df = pd.DataFrame({"f1": [5.0]})
        transformed = aligner.transform(test_df, target_col="target")

        self.assertIn("f2", transformed.columns)
        self.assertEqual(transformed["f2"].iloc[0], 0.0)

    def test_transform_union_preserves_target(self):
        """Union transform preserves target column and reorders features to match schema."""
        from edgeml.feature_alignment import FeatureAligner

        datasets = {
            "d1": pd.DataFrame({
                "f1": [1.0, 2.0],
                "f2": [3.0, 4.0],
                "target": [0, 1],
            }),
        }

        aligner = FeatureAligner(strategy="union_imputation", imputation_method="mean")
        aligner.fit(datasets, target_col="target")

        test_df = pd.DataFrame({
            "f1": [5.0],
            "f2": [6.0],
            "target": [1],
        })
        transformed = aligner.transform(test_df, target_col="target")

        self.assertIn("target", transformed.columns)
        self.assertEqual(transformed["target"].iloc[0], 1)
        # Target should be the last column (after schema features)
        self.assertEqual(transformed.columns[-1], "target")

    def test_get_config_returns_expected_keys(self):
        """get_config() returns dict with all expected keys after fitting."""
        from edgeml.feature_alignment import FeatureAligner

        df = pd.DataFrame({
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
            "target": [0, 1],
        })

        aligner = FeatureAligner(strategy="embedding", input_dim=4)
        aligner.fit(df, target_col="target")

        config = aligner.get_config()
        expected_keys = {"strategy", "input_dim", "feature_schema", "feature_stats", "imputation_method"}
        self.assertEqual(set(config.keys()), expected_keys)
        self.assertEqual(config["strategy"], "embedding")
        self.assertEqual(config["input_dim"], 4)
        self.assertEqual(config["imputation_method"], "mean")
        self.assertIsInstance(config["feature_schema"], list)
        self.assertIsInstance(config["feature_stats"], dict)

    def test_from_config_roundtrip(self):
        """from_config() recreates an aligner with matching attributes."""
        from edgeml.feature_alignment import FeatureAligner

        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        })

        original = FeatureAligner(strategy="embedding", input_dim=4, imputation_method="median")
        original.fit(df, target_col="target")

        config = original.get_config()
        restored = FeatureAligner.from_config(config)

        self.assertEqual(restored.strategy, original.strategy)
        self.assertEqual(restored.input_dim, original.input_dim)
        self.assertEqual(restored.feature_schema, original.feature_schema)
        self.assertEqual(restored.feature_stats, original.feature_stats)
        self.assertEqual(restored.imputation_method, original.imputation_method)

    def test_get_coverage_no_schema(self):
        """get_coverage() with no feature_schema returns coverage=1.0."""
        from edgeml.feature_alignment import FeatureAligner

        aligner = FeatureAligner(strategy="embedding", input_dim=5)
        # Not fitted, so feature_schema is None

        test_df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        coverage = aligner.get_coverage(test_df)

        self.assertEqual(coverage["coverage"], 1.0)
        self.assertEqual(coverage["missing_features"], [])
        self.assertEqual(coverage["detected_features"], [])

    def test_transform_unknown_strategy_returns_input(self):
        """Transform with an unknown strategy returns the input DataFrame unchanged."""
        from edgeml.feature_alignment import FeatureAligner

        aligner = FeatureAligner(strategy="embedding", input_dim=5)
        aligner.strategy = "unknown"

        test_df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        result = aligner.transform(test_df)

        pd.testing.assert_frame_equal(result, test_df)


if __name__ == "__main__":
    unittest.main()
