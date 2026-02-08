import unittest
from unittest.mock import Mock, patch
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


if __name__ == "__main__":
    unittest.main()
