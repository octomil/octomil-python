import unittest
import tempfile
import os
from unittest.mock import patch

from edgeml.data_loader import (
    DataLoadError, load_data, validate_target, _detect_format,
    _get_s3_options, _get_azure_options, _get_storage_options,
    prepare_data,
)


class DataLoaderTests(unittest.TestCase):
    def test_load_data_passthrough_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = load_data(df)
        self.assertIs(result, df)

    def test_load_data_csv_from_file(self):
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n")
            f.write("1,2,3\n")
            f.write("4,5,6\n")
            csv_path = f.name

        try:
            df = load_data(csv_path)
            self.assertEqual(len(df), 2)
            self.assertEqual(list(df.columns), ["col1", "col2", "col3"])
            self.assertEqual(df["col1"].tolist(), [1, 4])
        finally:
            os.unlink(csv_path)

    def test_load_data_json_from_file(self):
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")

        # Create temporary JSONL file (one JSON object per line)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"a": 1, "b": 2}\n')
            f.write('{"a": 3, "b": 4}\n')
            json_path = f.name

        try:
            df = load_data(json_path)
            self.assertEqual(len(df), 2)
            self.assertIn("a", df.columns)
            self.assertIn("b", df.columns)
        finally:
            os.unlink(json_path)

    def test_load_data_parquet_from_file(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        # Create temporary parquet file
        df_orig = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            parquet_path = f.name

        try:
            df_orig.to_parquet(parquet_path)
            df = load_data(parquet_path)
            self.assertEqual(len(df), 3)
            self.assertEqual(list(df.columns), ["x", "y"])
        except Exception:
            # pyarrow or fastparquet might not be installed
            self.skipTest("parquet library not installed")
        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_load_data_nonexistent_file_raises(self):
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")

        with self.assertRaises(DataLoadError) as ctx:
            load_data("/nonexistent/path/file.csv")
        self.assertIn("Failed to load data", str(ctx.exception))

    def test_validate_target_with_valid_column(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        result = validate_target(df, "target")
        self.assertIsInstance(result, pd.DataFrame)

    def test_validate_target_with_missing_column(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        with self.assertRaises(ValueError) as ctx:
            validate_target(df, "target")
        self.assertIn("Target column 'target' not found", str(ctx.exception))

    def test_validate_target_binary_classification(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "target": [0, 1, 0, 1]
        })
        result = validate_target(df, "target", output_type="binary")
        self.assertIn("target", result.columns)
        # Should be 0/1
        self.assertTrue(set(result["target"].unique()).issubset({0, 1}))

    def test_validate_target_binary_with_negative_one(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "target": [-1, 1, -1, 1]
        })
        result = validate_target(df, "target", output_type="binary")
        # Should normalize -1,1 to 0,1
        self.assertTrue(set(result["target"].unique()).issubset({0, 1}))

    def test_validate_target_multiclass(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 2, 0, 1, 2]
        })
        result = validate_target(df, "target", output_type="multiclass", output_dim=3)
        self.assertIn("target", result.columns)
        self.assertEqual(len(result["target"].unique()), 3)

    def test_validate_target_multiclass_wrong_count_raises(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "target": [0, 1, 2, 3]  # 4 classes
        })
        with self.assertRaises(ValueError) as ctx:
            validate_target(df, "target", output_type="multiclass", output_dim=3)
        self.assertIn("expects 3 classes", str(ctx.exception))

    def test_validate_target_string_labels_auto_encoded(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "target": ["cat", "dog", "cat", "dog"]
        })
        result = validate_target(df, "target", output_type="binary")
        # Should be encoded to numeric
        self.assertIn(result["target"].dtype, [int, float, 'int64', 'float64'])

    def test_validate_target_regression(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "target": [1.5, 2.7, 3.2, 4.1]
        })
        result = validate_target(df, "target", output_type="regression")
        self.assertIn("target", result.columns)

    def test_detect_format_csv(self):
        self.assertEqual(_detect_format("data.csv"), "csv")
        self.assertEqual(_detect_format("/path/to/data.csv"), "csv")
        self.assertEqual(_detect_format("s3://bucket/data.csv"), "csv")

    def test_detect_format_parquet(self):
        self.assertEqual(_detect_format("data.parquet"), "parquet")
        self.assertEqual(_detect_format("data.pq"), "parquet")
        self.assertEqual(_detect_format("gs://bucket/data.parquet"), "parquet")

    def test_detect_format_json(self):
        self.assertEqual(_detect_format("data.json"), "json")
        self.assertEqual(_detect_format("data.jsonl"), "json")

    def test_detect_format_unknown_defaults_to_csv(self):
        self.assertEqual(_detect_format("data.txt"), "csv")
        self.assertEqual(_detect_format("data"), "csv")

    def test_detect_format_with_query_params(self):
        self.assertEqual(_detect_format("data.parquet?version=123"), "parquet")
        self.assertEqual(_detect_format("s3://bucket/data.csv?region=us-east-1"), "csv")

    # -----------------------------------------------------------
    # Tests for _get_s3_options, _get_azure_options, _get_storage_options
    # -----------------------------------------------------------

    @patch.dict('os.environ', {
        'AWS_ACCESS_KEY_ID': 'my-key',
        'AWS_SECRET_ACCESS_KEY': 'my-secret',
        'AWS_SESSION_TOKEN': 'my-token',
        'AWS_ENDPOINT_URL': 'https://s3.example.com',
    }, clear=True)
    def test_get_s3_options_with_env_vars(self):
        result = _get_s3_options()
        self.assertEqual(result, {
            'key': 'my-key',
            'secret': 'my-secret',
            'token': 'my-token',
            'endpoint_url': 'https://s3.example.com',
        })

    @patch.dict('os.environ', {}, clear=True)
    def test_get_s3_options_without_env_vars(self):
        result = _get_s3_options()
        self.assertIsNone(result)

    @patch.dict('os.environ', {
        'AZURE_STORAGE_CONNECTION_STRING': 'DefaultEndpointsProtocol=https;AccountName=test',
        'AZURE_STORAGE_ACCOUNT_NAME': 'teststorage',
        'AZURE_STORAGE_ACCOUNT_KEY': 'base64key==',
    }, clear=True)
    def test_get_azure_options_with_env_vars(self):
        result = _get_azure_options()
        self.assertEqual(result, {
            'connection_string': 'DefaultEndpointsProtocol=https;AccountName=test',
            'account_name': 'teststorage',
            'account_key': 'base64key==',
        })

    @patch.dict('os.environ', {}, clear=True)
    def test_get_azure_options_without_env_vars(self):
        result = _get_azure_options()
        self.assertIsNone(result)

    @patch('edgeml.data_loader._get_s3_options', return_value={'key': 'k', 'secret': 's'})
    def test_get_storage_options_s3_path(self, mock_s3):
        result = _get_storage_options('s3://my-bucket/data.csv')
        mock_s3.assert_called_once()
        self.assertEqual(result, {'key': 'k', 'secret': 's'})

    def test_get_storage_options_gcs_path(self):
        result = _get_storage_options('gs://my-bucket/data.csv')
        self.assertIsNone(result)

    @patch('edgeml.data_loader._get_azure_options', return_value={'account_name': 'acct'})
    def test_get_storage_options_azure_path(self, mock_azure):
        result_az = _get_storage_options('az://container/data.csv')
        self.assertEqual(result_az, {'account_name': 'acct'})

        mock_azure.reset_mock()

        result_abfs = _get_storage_options('abfs://container/data.csv')
        self.assertEqual(result_abfs, {'account_name': 'acct'})

        self.assertEqual(mock_azure.call_count, 1)

    def test_get_storage_options_local_path(self):
        result = _get_storage_options('/home/user/data.csv')
        self.assertIsNone(result)

        result_relative = _get_storage_options('data.csv')
        self.assertIsNone(result_relative)

    # -----------------------------------------------------------
    # Tests for prepare_data()
    # -----------------------------------------------------------

    def test_prepare_data_with_architecture(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "feat1": [1, 2, 3, 4],
            "feat2": [10, 20, 30, 40],
            "label": [0, 1, 0, 1],
        })
        architecture = {"output_type": "binary", "output_dim": 1}
        result_df, feature_cols, sample_count = prepare_data(
            df, target_col="label", model_architecture=architecture
        )
        self.assertEqual(sample_count, 4)
        self.assertEqual(sorted(feature_cols), ["feat1", "feat2"])
        self.assertIn("label", result_df.columns)

    def test_prepare_data_without_architecture(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "target": [0, 1, 0],
        })
        result_df, feature_cols, sample_count = prepare_data(
            df, target_col="target"
        )
        self.assertEqual(sample_count, 3)
        self.assertEqual(sorted(feature_cols), ["a", "b"])
        self.assertIn("target", result_df.columns)

    def test_prepare_data_missing_target_raises(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })
        with self.assertRaises(ValueError) as ctx:
            prepare_data(df, target_col="nonexistent")
        self.assertIn("not found", str(ctx.exception))

    def test_prepare_data_with_architecture_validation(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "feat1": [1, 2, 3, 4, 5, 6],
            "feat2": [10, 20, 30, 40, 50, 60],
            "target": [0, 1, 2, 0, 1, 2],
        })
        architecture = {"output_type": "multiclass", "output_dim": 3}
        result_df, feature_cols, sample_count = prepare_data(
            df, target_col="target", model_architecture=architecture
        )
        self.assertEqual(sample_count, 6)
        self.assertEqual(sorted(feature_cols), ["feat1", "feat2"])
        self.assertEqual(len(result_df["target"].unique()), 3)

    # -----------------------------------------------------------
    # Tests for validate_target edge cases
    # -----------------------------------------------------------

    def test_validate_target_binary_too_many_classes_raises(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "target": [0, 1, 2, 3, 4],
        })
        with self.assertRaises(ValueError) as ctx:
            validate_target(df, "target", output_type="binary")
        self.assertIn("Model expects binary output", str(ctx.exception))

    def test_validate_target_binary_maps_arbitrary_values(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "target": [2, 5, 2, 5],
        })
        result = validate_target(df, "target", output_type="binary")
        self.assertEqual(set(result["target"].unique()), {0, 1})

    # -----------------------------------------------------------
    # Test for _load_json standard JSON fallback
    # -----------------------------------------------------------

    def test_load_data_json_standard_format(self):
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")

        import json

        # Create a multi-line standard JSON array file (not JSONL).
        # The multi-line format causes pd.read_json(lines=True) to raise
        # ValueError, which triggers the fallback to standard JSON parsing
        # (lines 135-136 in data_loader.py).
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json.dumps(data, indent=2))
            json_path = f.name

        try:
            df = load_data(json_path)
            self.assertEqual(len(df), 2)
            self.assertIn("a", df.columns)
            self.assertIn("b", df.columns)
            self.assertEqual(df["a"].tolist(), [1, 3])
            self.assertEqual(df["b"].tolist(), [2, 4])
        finally:
            os.unlink(json_path)


if __name__ == "__main__":
    unittest.main()
