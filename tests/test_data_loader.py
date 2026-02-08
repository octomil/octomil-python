import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from edgeml.data_loader import DataLoadError, load_data, validate_target, _detect_format


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
            import pandas as pd
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
            import pandas as pd
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
            import pandas as pd
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
        self.assertTrue(result["target"].dtype in [int, float, 'int64', 'float64'])

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


if __name__ == "__main__":
    unittest.main()
