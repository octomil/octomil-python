import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from octomil.data_loader import DataLoadError, load_data, validate_target


class DataLoaderTests(unittest.TestCase):
    def test_load_data_passthrough_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = load_data(df)
        self.assertIs(result, df)

    def test_validate_target_with_valid_column(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        validate_target(df, "target")  # Should not raise

    def test_validate_target_with_missing_column(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        with self.assertRaises(DataLoadError) as ctx:
            validate_target(df, "target")
        self.assertIn("Target column 'target' not found", str(ctx.exception))

    def test_validate_target_with_none_target(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({"feature1": [1, 2]})
        validate_target(df, None)  # Should not raise

    def test_detect_format_csv(self):
        from octomil.data_loader import _detect_format

        self.assertEqual(_detect_format("data.csv"), "csv")
        self.assertEqual(_detect_format("/path/to/data.csv"), "csv")
        self.assertEqual(_detect_format("s3://bucket/data.csv"), "csv")

    def test_detect_format_parquet(self):
        from octomil.data_loader import _detect_format

        self.assertEqual(_detect_format("data.parquet"), "parquet")
        self.assertEqual(_detect_format("data.parq"), "parquet")
        self.assertEqual(_detect_format("gs://bucket/data.parquet"), "parquet")

    def test_detect_format_json(self):
        from octomil.data_loader import _detect_format

        self.assertEqual(_detect_format("data.json"), "json")
        self.assertEqual(_detect_format("data.jsonl"), "json")

    def test_detect_format_unknown(self):
        from octomil.data_loader import _detect_format

        self.assertEqual(_detect_format("data.txt"), "csv")  # Default to CSV
        self.assertEqual(_detect_format("data"), "csv")


if __name__ == "__main__":
    unittest.main()
