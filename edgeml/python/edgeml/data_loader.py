"""
Data Loading Module for EdgeML SDK

Handles loading data from various sources:
- Local files (CSV, Parquet, JSON)
- Cloud storage (S3, GCS, Azure Blob)
- DataFrames (passthrough)

Credentials are read from environment variables:
- AWS: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
- GCS: GOOGLE_APPLICATION_CREDENTIALS
- Azure: AZURE_STORAGE_CONNECTION_STRING
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, List, Any, Dict, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Type alias for data sources
DataSource = Union[str, Path, "pd.DataFrame"]


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


def load_data(source: DataSource) -> "pd.DataFrame":
    """
    Load data from URI, path, or DataFrame.

    Supports:
        - pd.DataFrame (passthrough)
        - Local files: /path/to/data.csv, ./data.parquet
        - S3: s3://bucket/path/data.parquet
        - GCS: gs://bucket/path/data.csv
        - Azure: az://container/path/data.parquet
        - HTTP: https://example.com/data.csv

    Credentials from environment:
        - AWS: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or IAM role)
        - GCS: GOOGLE_APPLICATION_CREDENTIALS
        - Azure: AZURE_STORAGE_CONNECTION_STRING

    Args:
        source: DataFrame, file path, or URI

    Returns:
        pandas DataFrame

    Raises:
        DataLoadError: If loading fails

    Example:
        df = load_data("s3://my-bucket/patients.parquet")
        df = load_data("/local/path/data.csv")
        df = load_data(existing_dataframe)
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise DataLoadError("pandas is required for data loading") from e

    # Passthrough for DataFrames
    if isinstance(source, pd.DataFrame):
        return source

    source_str = str(source)

    # Determine format from extension
    format_type = _detect_format(source_str)

    try:
        if format_type == "parquet":
            return _load_parquet(source_str)
        elif format_type == "json":
            return _load_json(source_str)
        else:  # csv or unknown
            return _load_csv(source_str)
    except Exception as e:
        raise DataLoadError(f"Failed to load data from '{source_str}': {e}") from e


def _detect_format(path: str) -> str:
    """Detect file format from extension."""
    path_lower = path.lower()

    # Strip query params for URL detection
    base_path = path_lower.split('?')[0]

    if base_path.endswith('.parquet') or base_path.endswith('.pq'):
        return "parquet"
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        return "json"
    elif base_path.endswith('.csv'):
        return "csv"
    else:
        # Default to CSV for unknown formats
        logger.debug(f"Unknown format for '{path}', defaulting to CSV")
        return "csv"


def _load_csv(path: str) -> "pd.DataFrame":
    """Load CSV file (local or remote)."""
    import pandas as pd

    storage_options = _get_storage_options(path)
    return pd.read_csv(path, storage_options=storage_options)


def _load_parquet(path: str) -> "pd.DataFrame":
    """Load Parquet file (local or remote)."""
    import pandas as pd

    storage_options = _get_storage_options(path)
    return pd.read_parquet(path, storage_options=storage_options)


def _load_json(path: str) -> "pd.DataFrame":
    """Load JSON file (local or remote)."""
    import pandas as pd

    storage_options = _get_storage_options(path)

    # Try lines format first (JSONL), then standard JSON
    try:
        return pd.read_json(path, lines=True, storage_options=storage_options)
    except ValueError:
        return pd.read_json(path, storage_options=storage_options)


def _get_storage_options(path: str) -> Optional[Dict[str, Any]]:
    """Get storage options for cloud paths."""
    import os

    if path.startswith('s3://'):
        # AWS S3 - credentials from environment
        options: Dict[str, Any] = {}
        if os.environ.get('AWS_ACCESS_KEY_ID'):
            options['key'] = os.environ['AWS_ACCESS_KEY_ID']
        if os.environ.get('AWS_SECRET_ACCESS_KEY'):
            options['secret'] = os.environ['AWS_SECRET_ACCESS_KEY']
        if os.environ.get('AWS_SESSION_TOKEN'):
            options['token'] = os.environ['AWS_SESSION_TOKEN']
        if os.environ.get('AWS_ENDPOINT_URL'):
            options['endpoint_url'] = os.environ['AWS_ENDPOINT_URL']
        return options if options else None

    elif path.startswith('gs://'):
        # Google Cloud Storage - uses GOOGLE_APPLICATION_CREDENTIALS env var
        # gcsfs reads this automatically
        return None

    elif path.startswith('az://') or path.startswith('abfs://'):
        # Azure Blob Storage
        options = {}
        if os.environ.get('AZURE_STORAGE_CONNECTION_STRING'):
            options['connection_string'] = os.environ['AZURE_STORAGE_CONNECTION_STRING']
        if os.environ.get('AZURE_STORAGE_ACCOUNT_NAME'):
            options['account_name'] = os.environ['AZURE_STORAGE_ACCOUNT_NAME']
        if os.environ.get('AZURE_STORAGE_ACCOUNT_KEY'):
            options['account_key'] = os.environ['AZURE_STORAGE_ACCOUNT_KEY']
        return options if options else None

    else:
        # Local file or HTTP - no special options
        return None


def validate_target(
    df: "pd.DataFrame",
    target_col: str,
    output_type: str = "binary",
    output_dim: int = 1,
) -> "pd.DataFrame":
    """
    Validate and prepare target column.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        output_type: Expected output type ("binary", "multiclass", "regression")
        output_dim: Expected output dimension

    Returns:
        DataFrame with validated/transformed target

    Raises:
        ValueError: If target column missing or incompatible
    """
    import pandas as pd
    import numpy as np

    # Check column exists
    if target_col not in df.columns:
        available = sorted([c for c in df.columns])
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {available}"
        )

    df = df.copy()
    y = df[target_col]

    # Auto-encode string labels
    if y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype):
        unique_vals = sorted(y.dropna().unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        df[target_col] = y.map(mapping)
        logger.info(f"Auto-encoded target column: {mapping}")
        y = df[target_col]

    # Validate based on output type
    unique_values = sorted(y.dropna().unique())
    n_classes = len(unique_values)

    if output_type == "binary":
        if n_classes > 2:
            raise ValueError(
                f"Model expects binary output, but target has {n_classes} classes: {unique_values}"
            )

        # Normalize to 0/1 if needed
        if set(unique_values) == {-1, 1}:
            df[target_col] = (y + 1) / 2
            logger.info("Normalized target: {-1: 0, 1: 1}")
        elif set(unique_values) != {0, 1} and n_classes == 2:
            # Map to 0/1
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            df[target_col] = y.map(mapping)
            logger.info(f"Mapped target to binary: {mapping}")

    elif output_type == "multiclass":
        expected_classes = output_dim
        if n_classes != expected_classes:
            raise ValueError(
                f"Model expects {expected_classes} classes, but target has {n_classes}: {unique_values}"
            )

    # For regression, no validation needed

    return df


def prepare_data(
    source: DataSource,
    target_col: str,
    model_architecture: Optional[Dict[str, Any]] = None,
) -> tuple["pd.DataFrame", List[str], int]:
    """
    Load and prepare data for training.

    Args:
        source: Data source (URI, path, or DataFrame)
        target_col: Local target column name
        model_architecture: Model architecture dict with output_type, output_dim

    Returns:
        Tuple of (prepared_df, feature_columns, sample_count)
    """
    # Load data
    df = load_data(source)

    # Validate target if architecture provided
    if model_architecture:
        df = validate_target(
            df,
            target_col=target_col,
            output_type=model_architecture.get("output_type", "binary"),
            output_dim=model_architecture.get("output_dim", 1),
        )
    elif target_col not in df.columns:
        available = sorted([c for c in df.columns])
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {available}"
        )

    # Extract feature columns
    feature_cols = [c for c in df.columns if c != target_col]
    sample_count = len(df)

    logger.info(f"Prepared data: {sample_count} samples, {len(feature_cols)} features")

    return df, feature_cols, sample_count
