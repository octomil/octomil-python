"""
Feature Alignment Module

Handles heterogeneous feature sets across devices in federated learning.
Default behavior: Embedding projection (zero config, just works).

Strategies:
1. Embedding (default): Auto-project any feature set to model's expected input
2. Union with Imputation: Use all features, impute missing (explicit opt-in)
3. Intersection: Use only common features (explicit opt-in)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal, Any, Union, Set
from hashlib import md5

try:
    from sklearn.impute import SimpleImputer
except ImportError:
    SimpleImputer = None  # type: ignore[assignment, misc]  # Optional dependency

# Type aliases for clarity
FeatureAlignmentStrategy = Literal["embedding", "union_imputation", "intersection"]
ImputationMethod = Literal["mean", "median", "zero"]
FeatureStats = Dict[str, float]  # {'mean': 0.0, 'std': 1.0, ...}
DatasetDict = Dict[str, pd.DataFrame]  # {device_id: DataFrame}
CoverageInfo = Dict[str, Any]  # Coverage metrics dictionary


class FeatureAligner:
    """
    Aligns features across heterogeneous datasets in federated learning.

    Default behavior (embedding): Zero config, just works.
    - Auto-detects features from data
    - Projects any feature subset to model's expected input dimension
    - No errors, no coverage thresholds

    Example (simple - recommended):
        aligner = FeatureAligner(input_dim=30)
        aligned_data = aligner.transform(my_dataframe)
        # That's it. Any feature set works.

    Example (advanced - explicit control):
        aligner = FeatureAligner(
            strategy="intersection",  # Only common features
            feature_schema=["f1", "f2", "f3"]
        )
    """

    def __init__(
        self,
        strategy: FeatureAlignmentStrategy = "embedding",
        input_dim: Optional[int] = None,
        feature_schema: Optional[List[str]] = None,
        imputation_method: ImputationMethod = "mean",
    ) -> None:
        """
        Initialize feature aligner.

        Args:
            strategy: Alignment strategy (default: "embedding" for zero-config)
            input_dim: Model's expected input dimension (for embedding projection)
            feature_schema: Expected features (optional, auto-detected if not provided)
            imputation_method: For union_imputation strategy only
        """
        self.strategy: FeatureAlignmentStrategy = strategy
        self.input_dim: Optional[int] = input_dim
        self.feature_schema: Optional[List[str]] = feature_schema
        self.imputation_method: ImputationMethod = imputation_method
        self.feature_stats: Dict[str, FeatureStats] = {}
        self._projection_matrix: Optional[np.ndarray] = None
        self._feature_to_index: Dict[str, int] = {}

    def fit(
        self,
        datasets: Union[DatasetDict, pd.DataFrame],
        target_col: str = "diagnosis",
    ) -> FeatureAligner:
        """
        Learn feature alignment from data.

        For embedding strategy: builds projection matrix
        For union/intersection: learns feature schema and stats

        Args:
            datasets: Single DataFrame or dict of {device_id: DataFrame}
            target_col: Target column to exclude from features
        """
        # Normalize input to dict format
        if isinstance(datasets, pd.DataFrame):
            datasets = {"default": datasets}

        # Collect all features across datasets
        all_features = set()
        for df in datasets.values():
            features = set(df.columns) - {target_col}
            all_features.update(features)

        all_features = sorted(list(all_features))

        if self.strategy == "embedding":
            self._fit_embedding(datasets, all_features, target_col)
        elif self.strategy == "union_imputation":
            self._fit_union(datasets, all_features, target_col)
        elif self.strategy == "intersection":
            self._fit_intersection(datasets, target_col)

        return self

    def _fit_embedding(
        self,
        datasets: DatasetDict,
        all_features: List[str],
        target_col: str,
    ) -> None:
        """Fit embedding projection matrix."""
        # Create feature-to-index mapping using consistent hashing
        self._feature_to_index = {f: i for i, f in enumerate(all_features)}

        # Determine output dimension
        if self.input_dim is None:
            self.input_dim = len(all_features)

        # Build projection matrix using feature name hashing
        # Each feature maps to a sparse vector in the output space
        n_features = len(all_features)
        self._projection_matrix = np.zeros((n_features, self.input_dim))

        for i, feature in enumerate(all_features):
            # Use feature name hash to determine output positions
            # This ensures consistent projection across devices
            hash_val = int(md5(feature.encode()).hexdigest(), 16)

            # Primary position
            primary_idx = hash_val % self.input_dim
            self._projection_matrix[i, primary_idx] = 1.0

            # Secondary positions for better distribution (optional)
            if self.input_dim > 10:
                secondary_idx = (hash_val >> 8) % self.input_dim
                if secondary_idx != primary_idx:
                    self._projection_matrix[i, secondary_idx] = 0.5

        # Normalize columns
        col_norms = np.linalg.norm(self._projection_matrix, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1  # Avoid division by zero
        self._projection_matrix = self._projection_matrix / col_norms

        # Learn feature statistics for any needed normalization
        for feature in all_features:
            values = []
            for df in datasets.values():
                if feature in df.columns:
                    values.extend(df[feature].dropna().values)

            if values:
                self.feature_stats[feature] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)) or 1.0,
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
            else:
                self.feature_stats[feature] = {
                    'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 0.0
                }

        self.feature_schema = all_features

    def _fit_union(
        self,
        datasets: DatasetDict,
        all_features: List[str],
        target_col: str,
    ) -> None:
        """Fit union with imputation strategy."""
        self.feature_schema = all_features

        # Learn imputation statistics
        for feature in all_features:
            values = []
            for df in datasets.values():
                if feature in df.columns:
                    values.extend(df[feature].dropna().values)

            if values:
                self.feature_stats[feature] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                }
            else:
                self.feature_stats[feature] = {
                    'mean': 0.0, 'median': 0.0, 'std': 1.0
                }

    def _fit_intersection(
        self,
        datasets: DatasetDict,
        target_col: str,
    ) -> None:
        """Fit intersection strategy - only common features."""
        feature_sets = [
            set(df.columns) - {target_col}
            for df in datasets.values()
        ]
        common_features = set.intersection(*feature_sets) if feature_sets else set()
        self.feature_schema = sorted(list(common_features))

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = "diagnosis"
    ) -> pd.DataFrame:
        """
        Transform data to aligned feature space.

        For embedding: projects to fixed output dimension
        For union: adds missing features with imputation
        For intersection: keeps only common features

        Args:
            df: Input DataFrame
            target_col: Target column to preserve

        Returns:
            Aligned DataFrame
        """
        if self.strategy == "embedding":
            return self._transform_embedding(df, target_col)
        elif self.strategy == "union_imputation":
            return self._transform_union(df, target_col)
        elif self.strategy == "intersection":
            return self._transform_intersection(df, target_col)
        else:
            return df

    def _transform_embedding(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """Transform using embedding projection."""
        # Get feature columns (excluding target)
        feature_cols = [c for c in df.columns if c != target_col]

        # Build input matrix aligned to known features
        n_samples = len(df)
        n_known_features = len(self._feature_to_index)
        input_matrix = np.zeros((n_samples, n_known_features))

        for col in feature_cols:
            if col in self._feature_to_index:
                idx = self._feature_to_index[col]
                # Normalize using learned stats
                stats = self.feature_stats.get(col, {'mean': 0, 'std': 1})
                values = (df[col].values - stats['mean']) / stats['std']
                input_matrix[:, idx] = values

        # Project to output dimension
        output_matrix = input_matrix @ self._projection_matrix

        # Create output DataFrame
        output_cols = [f"emb_{i}" for i in range(self.input_dim)]
        result = pd.DataFrame(output_matrix, columns=output_cols, index=df.index)

        # Preserve target column if present
        if target_col in df.columns:
            result[target_col] = df[target_col].values

        return result

    def _transform_union(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """Transform using union with imputation."""
        result = df.copy()

        for feature in self.feature_schema:
            if feature not in result.columns:
                # Impute missing feature
                if self.imputation_method == "mean":
                    fill_value = self.feature_stats[feature]['mean']
                elif self.imputation_method == "median":
                    fill_value = self.feature_stats[feature]['median']
                else:
                    fill_value = 0.0
                result[feature] = fill_value

        # Reorder columns to match schema
        cols = self.feature_schema.copy()
        if target_col in result.columns:
            cols.append(target_col)
        result = result[[c for c in cols if c in result.columns]]

        return result

    def _transform_intersection(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """Transform using intersection - keep only common features."""
        cols = [c for c in self.feature_schema if c in df.columns]
        if target_col in df.columns:
            cols.append(target_col)
        return df[cols].copy()

    def get_coverage(self, df: pd.DataFrame, target_col: str = "diagnosis") -> CoverageInfo:
        """
        Get feature coverage info for a dataset.

        Returns coverage metrics for observability (not validation).
        """
        if not self.feature_schema:
            return {"coverage": 1.0, "missing_features": [], "detected_features": []}

        detected = [c for c in df.columns if c != target_col]
        schema_set = set(self.feature_schema)
        detected_set = set(detected)

        common = schema_set & detected_set
        missing = sorted(list(schema_set - detected_set))
        extra = sorted(list(detected_set - schema_set))

        coverage = len(common) / len(schema_set) if schema_set else 1.0

        return {
            "coverage": coverage,
            "coverage_percent": round(coverage * 100, 1),
            "detected_features": detected,
            "missing_features": missing,
            "extra_features": extra,
            "common_count": len(common),
            "total_expected": len(schema_set),
        }

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        return {
            'strategy': self.strategy,
            'input_dim': self.input_dim,
            'feature_schema': self.feature_schema,
            'feature_stats': self.feature_stats,
            'imputation_method': self.imputation_method,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> FeatureAligner:
        """Create aligner from configuration."""
        aligner = cls(
            strategy=config.get('strategy', 'embedding'),
            input_dim=config.get('input_dim'),
            feature_schema=config.get('feature_schema'),
            imputation_method=config.get('imputation_method', 'mean'),
        )
        aligner.feature_stats = config.get('feature_stats', {})
        return aligner


def auto_align(
    data: pd.DataFrame,
    input_dim: int,
    target_col: str = "diagnosis",
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Zero-config feature alignment.

    Just pass your data, get aligned features back.

    Args:
        data: Input DataFrame with any features
        input_dim: Model's expected input dimension
        target_col: Target column name

    Returns:
        Tuple of (aligned_features, detected_features, coverage_info)

    Example:
        X, features, info = auto_align(my_df, input_dim=30)
        # X is now (n_samples, 30) regardless of original features
    """
    aligner = FeatureAligner(strategy="embedding", input_dim=input_dim)
    aligner.fit(data, target_col=target_col)

    aligned_df = aligner.transform(data, target_col=target_col)
    feature_cols = [c for c in aligned_df.columns if c != target_col]

    X = aligned_df[feature_cols].values
    detected = [c for c in data.columns if c != target_col]
    coverage = aligner.get_coverage(data, target_col)

    return X, detected, coverage
