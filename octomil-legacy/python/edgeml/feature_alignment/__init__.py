"""
Feature Alignment Module for Octomil SDK

Handles heterogeneous feature sets across devices in federated learning.
Default: Embedding projection (zero config, just works).
"""

from .aligner import FeatureAligner, FeatureAlignmentStrategy, auto_align

__all__ = [
    'FeatureAligner',
    'FeatureAlignmentStrategy',
    'auto_align',
]
