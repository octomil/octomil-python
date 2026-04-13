"""Catalog module — manifest client, offline snapshot, alias resolution.

This module owns model alias resolution. New model alias logic should be
added here rather than in runtime engines directly.

Source of truth: server catalog API (GET /api/v2/catalog/manifest)
Offline fallback: generated snapshot from the same manifest schema
"""
