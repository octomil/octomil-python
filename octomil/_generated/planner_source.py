"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class PlannerSource(str, Enum):
    SERVER = "server"
    """Plan fetched from the server planner API"""
    CACHE = "cache"
    """Plan loaded from local cache"""
    OFFLINE = "offline"
    """Plan synthesized locally without server contact"""
