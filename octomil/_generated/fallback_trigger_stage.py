"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class FallbackTriggerStage(str, Enum):
    POLICY = "policy"
    """Candidate was rejected by routing policy"""
    PREPARE = "prepare"
    """Candidate failed during runtime preparation"""
    DOWNLOAD = "download"
    """Candidate failed while obtaining an artifact"""
    VERIFY = "verify"
    """Candidate failed artifact or runtime verification"""
    LOAD = "load"
    """Candidate failed model load"""
    BENCHMARK = "benchmark"
    """Candidate failed benchmark freshness or performance checks"""
    GATE = "gate"
    """Candidate failed a planner or SDK gate"""
    INFERENCE = "inference"
    """Candidate failed before producing inference output"""
    TIMEOUT = "timeout"
    """Candidate timed out before output was committed"""
    NOT_APPLICABLE = "not_applicable"
    """No fallback occurred"""
