"""HTTP server exposing Octomil tools via REST + A2A agent card + OpenAPI.

This package is a refactored version of the original ``http_server.py`` monolith.
All public symbols are re-exported here for backward compatibility.
"""

from .app import create_http_app  # noqa: F401
from .config import HTTPServerConfig  # noqa: F401
from .models import (  # noqa: F401
    BenchmarkModelRequest,
    CompressPromptRequest,
    ConvertModelRequest,
    DeployModelRequest,
    DetectEnginesRequest,
    EmbedRequest,
    ExplainCodeRequest,
    GeneralTaskRequest,
    GenerateCodeRequest,
    ListModelsRequest,
    OptimizeModelRequest,
    PlanDeploymentRequest,
    RecommendModelRequest,
    ResolveModelRequest,
    ReviewCodeRequest,
    RunInferenceRequest,
    ScanCodebaseRequest,
    WriteTestsRequest,
)

__all__ = [
    "HTTPServerConfig",
    "create_http_app",
    "BenchmarkModelRequest",
    "CompressPromptRequest",
    "ConvertModelRequest",
    "DeployModelRequest",
    "DetectEnginesRequest",
    "EmbedRequest",
    "ExplainCodeRequest",
    "GeneralTaskRequest",
    "GenerateCodeRequest",
    "ListModelsRequest",
    "OptimizeModelRequest",
    "PlanDeploymentRequest",
    "RecommendModelRequest",
    "ResolveModelRequest",
    "ReviewCodeRequest",
    "RunInferenceRequest",
    "ScanCodebaseRequest",
    "WriteTestsRequest",
]
