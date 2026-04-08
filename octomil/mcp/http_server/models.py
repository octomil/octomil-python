"""Request/response Pydantic models for the HTTP agent server.

These models drive OpenAPI schema generation for the REST API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Phase 1 request models
# ---------------------------------------------------------------------------


class ResolveModelRequest(BaseModel):
    name: str = Field(..., description="Model specifier (e.g. 'gemma-3b', 'phi-mini:4bit')")
    engine: str = Field("", description="Force a specific engine (empty = auto-select)")


class ListModelsRequest(BaseModel):
    pass


class DetectEnginesRequest(BaseModel):
    model_name: str = Field("", description="Optional: filter engines by model compatibility")


class RunInferenceRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to send to the model")
    model: str = Field("", description="Model override (default: server's configured model)")
    max_tokens: int = Field(2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")


class DeployModelRequest(BaseModel):
    name: str = Field(..., description="Model name to deploy")
    version: str = Field("", description="Model version")
    devices: str = Field("", description="Comma-separated device IDs")
    group: str = Field("", description="Device group name")
    strategy: str = Field("canary", description="Deployment strategy: canary or rolling")
    rollout: int = Field(100, description="Rollout percentage 1-100")


# ---------------------------------------------------------------------------
# Phase 2 request models
# ---------------------------------------------------------------------------


class ConvertModelRequest(BaseModel):
    model_path: str = Field(..., description="Path to PyTorch model file (.pt or .pth)")
    target: str = Field("onnx", description="Comma-separated formats: onnx, coreml, tflite")
    input_shape: str = Field("1,3,224,224", description="Comma-separated input tensor shape")


class OptimizeModelRequest(BaseModel):
    name: str = Field(..., description="Model name (must be registered on platform)")
    target_devices: str = Field("", description="Comma-separated target device types")
    accuracy_threshold: float = Field(0.95, description="Min acceptable accuracy ratio 0-1")
    size_budget_mb: float = Field(0, description="Max model size in MB (0 = no limit)")


class BenchmarkModelRequest(BaseModel):
    model_name: str = Field(..., description="Model to benchmark")
    n_tokens: int = Field(32, description="Tokens to generate per benchmark run")
    engine: str = Field("", description="Specific engine to benchmark (empty = all)")


class RecommendModelRequest(BaseModel):
    priority: str = Field("balanced", description="Optimization: speed, quality, or balanced")


class ScanCodebaseRequest(BaseModel):
    path: str = Field(..., description="Directory path to scan")
    platform: str = Field("", description="Filter: ios, android, python (empty = all)")


class CompressPromptRequest(BaseModel):
    messages: str = Field(..., description='JSON array of messages [{"role":"user","content":"..."}]')
    strategy: str = Field("token_pruning", description="Strategy: token_pruning or sliding_window")
    target_ratio: float = Field(0.5, description="Target compression ratio 0-1")


class PlanDeploymentRequest(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field("", description="Model version (default: latest)")
    devices: str = Field("", description="Comma-separated device IDs")
    group: str = Field("", description="Device group name")


class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed (string or JSON array of strings)")
    model: str = Field("", description="Model ID for embeddings")


# ---------------------------------------------------------------------------
# Code tool request models
# ---------------------------------------------------------------------------


class GenerateCodeRequest(BaseModel):
    description: str = Field(..., description="What the code should do")
    language: str = Field("", description="Target programming language")
    context: str = Field("", description="Additional context")


class ReviewCodeRequest(BaseModel):
    code: str = Field(..., description="Code to review")
    language: str = Field("", description="Programming language")
    focus: str = Field("", description="Focus area (security, performance, style)")


class ExplainCodeRequest(BaseModel):
    code: str = Field(..., description="Code to explain")
    language: str = Field("", description="Programming language")
    detail_level: str = Field("medium", description="Detail level: brief, medium, thorough")


class WriteTestsRequest(BaseModel):
    code: str = Field(..., description="Code to test")
    language: str = Field("", description="Programming language")
    framework: str = Field("", description="Test framework (pytest, jest, etc.)")
    focus: str = Field("", description="Focus areas to test")


class GeneralTaskRequest(BaseModel):
    prompt: str = Field(..., description="The prompt or question")
    context: str = Field("", description="Additional context")
