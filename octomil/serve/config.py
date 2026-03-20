"""Server state and configuration dataclasses."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .types import InferenceBackend

if TYPE_CHECKING:
    from ..early_exit import EarlyExitConfig, EarlyExitMonitor
    from ..telemetry import TelemetryReporter


@dataclass
class CloudConfig:
    """Configuration for cloud provider routing."""

    base_url: str
    api_key: str
    model: str


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts models.

    Controls expert memory management and telemetry behaviour.
    Both llama.cpp and MLX handle MoE natively, so these settings
    primarily control detection, logging, and telemetry -- not
    the actual expert routing algorithm.
    """

    enabled: bool = True  # enable MoE-aware features (detection, telemetry)
    expert_memory_limit_mb: int = 0  # 0 = no limit; cap RAM for experts
    log_expert_routing: bool = False  # log per-request expert activation
    offload_inactive: bool = False  # hint to offload inactive experts to disk


@dataclass
class ServerState:
    """Shared mutable state for the serve app."""

    backend: Optional[InferenceBackend] = None
    whisper_backend: Any = None  # _WhisperBackend instance (speech-to-text)
    model_name: str = ""
    engine_name: str = ""
    start_time: float = field(default_factory=time.time)
    request_count: int = 0
    api_key: Optional[str] = None
    api_base: str = "https://api.octomil.com/api/v1"
    default_json_mode: bool = False
    cache_size_mb: int = 2048
    cache_enabled: bool = True
    engine_override: Optional[str] = None
    reporter: Optional["TelemetryReporter"] = None
    max_queue_depth: int = 32
    request_queue: Any = None  # RequestQueue instance
    moe_config: MoEConfig = field(default_factory=MoEConfig)
    is_moe_model: bool = False
    moe_metadata: Any = None  # MoEMetadata from catalog
    compressor: Any = None  # PromptCompressor instance
    early_exit_config: Optional["EarlyExitConfig"] = None
    early_exit_monitor: Optional["EarlyExitMonitor"] = None
    tool_use: bool = False  # pre-load coding agent tool schemas
    is_reasoning_model: bool = False  # model emits <think>...</think>
    verbose_runtime_logs: bool = False  # emit rich runtime events when -v is used
    verbose_emitter: Any = None  # VerboseEventEmitter instance
    cloud_config: Optional["CloudConfig"] = None  # cloud provider for --cloud mode


@dataclass
class MultiModelServerState:
    """Shared mutable state for multi-model serving with routing."""

    backends: dict[str, InferenceBackend] = field(default_factory=dict)
    model_names: list[str] = field(default_factory=list)
    router: Any = None  # QueryRouter instance
    start_time: float = field(default_factory=time.time)
    request_count: int = 0
    routed_counts: dict[str, int] = field(default_factory=dict)
    fallback_counts: int = 0
    api_key: Optional[str] = None
    api_base: str = "https://api.octomil.com/api/v1"
    default_json_mode: bool = False
    cache_size_mb: int = 2048
    cache_enabled: bool = True
    engine_override: Optional[str] = None
    reporter: Optional["TelemetryReporter"] = None
    route_strategy: str = "complexity"
    reasoning_models: set[str] = field(default_factory=set)
    compressor: Any = None  # PromptCompressor instance
