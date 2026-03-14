"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum
from typing import NamedTuple


class ErrorCode(str, Enum):
    INVALID_API_KEY = "invalid_api_key"
    """401 — API key invalid or missing"""
    AUTHENTICATION_FAILED = "authentication_failed"
    """Auth failure (token expired, revoked, malformed)"""
    FORBIDDEN = "forbidden"
    """403 — insufficient permissions"""
    DEVICE_NOT_REGISTERED = "device_not_registered"
    """Device has not completed control.register()"""
    TOKEN_EXPIRED = "token_expired"
    """Access token has expired and must be refreshed or reissued"""
    DEVICE_REVOKED = "device_revoked"
    """Device registration has been revoked by an administrator"""
    NETWORK_UNAVAILABLE = "network_unavailable"
    """No connectivity"""
    REQUEST_TIMEOUT = "request_timeout"
    """Server did not respond in time"""
    SERVER_ERROR = "server_error"
    """5xx from server"""
    RATE_LIMITED = "rate_limited"
    """429 — too many requests"""
    INVALID_INPUT = "invalid_input"
    """Bad input data (malformed, wrong type, out of range)"""
    UNSUPPORTED_MODALITY = "unsupported_modality"
    """Input modality not supported by the target model"""
    CONTEXT_TOO_LARGE = "context_too_large"
    """Input exceeds model context window"""
    MODEL_NOT_FOUND = "model_not_found"
    """404 — requested model does not exist"""
    MODEL_DISABLED = "model_disabled"
    """Kill switch active for this model"""
    VERSION_NOT_FOUND = "version_not_found"
    """Requested version does not exist for this model"""
    DOWNLOAD_FAILED = "download_failed"
    """Model download error (network, server, storage)"""
    CHECKSUM_MISMATCH = "checksum_mismatch"
    """Integrity check failed after download"""
    INSUFFICIENT_STORAGE = "insufficient_storage"
    """Not enough disk space for model"""
    INSUFFICIENT_MEMORY = "insufficient_memory"
    """OOM during inference or model loading"""
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    """No compatible runtime for this model format"""
    ACCELERATOR_UNAVAILABLE = "accelerator_unavailable"
    """Required accelerator (GPU, NPU, ANE) not available"""
    MODEL_LOAD_FAILED = "model_load_failed"
    """Runtime initialization error"""
    INFERENCE_FAILED = "inference_failed"
    """Prediction error during inference"""
    STREAM_INTERRUPTED = "stream_interrupted"
    """Streaming response was interrupted before completion"""
    POLICY_DENIED = "policy_denied"
    """Routing policy explicitly denied the request"""
    CLOUD_FALLBACK_DISALLOWED = "cloud_fallback_disallowed"
    """Local inference failed and cloud fallback is disabled by policy"""
    MAX_TOOL_ROUNDS_EXCEEDED = "max_tool_rounds_exceeded"
    """Tool execution loop hit the iteration limit"""
    TRAINING_FAILED = "training_failed"
    """Training, aggregation, or update-processing operation failed"""
    TRAINING_NOT_SUPPORTED = "training_not_supported"
    """Requested training/update flow is not supported for this model, runtime, platform, or workspace"""
    WEIGHT_UPLOAD_FAILED = "weight_upload_failed"
    """Uploading model weights, deltas, or training updates failed before successful server acceptance"""
    CONTROL_SYNC_FAILED = "control_sync_failed"
    """Control plane sync returned an error"""
    ASSIGNMENT_NOT_FOUND = "assignment_not_found"
    """No model assignment exists for this device/experiment"""
    CANCELLED = "cancelled"
    """User or caller cancelled the operation"""
    APP_BACKGROUNDED = "app_backgrounded"
    """App moved to background, operation stopped"""
    UNKNOWN = "unknown"
    """Catch-all for unrecognized errors. SDKs MUST map unrecognized codes here."""


class ErrorCategory(str, Enum):
    AUTH = "auth"
    """Auth / Access"""
    NETWORK = "network"
    """Network / Transport"""
    INPUT = "input"
    """Input / Validation"""
    CATALOG = "catalog"
    """Catalog / Model Resolution"""
    DOWNLOAD = "download"
    """Download / Artifact Integrity"""
    DEVICE = "device"
    """Device / Environment"""
    RUNTIME = "runtime"
    """Runtime / Inference"""
    POLICY = "policy"
    """Policy / Routing"""
    TRAINING = "training"
    """Training / Federated Learning"""
    CONTROL = "control"
    """Control Plane / Rollout"""
    LIFECYCLE = "lifecycle"
    """Cancellation / Lifecycle"""
    UNKNOWN = "unknown"
    """Unknown"""


class RetryClass(str, Enum):
    NEVER = "never"
    IMMEDIATE_SAFE = "immediate_safe"
    BACKOFF_SAFE = "backoff_safe"
    CONDITIONAL = "conditional"


class SuggestedAction(str, Enum):
    FIX_CREDENTIALS = "fix_credentials"
    REAUTHENTICATE = "reauthenticate"
    CHECK_PERMISSIONS = "check_permissions"
    REGISTER_DEVICE = "register_device"
    RETRY_OR_FALLBACK = "retry_or_fallback"
    RETRY = "retry"
    RETRY_AFTER = "retry_after"
    FIX_REQUEST = "fix_request"
    REDUCE_INPUT_OR_FALLBACK = "reduce_input_or_fallback"
    CHECK_MODEL_ID = "check_model_id"
    USE_ALTERNATE_MODEL = "use_alternate_model"
    CHECK_VERSION = "check_version"
    REDOWNLOAD = "redownload"
    FREE_STORAGE_OR_FALLBACK = "free_storage_or_fallback"
    TRY_SMALLER_MODEL = "try_smaller_model"
    TRY_ALTERNATE_RUNTIME = "try_alternate_runtime"
    TRY_CPU_OR_FALLBACK = "try_cpu_or_fallback"
    CHECK_POLICY = "check_policy"
    CHANGE_POLICY_OR_FIX_LOCAL = "change_policy_or_fix_local"
    INCREASE_LIMIT_OR_SIMPLIFY = "increase_limit_or_simplify"
    CHECK_ASSIGNMENT = "check_assignment"
    NONE = "none"
    RESUME_ON_FOREGROUND = "resume_on_foreground"
    REPORT_BUG = "report_bug"


class ErrorClassification(NamedTuple):
    category: ErrorCategory
    retry_class: RetryClass
    fallback_eligible: bool
    suggested_action: SuggestedAction


ERROR_CLASSIFICATION: dict[ErrorCode, ErrorClassification] = {
    ErrorCode.INVALID_API_KEY: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_CREDENTIALS
    ),
    ErrorCode.AUTHENTICATION_FAILED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.REAUTHENTICATE
    ),
    ErrorCode.FORBIDDEN: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.CHECK_PERMISSIONS
    ),
    ErrorCode.DEVICE_NOT_REGISTERED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.REGISTER_DEVICE
    ),
    ErrorCode.TOKEN_EXPIRED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.REAUTHENTICATE
    ),
    ErrorCode.DEVICE_REVOKED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.REGISTER_DEVICE
    ),
    ErrorCode.NETWORK_UNAVAILABLE: ErrorClassification(
        ErrorCategory.NETWORK,
        RetryClass.BACKOFF_SAFE,
        True,
        SuggestedAction.RETRY_OR_FALLBACK,
    ),
    ErrorCode.REQUEST_TIMEOUT: ErrorClassification(
        ErrorCategory.NETWORK,
        RetryClass.CONDITIONAL,
        True,
        SuggestedAction.RETRY_OR_FALLBACK,
    ),
    ErrorCode.SERVER_ERROR: ErrorClassification(
        ErrorCategory.NETWORK, RetryClass.BACKOFF_SAFE, True, SuggestedAction.RETRY
    ),
    ErrorCode.RATE_LIMITED: ErrorClassification(
        ErrorCategory.NETWORK,
        RetryClass.CONDITIONAL,
        False,
        SuggestedAction.RETRY_AFTER,
    ),
    ErrorCode.INVALID_INPUT: ErrorClassification(
        ErrorCategory.INPUT, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.UNSUPPORTED_MODALITY: ErrorClassification(
        ErrorCategory.INPUT, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.CONTEXT_TOO_LARGE: ErrorClassification(
        ErrorCategory.INPUT,
        RetryClass.NEVER,
        True,
        SuggestedAction.REDUCE_INPUT_OR_FALLBACK,
    ),
    ErrorCode.MODEL_NOT_FOUND: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, False, SuggestedAction.CHECK_MODEL_ID
    ),
    ErrorCode.MODEL_DISABLED: ErrorClassification(
        ErrorCategory.CATALOG,
        RetryClass.NEVER,
        True,
        SuggestedAction.USE_ALTERNATE_MODEL,
    ),
    ErrorCode.VERSION_NOT_FOUND: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, False, SuggestedAction.CHECK_VERSION
    ),
    ErrorCode.DOWNLOAD_FAILED: ErrorClassification(
        ErrorCategory.DOWNLOAD,
        RetryClass.BACKOFF_SAFE,
        True,
        SuggestedAction.RETRY_OR_FALLBACK,
    ),
    ErrorCode.CHECKSUM_MISMATCH: ErrorClassification(
        ErrorCategory.DOWNLOAD,
        RetryClass.CONDITIONAL,
        False,
        SuggestedAction.REDOWNLOAD,
    ),
    ErrorCode.INSUFFICIENT_STORAGE: ErrorClassification(
        ErrorCategory.DEVICE,
        RetryClass.NEVER,
        True,
        SuggestedAction.FREE_STORAGE_OR_FALLBACK,
    ),
    ErrorCode.INSUFFICIENT_MEMORY: ErrorClassification(
        ErrorCategory.DEVICE, RetryClass.NEVER, True, SuggestedAction.TRY_SMALLER_MODEL
    ),
    ErrorCode.RUNTIME_UNAVAILABLE: ErrorClassification(
        ErrorCategory.DEVICE,
        RetryClass.NEVER,
        True,
        SuggestedAction.TRY_ALTERNATE_RUNTIME,
    ),
    ErrorCode.ACCELERATOR_UNAVAILABLE: ErrorClassification(
        ErrorCategory.DEVICE,
        RetryClass.NEVER,
        True,
        SuggestedAction.TRY_CPU_OR_FALLBACK,
    ),
    ErrorCode.MODEL_LOAD_FAILED: ErrorClassification(
        ErrorCategory.RUNTIME,
        RetryClass.CONDITIONAL,
        True,
        SuggestedAction.RETRY_OR_FALLBACK,
    ),
    ErrorCode.INFERENCE_FAILED: ErrorClassification(
        ErrorCategory.RUNTIME,
        RetryClass.CONDITIONAL,
        True,
        SuggestedAction.RETRY_OR_FALLBACK,
    ),
    ErrorCode.STREAM_INTERRUPTED: ErrorClassification(
        ErrorCategory.RUNTIME, RetryClass.IMMEDIATE_SAFE, True, SuggestedAction.RETRY
    ),
    ErrorCode.POLICY_DENIED: ErrorClassification(
        ErrorCategory.POLICY, RetryClass.NEVER, False, SuggestedAction.CHECK_POLICY
    ),
    ErrorCode.CLOUD_FALLBACK_DISALLOWED: ErrorClassification(
        ErrorCategory.POLICY,
        RetryClass.NEVER,
        False,
        SuggestedAction.CHANGE_POLICY_OR_FIX_LOCAL,
    ),
    ErrorCode.MAX_TOOL_ROUNDS_EXCEEDED: ErrorClassification(
        ErrorCategory.POLICY,
        RetryClass.NEVER,
        False,
        SuggestedAction.INCREASE_LIMIT_OR_SIMPLIFY,
    ),
    ErrorCode.TRAINING_FAILED: ErrorClassification(
        ErrorCategory.TRAINING, RetryClass.CONDITIONAL, False, SuggestedAction.RETRY
    ),
    ErrorCode.TRAINING_NOT_SUPPORTED: ErrorClassification(
        ErrorCategory.TRAINING, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.WEIGHT_UPLOAD_FAILED: ErrorClassification(
        ErrorCategory.TRAINING, RetryClass.BACKOFF_SAFE, False, SuggestedAction.RETRY
    ),
    ErrorCode.CONTROL_SYNC_FAILED: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.BACKOFF_SAFE, False, SuggestedAction.RETRY
    ),
    ErrorCode.ASSIGNMENT_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.CHECK_ASSIGNMENT
    ),
    ErrorCode.CANCELLED: ErrorClassification(ErrorCategory.LIFECYCLE, RetryClass.NEVER, False, SuggestedAction.NONE),
    ErrorCode.APP_BACKGROUNDED: ErrorClassification(
        ErrorCategory.LIFECYCLE,
        RetryClass.CONDITIONAL,
        False,
        SuggestedAction.RESUME_ON_FOREGROUND,
    ),
    ErrorCode.UNKNOWN: ErrorClassification(ErrorCategory.UNKNOWN, RetryClass.NEVER, False, SuggestedAction.REPORT_BUG),
}
