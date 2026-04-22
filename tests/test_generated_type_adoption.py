"""Contract-generated type adoption conformance test.

Verifies that the generated enum values from octomil-contracts match
the expected canonical values used across all SDKs.
"""

from octomil._generated.routing_policy import RoutingPolicy
from octomil._generated.runtime_executor import RuntimeExecutor


class TestRoutingPolicyGenerated:
    """Verify generated RoutingPolicy enum matches canonical values."""

    def test_has_core_policies(self) -> None:
        assert RoutingPolicy.PRIVATE.value == "private"
        assert RoutingPolicy.LOCAL_ONLY.value == "local_only"
        assert RoutingPolicy.LOCAL_FIRST.value == "local_first"
        assert RoutingPolicy.CLOUD_FIRST.value == "cloud_first"
        assert RoutingPolicy.CLOUD_ONLY.value == "cloud_only"
        assert RoutingPolicy.PERFORMANCE_FIRST.value == "performance_first"
        assert RoutingPolicy.AUTO.value == "auto"

    def test_is_str_enum(self) -> None:
        """Generated RoutingPolicy is a str enum so it serializes as string."""
        assert isinstance(RoutingPolicy.PRIVATE, str)
        assert RoutingPolicy.PRIVATE == "private"

    def test_used_in_policy_module(self) -> None:
        """Verify the SDK's runtime policy module uses generated type."""
        from octomil.runtime.core.policy import RoutingPolicy as SDKPolicy

        # The SDK wraps the generated enum in its own dataclass
        auto = SDKPolicy.auto()
        assert auto.mode == RoutingPolicy.AUTO


class TestRuntimeExecutorGenerated:
    """Verify generated RuntimeExecutor enum matches canonical engine names."""

    def test_blessed_engines(self) -> None:
        """Blessed engines (contracts tier) are present."""
        assert RuntimeExecutor.COREML.value == "coreml"
        assert RuntimeExecutor.LITERT.value == "litert"
        assert RuntimeExecutor.LLAMACPP.value == "llamacpp"

    def test_supported_engines(self) -> None:
        """Supported engines (contracts tier) are present."""
        assert RuntimeExecutor.MLX.value == "mlx"
        assert RuntimeExecutor.ONNXRUNTIME.value == "onnxruntime"
        assert RuntimeExecutor.CLOUD.value == "cloud"
        assert RuntimeExecutor.OLLAMA.value == "ollama"
        assert RuntimeExecutor.WHISPER.value == "whisper"

    def test_experimental_engines(self) -> None:
        """Experimental engines are present."""
        assert RuntimeExecutor.MLC.value == "mlc"
        assert RuntimeExecutor.CACTUS.value == "cactus"
        assert RuntimeExecutor.SAMSUNG_ONE.value == "samsung_one"
        assert RuntimeExecutor.MNN.value == "mnn"

    def test_test_engine(self) -> None:
        """Test-only engine is present."""
        assert RuntimeExecutor.ECHO.value == "echo"

    def test_is_str_enum(self) -> None:
        assert isinstance(RuntimeExecutor.LLAMACPP, str)
        assert RuntimeExecutor.LLAMACPP == "llamacpp"


class TestPlannerSourceCanonicalValues:
    """Verify hand-maintained CANONICAL_PLANNER_SOURCES matches expected contract values.

    Note: PlannerSource is not yet code-generated for Python (only for TypeScript).
    This test validates the hand-maintained frozenset against expected values.
    """

    def test_canonical_values(self) -> None:
        from octomil.runtime.planner.schemas import CANONICAL_PLANNER_SOURCES

        assert CANONICAL_PLANNER_SOURCES == frozenset({"server", "cache", "offline"})

    def test_normalize_maps_aliases(self) -> None:
        from octomil.runtime.planner.schemas import normalize_planner_source

        assert normalize_planner_source("server") == "server"
        assert normalize_planner_source("cache") == "cache"
        assert normalize_planner_source("offline") == "offline"
        assert normalize_planner_source("local_default") == "offline"
        assert normalize_planner_source("server_plan") == "server"
        assert normalize_planner_source("cached") == "cache"
        assert normalize_planner_source("unknown_value") == "offline"


class TestModelRefKindCanonicalValues:
    """Verify hand-maintained CANONICAL_MODEL_REF_KINDS matches expected contract values.

    Note: ModelRefKind is not yet code-generated for Python (only for TypeScript).
    """

    def test_canonical_values(self) -> None:
        from octomil.runtime.routing.model_ref import CANONICAL_MODEL_REF_KINDS

        expected = frozenset({"model", "app", "capability", "deployment", "experiment", "alias", "default", "unknown"})
        assert CANONICAL_MODEL_REF_KINDS == expected
