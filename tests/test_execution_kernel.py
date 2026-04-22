"""Tests for the shared execution kernel."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from octomil._generated.routing_policy import RoutingPolicy as ContractRoutingPolicy
from octomil.config.local import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_TRANSCRIPTION,
    CapabilityDefault,
    CloudProfile,
    FallbackConfig,
    InlinePolicy,
    LoadedConfigSet,
    LocalOctomilConfig,
    ResolvedExecutionDefaults,
)
from octomil.execution.kernel import (
    ExecutionKernel,
    _inline_to_routing_policy,
    _resolve_routing_policy,
    _select_locality_for_capability,
)
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.core.types import RuntimeResponse, RuntimeUsage

# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


class TestResolveRoutingPolicy:
    def test_private_preset(self):
        d = ResolvedExecutionDefaults(policy_preset="private")
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.LOCAL_ONLY

    def test_local_first_preset(self):
        d = ResolvedExecutionDefaults(policy_preset="local_first")
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.LOCAL_FIRST

    def test_cloud_only_preset(self):
        d = ResolvedExecutionDefaults(policy_preset="cloud_only")
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.CLOUD_ONLY

    def test_cloud_first_preset(self):
        d = ResolvedExecutionDefaults(policy_preset="cloud_first")
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.AUTO
        assert p.prefer_local is False

    def test_performance_first_preset(self):
        d = ResolvedExecutionDefaults(policy_preset="performance_first")
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.AUTO
        assert p.prefer_local is True

    def test_none_defaults_to_local_first(self):
        d = ResolvedExecutionDefaults()
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.LOCAL_FIRST


# ---------------------------------------------------------------------------
# Kernel — create_response
# ---------------------------------------------------------------------------


def _make_kernel(model: str = "test-model", policy: str = "local_first") -> ExecutionKernel:
    config = LocalOctomilConfig(
        capabilities={
            CAPABILITY_CHAT: CapabilityDefault(model=model, policy=policy),
            CAPABILITY_EMBEDDING: CapabilityDefault(model="embed-model", policy=policy),
            CAPABILITY_TRANSCRIPTION: CapabilityDefault(model="whisper-test", policy=policy),
        }
    )
    config_set = LoadedConfigSet(project=config)
    return ExecutionKernel(config_set=config_set)


def _mock_runtime_response(text: str = "Hello!") -> RuntimeResponse:
    return RuntimeResponse(
        text=text,
        usage=RuntimeUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
    )


class TestKernelCreateResponse:
    @pytest.mark.asyncio
    async def test_calls_router_with_policy(self):
        kernel = _make_kernel()

        mock_runtime = AsyncMock()
        mock_runtime.run = AsyncMock(return_value=_mock_runtime_response())
        mock_runtime.stream = AsyncMock()
        mock_runtime.capabilities = MagicMock()

        with patch.object(kernel, "_build_router") as mock_build:
            mock_router = MagicMock()
            mock_router.resolve_locality = MagicMock(return_value=("on_device", False))
            mock_router.run = AsyncMock(return_value=_mock_runtime_response())
            mock_build.return_value = mock_router

            result = await kernel.create_response("Hello!")
            assert result.capability == CAPABILITY_CHAT
            assert result.model == "test-model"
            assert result.output_text == "Hello!"
            assert result.locality == "on_device"
            assert result.fallback_used is False

    @pytest.mark.asyncio
    async def test_offline_planner_cloud_fallback_defers_to_policy(self):
        kernel = _make_kernel()
        selection = MagicMock()
        selection.source = "offline"
        selection.locality = "cloud"
        selection.engine = None
        selection.candidates = []

        with (
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
            patch.object(kernel, "_build_router") as mock_build,
        ):
            mock_router = MagicMock()
            mock_router.run = AsyncMock(return_value=_mock_runtime_response())
            mock_build.return_value = mock_router

            result = await kernel.create_response("Hello!")

        assert result.locality == "on_device"
        assert result.fallback_used is False

    @pytest.mark.asyncio
    async def test_missing_model_raises(self):
        config = LocalOctomilConfig(capabilities={})
        kernel = ExecutionKernel(config_set=LoadedConfigSet(project=config))

        # Override builtin defaults so no model is found
        with patch("octomil.config.local._BUILTIN_DEFAULTS", {}):
            with pytest.raises(RuntimeError, match="No default model configured for chat"):
                await kernel.create_response("Hello!")

    @pytest.mark.asyncio
    async def test_explicit_model_overrides_config(self):
        kernel = _make_kernel()

        with patch.object(kernel, "_build_router") as mock_build:
            mock_router = MagicMock()
            mock_router.resolve_locality = MagicMock(return_value=("on_device", False))
            mock_router.run = AsyncMock(return_value=_mock_runtime_response())
            mock_build.return_value = mock_router

            result = await kernel.create_response("Hello!", model="override-model")
            assert result.model == "override-model"


# ---------------------------------------------------------------------------
# Kernel — create_embeddings
# ---------------------------------------------------------------------------


class TestKernelCreateEmbeddings:
    @pytest.mark.asyncio
    async def test_local_only_without_runtime_raises(self):
        kernel = _make_kernel(policy="private")

        with patch.object(kernel, "_can_local", return_value=False):
            with pytest.raises(RuntimeError, match="Local embedding execution is required by policy"):
                await kernel.create_embeddings(["test input"])

    @pytest.mark.asyncio
    async def test_no_cloud_no_local_raises(self, monkeypatch):
        monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
        kernel = _make_kernel(policy="local_first")

        with patch.object(kernel, "_can_local", return_value=False):
            with pytest.raises(RuntimeError, match="No local embedding runtime available"):
                await kernel.create_embeddings(["test input"])

    @pytest.mark.asyncio
    async def test_cloud_only_embeddings_use_cloud_even_when_local_available(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_EMBEDDING: CapabilityDefault(model="embed-model", policy="cloud_only")},
            cloud_profiles={"default": CloudProfile()},
        )
        kernel = ExecutionKernel(config_set=LoadedConfigSet(project=config))

        expected = MagicMock()
        expected.capability = CAPABILITY_EMBEDDING
        with patch.object(kernel, "_can_local", return_value=True):
            with patch.object(kernel, "_cloud_embed", new_callable=AsyncMock, return_value=expected) as cloud_embed:
                with patch.object(kernel, "_local_embed", new_callable=AsyncMock) as local_embed:
                    result = await kernel.create_embeddings(["test input"])

        assert result is expected
        cloud_embed.assert_awaited_once()
        local_embed.assert_not_called()


# ---------------------------------------------------------------------------
# Kernel — transcribe_audio
# ---------------------------------------------------------------------------


class TestKernelTranscribeAudio:
    @pytest.mark.asyncio
    async def test_local_only_without_runtime_raises(self):
        kernel = _make_kernel(policy="private")

        with patch.object(kernel, "_has_local_transcription_backend", return_value=False):
            with pytest.raises(RuntimeError, match="Local transcription execution is required by policy"):
                await kernel.transcribe_audio(b"fake_audio")

    @pytest.mark.asyncio
    async def test_no_local_runtime_raises(self, monkeypatch):
        monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
        kernel = _make_kernel(policy="local_first")

        with patch.object(kernel, "_has_local_transcription_backend", return_value=False):
            with pytest.raises(RuntimeError, match="No local transcription runtime available"):
                await kernel.transcribe_audio(b"fake_audio")

    @pytest.mark.asyncio
    async def test_local_transcribe_success(self):
        kernel = _make_kernel(policy="local_first")

        mock_backend = MagicMock()
        mock_backend.transcribe = MagicMock(return_value={"text": "hello world", "segments": []})

        with patch.object(kernel, "_has_local_transcription_backend", return_value=True):
            with patch.object(kernel, "_resolve_local_transcription_backend", return_value=mock_backend):
                result = await kernel.transcribe_audio(b"fake_audio", language="en")

        assert result.output_text == "hello world"
        assert result.capability == CAPABILITY_TRANSCRIPTION
        assert result.locality == "on_device"
        assert result.fallback_used is False


# ---------------------------------------------------------------------------
# Inline policy resolution
# ---------------------------------------------------------------------------


class TestInlinePolicyResolution:
    def test_inline_local_only(self):
        ip = InlinePolicy(routing_mode="local_only")
        rp = _inline_to_routing_policy(ip)
        assert rp.mode == ContractRoutingPolicy.LOCAL_ONLY

    def test_inline_cloud_only(self):
        ip = InlinePolicy(routing_mode="cloud_only")
        rp = _inline_to_routing_policy(ip)
        assert rp.mode == ContractRoutingPolicy.CLOUD_ONLY

    def test_inline_local_preference_with_cloud_fallback(self):
        ip = InlinePolicy(
            routing_mode="auto",
            routing_preference="local",
            fallback=FallbackConfig(allow_cloud_fallback=True),
        )
        rp = _inline_to_routing_policy(ip)
        assert rp.mode == ContractRoutingPolicy.LOCAL_FIRST
        assert rp.fallback == "cloud"

    def test_inline_local_preference_no_fallback(self):
        ip = InlinePolicy(
            routing_mode="auto",
            routing_preference="local",
            fallback=FallbackConfig(allow_cloud_fallback=False),
        )
        rp = _inline_to_routing_policy(ip)
        assert rp.fallback == "none"

    def test_inline_cloud_preference(self):
        ip = InlinePolicy(
            routing_mode="auto",
            routing_preference="cloud",
            fallback=FallbackConfig(allow_local_fallback=True),
        )
        rp = _inline_to_routing_policy(ip)
        assert rp.mode == ContractRoutingPolicy.AUTO
        assert rp.prefer_local is False

    def test_inline_performance_preference(self):
        ip = InlinePolicy(routing_mode="auto", routing_preference="performance")
        rp = _inline_to_routing_policy(ip)
        assert rp.mode == ContractRoutingPolicy.AUTO
        assert rp.prefer_local is True


# ---------------------------------------------------------------------------
# Locality selection for capabilities
# ---------------------------------------------------------------------------


class TestSelectLocalityForCapability:
    def test_local_only_with_local(self):
        rp = RoutingPolicy.local_only()
        locality, fb = _select_locality_for_capability(
            rp, local_available=True, cloud_available=False, capability="embedding"
        )
        assert locality == "on_device"
        assert fb is False

    def test_local_only_without_local_raises(self):
        rp = RoutingPolicy.local_only()
        with pytest.raises(RuntimeError, match="Local embedding execution is required"):
            _select_locality_for_capability(rp, local_available=False, cloud_available=True, capability="embedding")

    def test_cloud_only_with_cloud(self):
        rp = RoutingPolicy.cloud_only()
        locality, fb = _select_locality_for_capability(
            rp, local_available=True, cloud_available=True, capability="embedding"
        )
        assert locality == "cloud"
        assert fb is False

    def test_cloud_only_without_cloud_raises(self):
        rp = RoutingPolicy.cloud_only()
        with pytest.raises(RuntimeError, match="Cloud embedding execution is required"):
            _select_locality_for_capability(rp, local_available=True, cloud_available=False, capability="embedding")

    def test_local_first_prefers_local(self):
        rp = RoutingPolicy.local_first()
        locality, fb = _select_locality_for_capability(
            rp, local_available=True, cloud_available=True, capability="embedding"
        )
        assert locality == "on_device"
        assert fb is False

    def test_local_first_falls_back_to_cloud(self):
        rp = RoutingPolicy.local_first(fallback="cloud")
        locality, fb = _select_locality_for_capability(
            rp, local_available=False, cloud_available=True, capability="embedding"
        )
        assert locality == "cloud"
        assert fb is True

    def test_cloud_first_prefers_cloud(self):
        rp = RoutingPolicy(mode=ContractRoutingPolicy.AUTO, prefer_local=False, fallback="local")
        locality, fb = _select_locality_for_capability(
            rp, local_available=True, cloud_available=True, capability="embedding"
        )
        assert locality == "cloud"
        assert fb is False

    def test_cloud_first_falls_back_to_local(self):
        rp = RoutingPolicy(mode=ContractRoutingPolicy.AUTO, prefer_local=False, fallback="local")
        locality, fb = _select_locality_for_capability(
            rp, local_available=True, cloud_available=False, capability="embedding"
        )
        assert locality == "on_device"
        assert fb is True


# ---------------------------------------------------------------------------
# Kernel — local embeddings
# ---------------------------------------------------------------------------


class TestKernelLocalEmbeddings:
    @pytest.mark.asyncio
    async def test_local_embed_success(self):
        kernel = _make_kernel(policy="private")

        mock_runtime = MagicMock()
        mock_runtime.embed = MagicMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        with patch.object(kernel, "_can_local", return_value=True):
            with patch("octomil.runtime.core.registry.ModelRuntimeRegistry") as mock_reg_cls:
                mock_registry = MagicMock()
                mock_registry.resolve.return_value = mock_runtime
                mock_reg_cls.shared.return_value = mock_registry
                result = await kernel.create_embeddings(["hello", "world"])

        assert result.capability == CAPABILITY_EMBEDDING
        assert result.locality == "on_device"
        assert result.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert result.dimensions == 3

    @pytest.mark.asyncio
    async def test_local_embed_async_runtime(self):
        kernel = _make_kernel(policy="private")

        async def mock_embed(inputs):
            return [[1.0, 2.0]]

        mock_runtime = MagicMock()
        mock_runtime.embed = mock_embed

        with patch.object(kernel, "_can_local", return_value=True):
            with patch("octomil.runtime.core.registry.ModelRuntimeRegistry") as mock_reg_cls:
                mock_registry = MagicMock()
                mock_registry.resolve.return_value = mock_runtime
                mock_reg_cls.shared.return_value = mock_registry
                result = await kernel.create_embeddings(["hello"])

        assert result.embeddings == [[1.0, 2.0]]

    @pytest.mark.asyncio
    async def test_local_embed_no_embed_method_raises(self):
        kernel = _make_kernel(policy="private")

        mock_runtime = MagicMock(spec=[])  # no embed or create_embeddings

        with patch.object(kernel, "_can_local", return_value=True):
            with patch("octomil.runtime.core.registry.ModelRuntimeRegistry") as mock_reg_cls:
                mock_registry = MagicMock()
                mock_registry.resolve.return_value = mock_runtime
                mock_reg_cls.shared.return_value = mock_registry
                with pytest.raises(RuntimeError, match="does not expose an embedding interface"):
                    await kernel.create_embeddings(["hello"])


# ---------------------------------------------------------------------------
# Kernel — stream_response
# ---------------------------------------------------------------------------


class TestKernelStreamResponse:
    @pytest.mark.asyncio
    async def test_stream_collects_chunks(self):
        kernel = _make_kernel()

        async def mock_stream(request, policy):
            for text in ["Hello", " ", "world!"]:
                chunk = MagicMock()
                chunk.text = text
                yield chunk

        with patch.object(kernel, "_build_router") as mock_build:
            mock_router = MagicMock()
            mock_router.resolve_locality = MagicMock(return_value=("on_device", False))
            mock_router.stream = mock_stream
            mock_build.return_value = mock_router

            chunks = []
            async for chunk in kernel.stream_response("Hello!"):
                chunks.append(chunk)

        # Should have 3 delta chunks + 1 final done chunk
        assert len(chunks) == 4
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " "
        assert chunks[2].delta == "world!"
        assert chunks[3].done is True
        assert chunks[3].result is not None
        assert chunks[3].result.output_text == "Hello world!"


class TestKernelStreamChatMessages:
    @pytest.mark.asyncio
    async def test_stream_preserves_message_roles(self):
        kernel = _make_kernel()
        captured_request = None

        async def mock_stream(request, policy):
            nonlocal captured_request
            captured_request = request
            for text in ["Hi", "!"]:
                chunk = MagicMock()
                chunk.text = text
                yield chunk

        with patch.object(kernel, "_build_router") as mock_build:
            mock_router = MagicMock()
            mock_router.resolve_locality = MagicMock(return_value=("on_device", False))
            mock_router.stream = mock_stream
            mock_build.return_value = mock_router

            chunks = []
            async for chunk in kernel.stream_chat_messages(
                [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "again"},
                ]
            ):
                chunks.append(chunk)

        assert captured_request is not None
        assert [m.role.value for m in captured_request.messages] == ["system", "user", "assistant", "user"]
        assert chunks[-1].done is True
        assert chunks[-1].result is not None
        assert chunks[-1].result.output_text == "Hi!"


# ---------------------------------------------------------------------------
# Kernel — resolve_chat_routing
# ---------------------------------------------------------------------------


class TestResolveChatRouting:
    """Unit tests for ExecutionKernel.resolve_chat_routing — covers all presets
    and missing local/cloud availability cases."""

    def test_private_returns_local_only(self):
        kernel = _make_kernel(policy="private")
        decision = kernel.resolve_chat_routing(local_available=True)
        assert decision.model == "test-model"
        assert decision.primary_locality == "on_device"
        assert decision.fallback_locality is None
        assert decision.policy_preset == "private"

    def test_private_no_local_raises(self):
        kernel = _make_kernel(policy="private")
        with pytest.raises(RuntimeError, match="local"):
            kernel.resolve_chat_routing(local_available=False)

    def test_cloud_only_returns_cloud(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_only")},
            cloud_profiles={"default": CloudProfile()},
        )
        kernel = ExecutionKernel(config_set=LoadedConfigSet(project=config))
        decision = kernel.resolve_chat_routing(local_available=False, cloud_available=True)
        assert decision.primary_locality == "cloud"
        assert decision.fallback_locality is None
        assert decision.policy_preset == "cloud_only"

    def test_cloud_only_no_cloud_raises(self, monkeypatch):
        monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
        kernel = _make_kernel(policy="cloud_only")
        with pytest.raises(RuntimeError, match="cloud"):
            kernel.resolve_chat_routing(local_available=True, cloud_available=False)

    def test_local_first_with_both_available(self):
        kernel = _make_kernel(policy="local_first")
        decision = kernel.resolve_chat_routing(local_available=True, cloud_available=True)
        assert decision.primary_locality == "on_device"
        assert decision.fallback_locality == "cloud"

    def test_local_first_no_local_falls_back(self):
        kernel = _make_kernel(policy="local_first")
        decision = kernel.resolve_chat_routing(local_available=False, cloud_available=True)
        assert decision.primary_locality == "cloud"

    def test_cloud_first_with_both_available(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_first")},
            cloud_profiles={"default": CloudProfile()},
        )
        kernel = ExecutionKernel(config_set=LoadedConfigSet(project=config))
        decision = kernel.resolve_chat_routing(local_available=True, cloud_available=True)
        assert decision.primary_locality == "cloud"
        assert decision.fallback_locality == "on_device"

    def test_cloud_first_no_cloud_falls_back(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_first")},
            cloud_profiles={"default": CloudProfile()},
        )
        kernel = ExecutionKernel(config_set=LoadedConfigSet(project=config))
        decision = kernel.resolve_chat_routing(local_available=True, cloud_available=False)
        assert decision.primary_locality == "on_device"

    def test_performance_first_prefers_local(self):
        kernel = _make_kernel(policy="performance_first")
        decision = kernel.resolve_chat_routing(local_available=True, cloud_available=True)
        assert decision.primary_locality == "on_device"

    def test_explicit_model_overrides_config(self):
        kernel = _make_kernel(model="config-model")
        decision = kernel.resolve_chat_routing(model="explicit-model", local_available=True)
        assert decision.model == "explicit-model"

    def test_no_model_raises(self):
        config = LocalOctomilConfig(capabilities={})
        kernel = ExecutionKernel(config_set=LoadedConfigSet(project=config))
        with patch("octomil.config.local._BUILTIN_DEFAULTS", {}):
            with pytest.raises(RuntimeError, match="No default model"):
                kernel.resolve_chat_routing(local_available=True)

    def test_cloud_available_auto_resolved_from_config(self, monkeypatch):
        """When cloud_available is None, kernel checks config for cloud credentials."""
        monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
        kernel = _make_kernel(policy="local_first")
        # No cloud profile or key: cloud_available should auto-resolve to False
        decision = kernel.resolve_chat_routing(local_available=True, cloud_available=None)
        assert decision.primary_locality == "on_device"
        assert decision.fallback_locality is None  # no cloud available means no fallback
