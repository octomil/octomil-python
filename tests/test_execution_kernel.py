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
    LoadedConfigSet,
    LocalOctomilConfig,
)
from octomil.execution.kernel import (
    ExecutionKernel,
    _resolve_routing_policy,
)
from octomil.runtime.core.types import RuntimeResponse, RuntimeUsage

# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


class TestResolveRoutingPolicy:
    def test_private_preset(self):
        from octomil.config.local import ResolvedExecutionDefaults

        d = ResolvedExecutionDefaults(policy_preset="private")
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.LOCAL_ONLY

    def test_local_first_preset(self):
        from octomil.config.local import ResolvedExecutionDefaults

        d = ResolvedExecutionDefaults(policy_preset="local_first")
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.LOCAL_FIRST

    def test_cloud_only_preset(self):
        from octomil.config.local import ResolvedExecutionDefaults

        d = ResolvedExecutionDefaults(policy_preset="cloud_only")
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.CLOUD_ONLY

    def test_cloud_first_preset(self):
        from octomil.config.local import ResolvedExecutionDefaults

        d = ResolvedExecutionDefaults(policy_preset="cloud_first")
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.AUTO
        assert p.prefer_local is False

    def test_performance_first_preset(self):
        from octomil.config.local import ResolvedExecutionDefaults

        d = ResolvedExecutionDefaults(policy_preset="performance_first")
        p = _resolve_routing_policy(d)
        assert p.mode == ContractRoutingPolicy.AUTO
        assert p.prefer_local is True

    def test_none_defaults_to_local_first(self):
        from octomil.config.local import ResolvedExecutionDefaults

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
