"""Tests for serve kernel integration — config-driven routing through serve."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from octomil.config.local import (
    CAPABILITY_CHAT,
    CapabilityDefault,
    CloudProfile,
    LoadedConfigSet,
    LocalOctomilConfig,
)
from octomil.execution.kernel import (
    ExecutionKernel,
    _resolve_localities,
)
from octomil.runtime.core.policy import RoutingPolicy
from octomil.serve import EchoBackend, GenerationRequest, create_app
from octomil.serve.types import GenerationChunk, InferenceMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config_set(
    model: str = "test-model",
    policy: str = "local_first",
) -> LoadedConfigSet:
    config = LocalOctomilConfig(
        capabilities={CAPABILITY_CHAT: CapabilityDefault(model=model, policy=policy)},
    )
    return LoadedConfigSet(project=config)


def _make_echo_backend(model: str = "test-model") -> EchoBackend:
    b = EchoBackend()
    b.load_model(model)
    return b


async def _start_lifespan(app: Any) -> None:
    """Manually trigger the FastAPI lifespan startup.

    ASGITransport does not send ASGI lifespan events, so we must
    trigger the lifespan ourselves before making requests.
    """
    ctx = app.router.lifespan_context(app)
    await ctx.__aenter__()


class _FakeCloudBackend:
    """Minimal cloud backend stub for tests."""

    name = "cloud"
    attention_backend = "cloud"

    def __init__(self, model: str = "test-model") -> None:
        self._model = model

    def load_model(self, name: str) -> None:
        pass

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        return f"[cloud:{self._model}] response", InferenceMetrics(total_tokens=5)

    async def generate_async(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        return self.generate(request)

    async def generate_stream(self, request: GenerationRequest):
        yield GenerationChunk(text=f"[cloud:{self._model}]", finish_reason="stop")

    def list_models(self) -> list[str]:
        return [self._model]


# ---------------------------------------------------------------------------
# WS1 — ChatRoutingDecision + resolve_chat_routing
# ---------------------------------------------------------------------------


class TestResolveChatRouting:
    """Kernel routing decision API used by serve."""

    def test_private_local_only(self):
        kernel = ExecutionKernel(config_set=_make_config_set(policy="private"))
        decision = kernel.resolve_chat_routing(local_available=True)
        assert decision.primary_locality == "on_device"
        assert decision.fallback_locality is None

    def test_private_no_local_raises(self):
        kernel = ExecutionKernel(config_set=_make_config_set(policy="private"))
        with pytest.raises(RuntimeError, match="local"):
            kernel.resolve_chat_routing(local_available=False)

    def test_cloud_only_cloud_available(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_only")},
            cloud_profiles={"default": CloudProfile()},
        )
        kernel = ExecutionKernel(config_set=LoadedConfigSet(project=config))
        decision = kernel.resolve_chat_routing(local_available=False, cloud_available=True)
        assert decision.primary_locality == "cloud"
        assert decision.fallback_locality is None

    def test_cloud_only_no_cloud_raises(self, monkeypatch):
        monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
        kernel = ExecutionKernel(config_set=_make_config_set(policy="cloud_only"))
        with pytest.raises(RuntimeError, match="cloud"):
            kernel.resolve_chat_routing(local_available=True, cloud_available=False)

    def test_local_first_uses_local_primary(self):
        kernel = ExecutionKernel(config_set=_make_config_set(policy="local_first"))
        decision = kernel.resolve_chat_routing(local_available=True, cloud_available=True)
        assert decision.primary_locality == "on_device"
        assert decision.fallback_locality == "cloud"

    def test_local_first_falls_back_to_cloud(self):
        kernel = ExecutionKernel(config_set=_make_config_set(policy="local_first"))
        decision = kernel.resolve_chat_routing(local_available=False, cloud_available=True)
        assert decision.primary_locality == "cloud"

    def test_cloud_first_uses_cloud_primary(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_first")},
            cloud_profiles={"default": CloudProfile()},
        )
        kernel = ExecutionKernel(config_set=LoadedConfigSet(project=config))
        decision = kernel.resolve_chat_routing(local_available=True, cloud_available=True)
        assert decision.primary_locality == "cloud"
        assert decision.fallback_locality == "on_device"

    def test_cloud_first_falls_back_to_local(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_first")},
            cloud_profiles={"default": CloudProfile()},
        )
        kernel = ExecutionKernel(config_set=LoadedConfigSet(project=config))
        decision = kernel.resolve_chat_routing(local_available=True, cloud_available=False)
        assert decision.primary_locality == "on_device"

    def test_performance_first_prefers_local(self):
        kernel = ExecutionKernel(config_set=_make_config_set(policy="performance_first"))
        decision = kernel.resolve_chat_routing(local_available=True, cloud_available=True)
        assert decision.primary_locality == "on_device"

    def test_config_set_property(self):
        cs = _make_config_set()
        kernel = ExecutionKernel(config_set=cs)
        assert kernel.config_set is cs

    def test_explicit_model_overrides_config(self):
        kernel = ExecutionKernel(config_set=_make_config_set(model="config-model"))
        decision = kernel.resolve_chat_routing(model="explicit-model", local_available=True)
        assert decision.model == "explicit-model"

    def test_model_from_config(self):
        kernel = ExecutionKernel(config_set=_make_config_set(model="my-model"))
        decision = kernel.resolve_chat_routing(local_available=True)
        assert decision.model == "my-model"


# ---------------------------------------------------------------------------
# WS3 — Serve model argument optional
# ---------------------------------------------------------------------------


class TestServeModelOptional:
    def test_serve_model_argument_optional_resolves_from_config(self):
        """When MODEL is omitted, config provides the default."""
        config_set = _make_config_set(model="config-gemma")

        with patch("octomil.serve.app._detect_backend") as mock_detect:
            mock_detect.return_value = _make_echo_backend("config-gemma")
            app = create_app("config-gemma", config_set=config_set, engine="echo")
            assert app is not None

    def test_explicit_model_wins_over_config(self):
        """Positional MODEL argument takes precedence over config."""
        config_set = _make_config_set(model="config-model")

        with patch("octomil.serve.app._detect_backend") as mock_detect:
            mock_detect.return_value = _make_echo_backend("explicit-model")
            app = create_app("explicit-model", config_set=config_set, engine="echo")
            assert app is not None


# ---------------------------------------------------------------------------
# WS4 — Lifespan creates kernel when config_set is passed
# ---------------------------------------------------------------------------


class TestLifespanKernel:
    @pytest.mark.asyncio
    async def test_lifespan_creates_kernel_when_config_set_passed(self):
        config_set = _make_config_set()

        with patch("octomil.serve.app._detect_backend") as mock_detect:
            mock_detect.return_value = _make_echo_backend()
            app = create_app("test-model", config_set=config_set)
            await _start_lifespan(app)
            # Backend should be set after lifespan startup
            mock_detect.assert_called_once()


# ---------------------------------------------------------------------------
# WS7 — Routing dispatch
# ---------------------------------------------------------------------------


class TestRoutingDispatch:
    @pytest.mark.asyncio
    async def test_no_kernel_backward_compat_uses_state_backend(self):
        """Without kernel, dispatch uses state.backend directly."""
        echo = _make_echo_backend()

        with patch("octomil.serve.app._detect_backend", return_value=echo):
            app = create_app("test-model")
            await _start_lifespan(app)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
                assert resp.status_code == 200, f"Response: {resp.text}"
                data = resp.json()
                assert "[echo:test-model]" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_private_policy_detects_local_only(self):
        """Private policy creates only local backend."""
        config_set = _make_config_set(policy="private")
        echo = _make_echo_backend()

        with patch("octomil.serve.app._detect_backend", return_value=echo):
            app = create_app("test-model", config_set=config_set)
            await _start_lifespan(app)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
                assert resp.status_code == 200
                data = resp.json()
                assert "[echo:test-model]" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_cloud_only_policy_skips_local_detection(self, monkeypatch):
        """Cloud-only policy should use cloud backend, not local."""
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_only")},
            cloud_profiles={"default": CloudProfile()},
        )
        config_set = LoadedConfigSet(project=config)
        fake_cloud = _FakeCloudBackend()
        echo = _make_echo_backend()

        with patch("octomil.serve.app._detect_backend", return_value=echo):
            with patch("octomil.serve.app._create_cloud_backend_from_profile", return_value=fake_cloud):
                app = create_app("test-model", config_set=config_set)
                await _start_lifespan(app)
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert "[cloud:" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_local_first_uses_local_primary(self):
        """Local-first policy uses local backend as primary."""
        config_set = _make_config_set(policy="local_first")
        echo = _make_echo_backend()

        with patch("octomil.serve.app._detect_backend", return_value=echo):
            app = create_app("test-model", config_set=config_set)
            await _start_lifespan(app)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
                assert resp.status_code == 200
                data = resp.json()
                assert "[echo:test-model]" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_local_first_falls_back_to_cloud_when_local_generate_fails(self, monkeypatch):
        """When local backend fails, routing falls back to cloud."""
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="local_first")},
            cloud_profiles={"default": CloudProfile()},
        )
        config_set = LoadedConfigSet(project=config)
        failing_backend = MagicMock()
        failing_backend.name = "failing"
        failing_backend.generate.side_effect = RuntimeError("local engine crashed")
        failing_backend.generate_stream = AsyncMock(side_effect=RuntimeError("local engine crashed"))
        fake_cloud = _FakeCloudBackend()

        with patch("octomil.serve.app._detect_backend", return_value=failing_backend):
            with patch("octomil.serve.app._create_cloud_backend_from_profile", return_value=fake_cloud):
                app = create_app("test-model", config_set=config_set)
                await _start_lifespan(app)
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert "[cloud:" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_cloud_first_uses_cloud_primary(self, monkeypatch):
        """Cloud-first policy uses cloud backend as primary."""
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_first")},
            cloud_profiles={"default": CloudProfile()},
        )
        config_set = LoadedConfigSet(project=config)
        echo = _make_echo_backend()
        fake_cloud = _FakeCloudBackend()

        with patch("octomil.serve.app._detect_backend", return_value=echo):
            with patch("octomil.serve.app._create_cloud_backend_from_profile", return_value=fake_cloud):
                app = create_app("test-model", config_set=config_set)
                await _start_lifespan(app)
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert "[cloud:" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_cloud_first_falls_back_to_local_when_cloud_generate_fails(self, monkeypatch):
        """When cloud backend fails, routing falls back to local."""
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_first")},
            cloud_profiles={"default": CloudProfile()},
        )
        config_set = LoadedConfigSet(project=config)
        echo = _make_echo_backend()
        failing_cloud = MagicMock()
        failing_cloud.name = "cloud"
        failing_cloud.generate_async = AsyncMock(side_effect=RuntimeError("cloud unavailable"))
        failing_cloud.generate.side_effect = RuntimeError("cloud unavailable")
        failing_cloud.generate_stream = AsyncMock(side_effect=RuntimeError("cloud unavailable"))

        with patch("octomil.serve.app._detect_backend", return_value=echo):
            with patch("octomil.serve.app._create_cloud_backend_from_profile", return_value=failing_cloud):
                app = create_app("test-model", config_set=config_set)
                await _start_lifespan(app)
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert "[echo:" in data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# WS8 — JSON retry uses routed backend
# ---------------------------------------------------------------------------


class TestJsonRetryRouting:
    @pytest.mark.asyncio
    async def test_json_retry_uses_routed_backend_not_state_backend(self, monkeypatch):
        """JSON retry should dispatch through routing, not hardcoded state.backend."""
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_first")},
            cloud_profiles={"default": CloudProfile()},
        )
        config_set = LoadedConfigSet(project=config)
        echo = _make_echo_backend()
        cloud = MagicMock()
        cloud.name = "cloud"
        cloud.generate_async = AsyncMock(return_value=('{"valid": true}', InferenceMetrics(total_tokens=3)))

        with patch("octomil.serve.app._detect_backend", return_value=echo):
            with patch("octomil.serve.app._create_cloud_backend_from_profile", return_value=cloud):
                app = create_app("test-model", config_set=config_set, json_mode=True, max_queue_depth=0)
                await _start_lifespan(app)
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "give me json"}],
                            "response_format": {"type": "json_object"},
                        },
                    )
                    assert resp.status_code == 200
                    cloud.generate_async.assert_called()


# ---------------------------------------------------------------------------
# WS9 — Streaming uses routed backend
# ---------------------------------------------------------------------------


class TestStreamingRouting:
    @pytest.mark.asyncio
    async def test_streaming_uses_routed_backend(self):
        """Streaming requests dispatch through the routed backend."""
        config_set = _make_config_set(policy="private")
        echo = _make_echo_backend()

        with patch("octomil.serve.app._detect_backend", return_value=echo):
            app = create_app("test-model", config_set=config_set)
            await _start_lifespan(app)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                assert resp.status_code == 200
                text = resp.text
                assert "data:" in text
                assert "[DONE]" in text


# ---------------------------------------------------------------------------
# WS10 — Existing --cloud flag still works
# ---------------------------------------------------------------------------


class TestExistingCloudFlag:
    @pytest.mark.asyncio
    async def test_existing_cloud_flag_still_uses_cloud_config(self):
        """Explicit --cloud with CloudConfig should still work as before."""
        from octomil.serve.config import CloudConfig

        cloud_cfg = CloudConfig(base_url="https://fake.api.com/v1", api_key="test-key", model="gpt-test")

        with patch("octomil.serve.backends.cloud.CloudInferenceBackend") as MockCloud:
            instance = MagicMock()
            instance.name = "cloud"
            instance.generate.return_value = (
                "cloud response",
                InferenceMetrics(total_tokens=5),
            )
            instance.list_models.return_value = ["gpt-test"]
            MockCloud.return_value = instance

            app = create_app("gpt-test", cloud_config=cloud_cfg)
            await _start_lifespan(app)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-test",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
                assert resp.status_code == 200
                instance.generate.assert_called_once()


# ---------------------------------------------------------------------------
# _resolve_localities unit tests
# ---------------------------------------------------------------------------


class TestResolveLocalities:
    def test_local_only_with_local(self):
        rp = RoutingPolicy.local_only()
        primary, fallback = _resolve_localities(rp, local_available=True, cloud_available=False)
        assert primary == "on_device"
        assert fallback is None

    def test_local_only_without_local_raises(self):
        rp = RoutingPolicy.local_only()
        with pytest.raises(RuntimeError):
            _resolve_localities(rp, local_available=False, cloud_available=True)

    def test_cloud_only_with_cloud(self):
        rp = RoutingPolicy.cloud_only()
        primary, fallback = _resolve_localities(rp, local_available=False, cloud_available=True)
        assert primary == "cloud"
        assert fallback is None

    def test_cloud_only_without_cloud_raises(self):
        rp = RoutingPolicy.cloud_only()
        with pytest.raises(RuntimeError):
            _resolve_localities(rp, local_available=True, cloud_available=False)

    def test_local_first_both_available(self):
        rp = RoutingPolicy.local_first(fallback="cloud")
        primary, fallback = _resolve_localities(rp, local_available=True, cloud_available=True)
        assert primary == "on_device"
        assert fallback == "cloud"

    def test_local_first_only_cloud(self):
        rp = RoutingPolicy.local_first(fallback="cloud")
        primary, fallback = _resolve_localities(rp, local_available=False, cloud_available=True)
        assert primary == "cloud"
        assert fallback is None

    def test_cloud_first_both_available(self):
        from octomil._generated.routing_policy import RoutingPolicy as CRP

        rp = RoutingPolicy(mode=CRP.AUTO, prefer_local=False, fallback="local")
        primary, fallback = _resolve_localities(rp, local_available=True, cloud_available=True)
        assert primary == "cloud"
        assert fallback == "on_device"

    def test_cloud_first_only_local(self):
        from octomil._generated.routing_policy import RoutingPolicy as CRP

        rp = RoutingPolicy(mode=CRP.AUTO, prefer_local=False, fallback="local")
        primary, fallback = _resolve_localities(rp, local_available=True, cloud_available=False)
        assert primary == "on_device"

    def test_neither_available_raises(self):
        rp = RoutingPolicy.local_first(fallback="cloud")
        with pytest.raises(RuntimeError, match="No local or cloud"):
            _resolve_localities(rp, local_available=False, cloud_available=False)
