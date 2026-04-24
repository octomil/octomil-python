"""Tests for @app/{slug}/{capability} ref parsing and planner integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from octomil.execution.kernel import (
    _route_metadata_from_selection,
)
from octomil.runtime.planner.app_ref import is_app_ref, parse_app_ref
from octomil.runtime.planner.client import RuntimePlannerClient
from octomil.runtime.planner.planner import RuntimePlanner
from octomil.runtime.planner.schemas import (
    AppResolution,
    DeviceRuntimeProfile,
    InstalledRuntime,
    ModelResolution,
    RuntimeCandidatePlan,
    RuntimePlanResponse,
    RuntimeSelection,
)
from octomil.runtime.planner.store import RuntimePlannerStore

# ---------------------------------------------------------------------------
# parse_app_ref / is_app_ref
# ---------------------------------------------------------------------------


class TestParseAppRef:
    def test_valid_app_ref(self):
        slug, cap = parse_app_ref("@app/my-app/chat")
        assert slug == "my-app"
        assert cap == "chat"

    def test_valid_app_ref_with_underscores(self):
        slug, cap = parse_app_ref("@app/my_app/embeddings")
        assert slug == "my_app"
        assert cap == "embeddings"

    def test_not_app_ref_plain_model(self):
        slug, cap = parse_app_ref("gemma-3-1b")
        assert slug is None
        assert cap is None

    def test_not_app_ref_partial(self):
        slug, cap = parse_app_ref("@app/only-slug")
        assert slug is None
        assert cap is None

    def test_not_app_ref_extra_segments(self):
        slug, cap = parse_app_ref("@app/slug/cap/extra")
        assert slug is None
        assert cap is None

    def test_not_app_ref_empty(self):
        slug, cap = parse_app_ref("")
        assert slug is None
        assert cap is None

    def test_is_app_ref_true(self):
        assert is_app_ref("@app/test/chat") is True

    def test_is_app_ref_false(self):
        assert is_app_ref("gemma-3-1b") is False

    def test_is_app_ref_prefix_only(self):
        # Starts with @app/ but missing full structure -- still True for
        # the prefix check (parse_app_ref would return None, None).
        assert is_app_ref("@app/") is True


# ---------------------------------------------------------------------------
# AppResolution schema
# ---------------------------------------------------------------------------


class TestAppResolution:
    def test_dataclass_fields(self):
        ar = AppResolution(
            app_id="app-123",
            capability="chat",
            routing_policy="private",
            selected_model="gemma3-1b",
            app_slug="my-app",
        )
        assert ar.app_id == "app-123"
        assert ar.selected_model == "gemma3-1b"
        assert ar.routing_policy == "private"
        assert ar.preferred_engines == []
        assert ar.artifact_candidates == []

    def test_plan_response_with_app_resolution(self):
        ar = AppResolution(
            app_id="app-123",
            capability="chat",
            routing_policy="local_first",
            selected_model="gemma3-1b",
        )
        plan = RuntimePlanResponse(
            model="@app/my-app/chat",
            capability="chat",
            policy="local_first",
            candidates=[],
            app_resolution=ar,
        )
        assert plan.app_resolution is not None
        assert plan.app_resolution.selected_model == "gemma3-1b"


# ---------------------------------------------------------------------------
# Planner client -- app_slug passthrough
# ---------------------------------------------------------------------------


class TestClientPassesAppSlug:
    def test_fetch_plan_includes_app_slug_in_payload(self):
        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "model": "@app/my-app/chat",
                "capability": "chat",
                "policy": "local_first",
                "candidates": [],
                "app_resolution": {
                    "app_id": "app-123",
                    "capability": "chat",
                    "routing_policy": "local_first",
                    "selected_model": "gemma3-1b",
                    "app_slug": "my-app",
                    "preferred_engines": ["mlx-lm"],
                },
            },
            request=httpx.Request("POST", "http://test/api/v2/runtime/plan"),
        )

        client = RuntimePlannerClient(base_url="http://test", api_key="key")
        device = DeviceRuntimeProfile(
            sdk="python",
            sdk_version="4.6.0",
            platform="Darwin",
            arch="arm64",
        )

        with patch("httpx.Client") as mock_client_cls:
            mock_http = MagicMock()
            mock_http.__enter__ = MagicMock(return_value=mock_http)
            mock_http.__exit__ = MagicMock(return_value=False)
            mock_http.post.return_value = mock_response
            mock_client_cls.return_value = mock_http

            plan = client.fetch_plan(
                model="@app/my-app/chat",
                capability="chat",
                device=device,
                app_slug="my-app",
            )

            # Verify app_slug is in the POST payload
            call_args = mock_http.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert payload["app_slug"] == "my-app"

        assert plan is not None
        assert plan.app_resolution is not None
        assert plan.app_resolution.selected_model == "gemma3-1b"
        assert plan.app_resolution.app_slug == "my-app"
        assert plan.app_resolution.preferred_engines == ["mlx-lm"]

    def test_fetch_plan_omits_app_slug_when_none(self):
        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "model": "gemma-2b",
                "capability": "text",
                "policy": "local_first",
                "candidates": [],
            },
            request=httpx.Request("POST", "http://test/api/v2/runtime/plan"),
        )

        client = RuntimePlannerClient(base_url="http://test", api_key="key")
        device = DeviceRuntimeProfile(
            sdk="python",
            sdk_version="4.6.0",
            platform="Darwin",
            arch="arm64",
        )

        with patch("httpx.Client") as mock_client_cls:
            mock_http = MagicMock()
            mock_http.__enter__ = MagicMock(return_value=mock_http)
            mock_http.__exit__ = MagicMock(return_value=False)
            mock_http.post.return_value = mock_response
            mock_client_cls.return_value = mock_http

            client.fetch_plan(
                model="gemma-2b",
                capability="text",
                device=device,
            )

            call_args = mock_http.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "app_slug" not in payload


# ---------------------------------------------------------------------------
# Planner -- app ref resolution
# ---------------------------------------------------------------------------


def _make_planner(tmp_path: Path, *, client: RuntimePlannerClient | None = None) -> RuntimePlanner:
    db = tmp_path / "planner.sqlite3"
    store = RuntimePlannerStore(db_path=db)
    return RuntimePlanner(store=store, client=client)


class TestAppRefResolvesThroughPlanner:
    def test_app_ref_passes_app_slug_to_client(self, tmp_path: Path):
        """When model is @app/my-app/chat, planner sends app_slug='my-app' to client."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        mock_client.fetch_plan.return_value = RuntimePlanResponse(
            model="@app/my-app/chat",
            capability="chat",
            policy="local_first",
            candidates=[
                RuntimeCandidatePlan(
                    locality="local",
                    priority=1,
                    confidence=0.9,
                    reason="mlx available",
                    engine="mlx-lm",
                )
            ],
            app_resolution=AppResolution(
                app_id="app-123",
                capability="chat",
                routing_policy="local_first",
                selected_model="gemma3-1b",
            ),
        )

        planner = _make_planner(tmp_path, client=mock_client)

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile:
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
                installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
            )
            result = planner.resolve(
                model="@app/my-app/chat",
                capability="responses",
                routing_policy="local_first",
            )

        # Verify app_slug was passed to fetch_plan
        call_kwargs = mock_client.fetch_plan.call_args[1]
        assert call_kwargs["app_slug"] == "my-app"
        assert result.engine == "mlx-lm"

    def test_app_ref_uses_resolved_model_for_local_engine(self, tmp_path: Path):
        """Server returns app_resolution.selected_model -- planner uses it for local resolution."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        mock_client.fetch_plan.return_value = RuntimePlanResponse(
            model="@app/my-app/chat",
            capability="chat",
            policy="local_first",
            candidates=[
                RuntimeCandidatePlan(
                    locality="local",
                    priority=1,
                    confidence=0.9,
                    reason="local engine",
                    engine="mlx-lm",
                )
            ],
            app_resolution=AppResolution(
                app_id="app-123",
                capability="chat",
                routing_policy="local_first",
                selected_model="gemma3-1b",
            ),
        )

        planner = _make_planner(tmp_path, client=mock_client)

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile:
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
                installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
            )
            result = planner.resolve(
                model="@app/my-app/chat",
                capability="responses",
                routing_policy="local_first",
            )

        assert result.app_resolution is not None
        assert result.app_resolution.selected_model == "gemma3-1b"
        assert result.source == "server_plan"

    def test_app_ref_refreshes_server_plan_when_network_available(self, tmp_path: Path):
        """App refs should re-fetch the live plan so policy flips apply immediately."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        mock_client.fetch_plan.side_effect = [
            RuntimePlanResponse(
                model="@app/my-app/chat",
                capability="chat",
                policy="local_first",
                candidates=[
                    RuntimeCandidatePlan(
                        locality="local",
                        priority=1,
                        confidence=0.9,
                        reason="local engine",
                        engine="mlx-lm",
                    )
                ],
                app_resolution=AppResolution(
                    app_id="app-123",
                    capability="chat",
                    routing_policy="local_first",
                    selected_model="gemma3-1b",
                ),
            ),
            RuntimePlanResponse(
                model="@app/my-app/chat",
                capability="chat",
                policy="local_first",
                candidates=[
                    RuntimeCandidatePlan(
                        locality="local",
                        priority=1,
                        confidence=0.9,
                        reason="local engine",
                        engine="mlx-lm",
                    )
                ],
                app_resolution=AppResolution(
                    app_id="app-123",
                    capability="chat",
                    routing_policy="private",
                    selected_model="gemma3-1b",
                ),
            ),
        ]

        planner = _make_planner(tmp_path, client=mock_client)

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile:
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
                installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
            )
            first = planner.resolve(
                model="@app/my-app/chat",
                capability="responses",
                routing_policy="local_first",
            )
            second = planner.resolve(
                model="@app/my-app/chat",
                capability="responses",
                routing_policy="local_first",
            )

        assert first.app_resolution is not None
        assert first.app_resolution.selected_model == "gemma3-1b"
        assert second.app_resolution is not None
        assert second.app_resolution.routing_policy == "private"
        assert second.source == "server_plan"
        assert mock_client.fetch_plan.call_count == 2

    def test_app_ref_uses_cached_plan_when_offline(self, tmp_path: Path):
        """A cached app plan still works as an offline fallback."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        mock_client.fetch_plan.return_value = RuntimePlanResponse(
            model="@app/my-app/chat",
            capability="chat",
            policy="local_first",
            candidates=[
                RuntimeCandidatePlan(
                    locality="local",
                    priority=1,
                    confidence=0.9,
                    reason="local engine",
                    engine="mlx-lm",
                )
            ],
            app_resolution=AppResolution(
                app_id="app-123",
                capability="chat",
                routing_policy="local_first",
                selected_model="gemma3-1b",
            ),
        )

        planner = _make_planner(tmp_path, client=mock_client)

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile:
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
                installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
            )
            planner.resolve(
                model="@app/my-app/chat",
                capability="responses",
                routing_policy="local_first",
            )

        mock_client.fetch_plan.reset_mock()

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile:
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
                installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
            )
            result = planner.resolve(
                model="@app/my-app/chat",
                capability="responses",
                routing_policy="local_first",
                allow_network=False,
            )

        mock_client.fetch_plan.assert_not_called()
        assert result.app_resolution is not None
        assert result.app_resolution.selected_model == "gemma3-1b"
        assert result.source == "cache"

    def test_cached_plan_preserves_generic_resolution(self, tmp_path: Path):
        """Cached plans must preserve generic resolution for alias/deployment refs."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        mock_client.fetch_plan.return_value = RuntimePlanResponse(
            model="alias:production",
            capability="chat",
            policy="local_first",
            candidates=[
                RuntimeCandidatePlan(
                    locality="local",
                    priority=1,
                    confidence=0.9,
                    reason="local engine",
                    engine="mlx-lm",
                )
            ],
            resolution=ModelResolution(
                ref_kind="alias",
                original_ref="alias:production",
                resolved_model="phi-4-mini",
                variant_id="variant-b",
            ),
        )

        planner = _make_planner(tmp_path, client=mock_client)

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile:
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
                installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
            )
            first = planner.resolve(
                model="alias:production",
                capability="responses",
                routing_policy="local_first",
            )
            mock_client.fetch_plan.side_effect = AssertionError("planner should use cache on second resolve")
            second = planner.resolve(
                model="alias:production",
                capability="responses",
                routing_policy="local_first",
            )

        assert first.resolution is not None
        assert first.resolution.resolved_model == "phi-4-mini"
        assert second.resolution is not None
        assert second.resolution.resolved_model == "phi-4-mini"
        assert second.source == "cache"

    def test_app_ref_private_never_calls_hosted(self, tmp_path: Path):
        """When app_resolution.routing_policy is 'private', planner must not call cloud."""
        mock_client = MagicMock(spec=RuntimePlannerClient)

        planner = _make_planner(tmp_path, client=mock_client)

        with (
            patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile,
            patch("octomil.runtime.planner.planner.RuntimePlanner._resolve_locally") as mock_local,
        ):
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
            )
            mock_local.return_value = RuntimeSelection(
                locality="local",
                engine="mlx-lm",
                source="local_benchmark",
                reason="private local",
            )
            result = planner.resolve(
                model="@app/my-app/chat",
                capability="responses",
                routing_policy="private",
            )

        # Private policy should NOT call fetch_plan
        mock_client.fetch_plan.assert_not_called()
        assert result.locality == "local"

    def test_app_ref_capability_overrides_from_ref(self, tmp_path: Path):
        """The capability parsed from @app/{slug}/{cap} should override the call arg."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        mock_client.fetch_plan.return_value = RuntimePlanResponse(
            model="@app/my-app/embeddings",
            capability="embeddings",
            policy="local_first",
            candidates=[
                RuntimeCandidatePlan(
                    locality="cloud",
                    priority=1,
                    confidence=0.9,
                    reason="cloud",
                    engine=None,
                )
            ],
        )

        planner = _make_planner(tmp_path, client=mock_client)

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile:
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
            )
            planner.resolve(
                model="@app/my-app/embeddings",
                capability="responses",  # should be overridden to "embeddings"
                routing_policy="local_first",
            )

        call_kwargs = mock_client.fetch_plan.call_args[1]
        assert call_kwargs["capability"] == "embeddings"

    def test_app_ref_server_policy_overrides_caller(self, tmp_path: Path):
        """Server app_resolution.routing_policy should override the caller's policy."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        mock_client.fetch_plan.return_value = RuntimePlanResponse(
            model="@app/my-app/chat",
            capability="chat",
            policy="local_first",
            candidates=[],
            app_resolution=AppResolution(
                app_id="app-123",
                capability="chat",
                routing_policy="private",
                selected_model="gemma3-1b",
            ),
        )

        planner = _make_planner(tmp_path, client=mock_client)

        with (
            patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile,
            patch("octomil.runtime.planner.planner.RuntimePlanner._resolve_locally") as mock_local,
        ):
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
            )
            mock_local.return_value = RuntimeSelection(
                locality="local",
                engine="mlx-lm",
                source="local_benchmark",
                reason="local",
            )
            planner.resolve(
                model="@app/my-app/chat",
                capability="responses",
                routing_policy="local_first",
            )

        # Local resolution should be called with the server-overridden
        # private policy. Since no candidates matched, _resolve_locally
        # is called and should receive the private routing_policy.
        call_kwargs = mock_local.call_args[1]
        assert call_kwargs["routing_policy"] == "private"
        assert call_kwargs["is_private"] is True


# ---------------------------------------------------------------------------
# RouteMetadata for app refs
# ---------------------------------------------------------------------------


class TestRouteMetadataForAppRefs:
    def test_app_ref_sets_model_kind_to_app(self):
        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server_plan",
            reason="test",
        )
        route = _route_metadata_from_selection(
            selection,
            "on_device",
            False,
            model_name="@app/my-app/chat",
            capability="responses",
        )
        assert route.model.requested.kind == "app"

    def test_app_ref_sets_capability_from_ref(self):
        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server_plan",
            reason="test",
        )
        route = _route_metadata_from_selection(
            selection,
            "on_device",
            False,
            model_name="@app/my-app/chat",
            capability="responses",
        )
        assert route.model.requested.capability == "chat"

    def test_app_ref_shows_resolved_model(self):
        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server_plan",
            reason="test",
            app_resolution=AppResolution(
                app_id="app-123",
                capability="chat",
                routing_policy="local_first",
                selected_model="gemma3-1b",
                selected_model_variant_id="var-456",
                selected_model_version="v1.0",
            ),
        )
        route = _route_metadata_from_selection(
            selection,
            "on_device",
            False,
            model_name="@app/my-app/chat",
            capability="responses",
        )
        assert route.model.resolved is not None
        assert route.model.resolved.slug == "gemma3-1b"
        assert route.model.resolved.variant_id == "var-456"
        assert route.model.resolved.version_id == "v1.0"

    def test_plain_model_keeps_kind_model(self):
        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server_plan",
            reason="test",
        )
        route = _route_metadata_from_selection(
            selection,
            "on_device",
            False,
            model_name="gemma-2b",
            capability="responses",
        )
        assert route.model.requested.kind == "model"
        assert route.model.resolved is None

    def test_no_selection_app_ref(self):
        route = _route_metadata_from_selection(
            None,
            "on_device",
            False,
            model_name="@app/my-app/chat",
            capability="responses",
        )
        assert route.model.requested.kind == "app"
        assert route.model.requested.capability == "chat"

    def test_execution_locality_is_public(self):
        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server_plan",
            reason="test",
        )
        route = _route_metadata_from_selection(
            selection,
            "on_device",
            False,
            model_name="@app/my-app/chat",
        )
        assert route.execution is not None
        assert route.execution.locality == "local"
        assert route.execution.mode == "sdk_runtime"

    def test_cloud_execution_sets_hosted_gateway(self):
        selection = RuntimeSelection(
            locality="cloud",
            engine=None,
            source="server_plan",
            reason="cloud route",
        )
        route = _route_metadata_from_selection(
            selection,
            "cloud",
            False,
            model_name="@app/my-app/chat",
        )
        assert route.execution is not None
        assert route.execution.locality == "cloud"
        assert route.execution.mode == "hosted_gateway"
