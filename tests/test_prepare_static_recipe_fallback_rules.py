"""Issue F: ``ExecutionKernel.prepare()`` / ``warmup()`` substitution rules.

The kernel — not ``PrepareManager`` — owns the decision to substitute
the public static recipe for a planner candidate. ``PrepareManager``
correctly stays strict and rejects an echo-only candidate with no
``download_urls``. The kernel must check four conditions to decide:

  - direct vs app-scoped (``app=`` / ``@app/...`` is app-scoped)
  - planner candidate present?
  - candidate has meaningful artifact identity (digest, or
    ``artifact_id != model``)?
  - candidate identity matches the static recipe?

Substitution is allowed iff:
  - request is direct (not app-scoped), AND
  - planner candidate is missing OR carries no meaningful identity,
    AND
  - a static recipe is registered for ``(model, capability)``.

A planner candidate that explicitly carries
``source='static_recipe', recipe_id=...`` is NOT a substitution —
``PrepareManager._expand_static_recipe_source`` handles it.
"""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock, patch


def _make_candidate(
    *,
    model_id: str,
    artifact_id: str | None = None,
    digest: str | None = None,
    download_urls: list | None = None,
    source: Literal["static_recipe"] | None = None,
    recipe_id: str | None = None,
) -> Any:
    """Build a minimal RuntimeCandidatePlan for the helper tests."""
    from octomil.runtime.planner.schemas import (
        RuntimeArtifactPlan,
        RuntimeCandidatePlan,
    )

    artifact = RuntimeArtifactPlan(
        model_id=model_id,
        artifact_id=artifact_id,
        digest=digest,
        download_urls=download_urls or [],
        source=source,
        recipe_id=recipe_id,
    )
    return RuntimeCandidatePlan(
        locality="local",
        engine="sherpa-onnx",
        artifact=artifact,
        priority=0,
        confidence=1.0,
        reason="test",
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )


# ---------------------------------------------------------------------------
# _select_prepare_candidate — pure helper unit tests
# ---------------------------------------------------------------------------


def test_select_substitutes_static_recipe_for_direct_echo_only_kokoro():
    """Direct ``model='kokoro-82m'``, planner echoes the model name
    with no digest / urls. Kernel substitutes the static recipe."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    echo = _make_candidate(
        model_id="kokoro-82m",
        artifact_id="kokoro-82m",  # same as model — echo only
        digest=None,
        download_urls=[],
    )
    candidate, recipe = kernel._select_prepare_candidate(
        effective_model="kokoro-82m",
        capability="tts",
        planner_candidate=echo,
        app_scoped=False,
    )
    assert candidate is not None
    assert recipe is not None
    # Substituted candidate must carry the recipe's digest + urls.
    assert candidate.artifact.digest is not None and candidate.artifact.digest.startswith("sha256:")
    assert len(candidate.artifact.download_urls) >= 1


def test_select_does_not_substitute_for_meaningful_mismatched_identity():
    """Planner committed to a different artifact (private re-cut).
    Kernel must NOT substitute the public recipe — let PrepareManager
    prepare the planner-named artifact."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    private = _make_candidate(
        model_id="kokoro-82m",
        artifact_id="private-kokoro-v2",
        digest="sha256:" + "ab" * 32,  # meaningful identity, mismatches recipe
        download_urls=[],
    )
    candidate, recipe = kernel._select_prepare_candidate(
        effective_model="kokoro-82m",
        capability="tts",
        planner_candidate=private,
        app_scoped=False,
    )
    # Returned the planner candidate as-is; no static-recipe
    # substitution. PrepareManager will prepare or reject based on
    # the candidate's actual download_urls / digest.
    assert candidate is private
    assert recipe is None


def test_select_does_not_substitute_for_app_scoped_echo_only():
    """App-scoped requests must never silently substitute the public
    static recipe even on an echo-only planner candidate. The user
    asked for the app's artifact, not the public Kokoro."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    echo = _make_candidate(
        model_id="kokoro-82m",
        artifact_id="kokoro-82m",
        digest=None,
        download_urls=[],
    )
    candidate, recipe = kernel._select_prepare_candidate(
        effective_model="kokoro-82m",
        capability="tts",
        planner_candidate=echo,
        app_scoped=True,
    )
    assert candidate is echo
    assert recipe is None


def test_select_passes_through_planner_static_recipe_source():
    """``source='static_recipe'`` is the planner explicitly choosing
    the built-in recipe. PrepareManager's ``_expand_static_recipe_source``
    handles the materialization — the kernel must not also try to
    substitute. Returns ``recipe=None`` so the kernel doesn't run
    Materializer twice."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    explicit = _make_candidate(
        model_id="kokoro-82m",
        artifact_id="kokoro-82m",
        digest="sha256:912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7",
        source="static_recipe",
        recipe_id="kokoro-82m",
    )
    candidate, recipe = kernel._select_prepare_candidate(
        effective_model="kokoro-82m",
        capability="tts",
        planner_candidate=explicit,
        app_scoped=True,  # app-scoped doesn't matter when planner asks explicitly
    )
    assert candidate is explicit
    assert recipe is None


def test_select_substitutes_when_no_planner_candidate_at_all():
    """Embedded / offline planner returns nothing for a direct
    public model. Substitute the static recipe so the embedded TTS
    bootstrap one-liner works."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    candidate, recipe = kernel._select_prepare_candidate(
        effective_model="kokoro-82m",
        capability="tts",
        planner_candidate=None,
        app_scoped=False,
    )
    assert candidate is not None
    assert recipe is not None


# ---------------------------------------------------------------------------
# Integration: ExecutionKernel.prepare() / warmup() use the helper
# ---------------------------------------------------------------------------


def test_kernel_prepare_uses_static_recipe_for_direct_echo_only(monkeypatch):
    """End-to-end: direct ``prepare(model='kokoro-82m', capability='tts')``
    when the planner emits an echo-only candidate must NOT pass the
    echo-only candidate to PrepareManager. It must substitute the
    static recipe (which carries download_urls + digest)."""
    import asyncio

    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()
    kernel._prepare_manager = MagicMock()
    kernel._warmed_backends = {}

    # Stub PrepareManager.prepare to capture what the kernel hands it.
    captured: dict = {}

    def stub_prepare(candidate, mode=None):
        captured["candidate"] = candidate
        outcome = MagicMock()
        outcome.artifact_dir = MagicMock()
        return outcome

    kernel._prepare_manager.prepare = stub_prepare

    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )
    echo = _make_candidate(
        model_id="kokoro-82m",
        artifact_id="kokoro-82m",
        digest=None,
        download_urls=[],
    )
    selection = MagicMock()

    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
        patch("octomil.execution.kernel._local_sdk_runtime_candidate", return_value=echo),
        patch("octomil.runtime.lifecycle.materialization.Materializer") as MockMat,
    ):
        MockMat.return_value.materialize = MagicMock()
        asyncio.run(_call_prepare(kernel, model="kokoro-82m", capability="tts"))

    # The candidate handed to PrepareManager must NOT be the
    # echo-only one — it must be the substituted static recipe
    # candidate carrying the canonical Kokoro download URL + digest.
    handed = captured["candidate"]
    assert handed is not echo, "kernel should NOT pass the echo-only candidate to PrepareManager"
    assert handed.artifact.digest is not None
    assert len(handed.artifact.download_urls) >= 1


def test_kernel_prepare_does_not_substitute_for_app_scoped(monkeypatch):
    """App-scoped (``app='tts-tester'``) requests with echo-only
    planner candidates must NOT silently substitute the public
    Kokoro recipe — even though the underlying runtime model is
    ``kokoro-82m``. The user asked for the app's artifact."""
    import asyncio

    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()
    kernel._prepare_manager = MagicMock()
    kernel._warmed_backends = {}

    captured: dict = {}

    def stub_prepare(candidate, mode=None):
        captured["candidate"] = candidate
        outcome = MagicMock()
        outcome.artifact_dir = MagicMock()
        return outcome

    kernel._prepare_manager.prepare = stub_prepare

    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )
    echo = _make_candidate(
        model_id="kokoro-82m",
        artifact_id="kokoro-82m",
        digest=None,
        download_urls=[],
    )
    selection = MagicMock()

    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
        patch("octomil.execution.kernel._local_sdk_runtime_candidate", return_value=echo),
    ):
        # We pass app='tts-tester' to make the request app-scoped.
        # PrepareManager will reject the echo-only candidate; the
        # kernel must hand the planner candidate through unchanged
        # so that rejection reaches the caller. Catch it.
        try:
            asyncio.run(_call_prepare(kernel, model="kokoro-82m", capability="tts", app="tts-tester"))
        except Exception:
            pass

    handed = captured.get("candidate")
    assert handed is echo, (
        "kernel must hand the planner candidate (not the static recipe) for app-scoped "
        "requests; PrepareManager surfaces the rejection and the user fixes the planner."
    )


def test_kernel_warmup_uses_same_static_recipe_fallback(monkeypatch):
    """``warmup()`` must follow the same substitution rules as
    ``prepare()`` — otherwise prepare succeeds but warmup raises
    "no local sdk_runtime candidate" on identical inputs."""
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()
    kernel._prepare_manager = MagicMock()
    kernel._warmed_backends = {}

    captured: dict = {}

    def stub_prepare(candidate, mode=None):
        captured["candidate"] = candidate
        outcome = MagicMock()
        outcome.artifact_dir = MagicMock()
        return outcome

    kernel._prepare_manager.prepare = stub_prepare

    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )
    echo = _make_candidate(
        model_id="kokoro-82m",
        artifact_id="kokoro-82m",
        digest=None,
        download_urls=[],
    )
    selection = MagicMock()

    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
        patch("octomil.execution.kernel._local_sdk_runtime_candidate", return_value=echo),
        patch("octomil.execution.kernel._runtime_model_for_selection", return_value="kokoro-82m"),
        patch.object(kernel, "_resolve_local_tts_backend", return_value=None),
        patch("octomil.runtime.lifecycle.materialization.Materializer") as MockMat,
    ):
        MockMat.return_value.materialize = MagicMock()
        kernel.warmup(model="kokoro-82m", capability="tts")

    handed = captured["candidate"]
    assert handed is not echo, "warmup must mirror prepare's static-recipe substitution"
    assert handed.artifact.digest is not None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _call_prepare(kernel, **kwargs):
    """``ExecutionKernel.prepare`` is sync; this wrapper exists so
    the integration tests that spin up an event loop have a
    consistent shape with future async variants."""
    return kernel.prepare(**kwargs)
