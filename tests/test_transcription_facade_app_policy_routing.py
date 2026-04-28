"""Public-facade evidence for transcription/app_policy_routing.

The kernel-level refusal gate already exists (see
``test_routing_controls.py::test_transcribe_audio_refuses_app_ref_when_planner_unavailable_and_no_policy``);
these tests pin the contract that the public ``client.audio.transcriptions``
namespace on the unified ``Octomil`` facade actually exposes ``app=``
and ``policy=`` and routes through the kernel.

Without ``FacadeTranscriptions``, the unified facade had no public
transcription surface at all (only ``client.audio.speech.create`` was
exposed); the kernel's app/policy gates were unreachable from the
public facade. These tests pin the gates against the public path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from octomil.audio import FacadeAudio, FacadeTranscriptions
from octomil.errors import OctomilError, OctomilErrorCode
from octomil.execution.kernel import ExecutionResult

# ---------------------------------------------------------------------------
# Namespace shape
# ---------------------------------------------------------------------------


def test_facade_audio_exposes_transcriptions_namespace():
    fake_kernel = MagicMock()
    audio = FacadeAudio(fake_kernel)
    assert isinstance(audio.transcriptions, FacadeTranscriptions)
    assert callable(audio.transcriptions.create)


# ---------------------------------------------------------------------------
# Forwards through the kernel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facade_transcriptions_forwards_app_and_policy_to_kernel():
    captured: dict[str, Any] = {}

    async def fake_transcribe(audio_data, *, model, policy, app, language):
        captured["audio_data"] = audio_data
        captured["model"] = model
        captured["policy"] = policy
        captured["app"] = app
        captured["language"] = language
        return ExecutionResult(
            id="t1",
            model=model or "",
            capability="transcription",
            locality="on_device",
            output_text="hello",
        )

    kernel = MagicMock()
    kernel.transcribe_audio = fake_transcribe

    facade = FacadeTranscriptions(kernel)
    result = await facade.create(
        audio=b"\x00\x01\x02",
        model="@app/notes/transcription",
        language="en",
        policy="local_only",
        app="notes",
    )

    assert captured["audio_data"] == b"\x00\x01\x02"
    assert captured["model"] == "@app/notes/transcription"
    assert captured["policy"] == "local_only"
    assert captured["app"] == "notes"
    assert captured["language"] == "en"
    assert result.text == "hello"
    assert result.language == "en"


# ---------------------------------------------------------------------------
# Refusal gate at the public facade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facade_transcriptions_app_scoped_planner_offline_refuses():
    """``client.audio.transcriptions.create(model='whisper-tiny',
    app='notes')`` with the planner offline AND no explicit policy
    MUST raise. The kernel's refusal gate must reach the caller
    through the facade — the facade must not eat the error."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel()
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "whisper-tiny",
            "policy_preset": "local_first",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()

    facade = FacadeTranscriptions(kernel)
    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=None):
        with pytest.raises(OctomilError) as excinfo:
            await facade.create(
                audio=b"\x00",
                model="whisper-tiny",
                app="notes",
            )
    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    msg = str(excinfo.value)
    assert "notes" in msg


@pytest.mark.asyncio
async def test_facade_transcriptions_synthesizes_app_ref_into_planner_model():
    """Reviewer P1: ``app=`` must reach planner routing. The
    facade-level test for transcription, mirroring the speech one
    in ``test_routing_controls``."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel()
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "whisper-tiny",
            "policy_preset": "local_first",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()

    captured: dict[str, Any] = {}

    def fake_resolve_planner_selection(model, capability, policy_preset):
        captured["planner_model"] = model
        captured["capability"] = capability
        return None

    facade = FacadeTranscriptions(kernel)
    with patch(
        "octomil.execution.kernel._resolve_planner_selection",
        side_effect=fake_resolve_planner_selection,
    ):
        with pytest.raises(OctomilError):
            await facade.create(
                audio=b"\x00",
                model="whisper-tiny",
                app="notes",
            )
    assert captured["planner_model"] == "@app/notes/transcription", captured
    assert captured["capability"] == "transcription"


# ---------------------------------------------------------------------------
# Local_only / private must not silently leak to cloud at the facade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facade_transcriptions_local_only_routes_through_kernel():
    """``policy='local_only'`` must reach the kernel — without this
    the facade would drop the kwarg and the kernel's local-only gate
    would never fire on the public path."""
    captured: dict[str, Any] = {}

    async def fake_transcribe(audio_data, *, model, policy, app, language):
        captured["policy"] = policy
        return ExecutionResult(
            id="t1",
            model=model or "",
            capability="transcription",
            locality="on_device",
            output_text="local",
        )

    kernel = MagicMock()
    kernel.transcribe_audio = fake_transcribe

    facade = FacadeTranscriptions(kernel)
    await facade.create(
        audio=b"\x00",
        model="whisper-tiny",
        policy="local_only",
    )
    assert captured["policy"] == "local_only"
