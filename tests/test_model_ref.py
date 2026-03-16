"""Tests for octomil.model_ref — ModelRef discriminated union."""

from __future__ import annotations

from octomil._generated.model_capability import ModelCapability
from octomil.model_ref import (
    ModelRefFactory,
    _ModelRefCapability,
    _ModelRefId,
    get_capability,
    get_model_id,
    is_capability_ref,
    is_id_ref,
    model_ref_capability,
    model_ref_id,
)


class TestModelRefId:
    def test_create_by_id(self) -> None:
        ref = model_ref_id("gemma-2-2b-q4")
        assert isinstance(ref, _ModelRefId)
        assert ref.model_id == "gemma-2-2b-q4"

    def test_factory_id(self) -> None:
        ref = ModelRefFactory.id("gemma-2-2b-q4")
        assert isinstance(ref, _ModelRefId)
        assert ref.model_id == "gemma-2-2b-q4"

    def test_is_id_ref(self) -> None:
        ref = model_ref_id("test-model")
        assert is_id_ref(ref)
        assert not is_capability_ref(ref)

    def test_get_model_id(self) -> None:
        ref = model_ref_id("test-model")
        assert get_model_id(ref) == "test-model"
        assert get_capability(ref) is None

    def test_frozen(self) -> None:
        ref = model_ref_id("test")
        assert hash(ref) is not None


class TestModelRefCapability:
    def test_create_by_capability(self) -> None:
        ref = model_ref_capability(ModelCapability.CHAT)
        assert isinstance(ref, _ModelRefCapability)
        assert ref.capability == ModelCapability.CHAT

    def test_factory_capability(self) -> None:
        ref = ModelRefFactory.capability(ModelCapability.TRANSCRIPTION)
        assert isinstance(ref, _ModelRefCapability)
        assert ref.capability == ModelCapability.TRANSCRIPTION

    def test_is_capability_ref(self) -> None:
        ref = model_ref_capability(ModelCapability.CHAT)
        assert is_capability_ref(ref)
        assert not is_id_ref(ref)

    def test_get_capability(self) -> None:
        ref = model_ref_capability(ModelCapability.CHAT)
        assert get_capability(ref) == ModelCapability.CHAT
        assert get_model_id(ref) is None

    def test_all_capabilities(self) -> None:
        for cap in ModelCapability:
            ref = model_ref_capability(cap)
            assert get_capability(ref) == cap
