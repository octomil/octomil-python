"""Single shared helper for parsing artifact dicts into RuntimeArtifactPlan.

Both ``client._parse_artifact`` (live HTTP planner response) and
``planner.plan_dict_to_*`` (cached plan rehydration) project the same
JSON-shaped artifact dict into the same Python type. They drifted in
the past — the cache path silently dropped ``required_files`` /
``download_urls`` / ``manifest_uri`` / ``source`` / ``recipe_id`` and
left app-scoped Kokoro candidates stuck on "download the tarball but
never run the static-recipe MaterializationPlan", failing at local
load. Centralizing the projection here is how we keep them honest.
"""

from __future__ import annotations

from typing import Any

from .schemas import ArtifactDownloadEndpoint, RuntimeArtifactPlan


def parse_download_endpoints(raw: Any) -> list[ArtifactDownloadEndpoint]:
    """Project a server-emitted ``download_urls`` list into typed endpoints.

    Tolerates the shape variants we've seen on the wire: a missing
    field, ``None``, an empty list, an entry with no ``url``. Each
    bad entry is dropped silently — we never want a malformed
    endpoint to crash the SDK at parse time.
    """
    out: list[ArtifactDownloadEndpoint] = []
    for ep in raw or []:
        if isinstance(ep, dict) and ep.get("url"):
            out.append(
                ArtifactDownloadEndpoint(
                    url=ep["url"],
                    expires_at=ep.get("expires_at"),
                    headers=ep.get("headers"),
                )
            )
    return out


def parse_artifact_dict(artifact_data: dict[str, Any]) -> RuntimeArtifactPlan:
    """Project an artifact dict (from live HTTP plan OR cached plan)
    into a :class:`RuntimeArtifactPlan`.

    All prepare-lifecycle fields are preserved:

      - ``required_files`` / ``download_urls`` / ``manifest_uri`` —
        without these the durable downloader has nothing to fetch.
      - ``source`` (``"static_recipe"``) + ``recipe_id`` — the
        discriminators ``PrepareManager._expand_static_recipe_source``
        keys off so the recipe's MaterializationPlan runs after the
        download lands. Without them the SDK leaves the raw archive
        on disk and the runtime engine fails to load.
    """
    return RuntimeArtifactPlan(
        model_id=artifact_data.get("model_id", ""),
        artifact_id=artifact_data.get("artifact_id"),
        model_version=artifact_data.get("model_version"),
        format=artifact_data.get("format"),
        quantization=artifact_data.get("quantization"),
        uri=artifact_data.get("uri"),
        digest=artifact_data.get("digest"),
        size_bytes=artifact_data.get("size_bytes"),
        min_ram_bytes=artifact_data.get("min_ram_bytes"),
        required_files=list(artifact_data.get("required_files", []) or []),
        download_urls=parse_download_endpoints(artifact_data.get("download_urls")),
        manifest_uri=artifact_data.get("manifest_uri"),
        source=artifact_data.get("source"),
        recipe_id=artifact_data.get("recipe_id"),
    )
