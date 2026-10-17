from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from octomil.serve.security import install_security


def _app(token=None):
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    install_security(app, token)
    return app


def test_no_token_leaves_api_open(monkeypatch):
    monkeypatch.delenv("OCTOMIL_SERVE_TOKEN", raising=False)
    assert TestClient(_app()).get("/v1/models").status_code == 200


def test_token_required(monkeypatch):
    monkeypatch.delenv("OCTOMIL_SERVE_TOKEN", raising=False)
    c = TestClient(_app("s3cret"))
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/health").status_code == 200


def test_cors_not_wildcard(monkeypatch):
    monkeypatch.delenv("OCTOMIL_SERVE_TOKEN", raising=False)
    c = TestClient(_app())
    evil = c.get("/v1/models", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
    local = c.get("/v1/models", headers={"Origin": "http://localhost:3000"})
    assert local.headers["access-control-allow-origin"] == "http://localhost:3000"
