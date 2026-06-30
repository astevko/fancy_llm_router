"""Tests for shared API key authentication."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fancy_llm_router.api.auth import (
    ApiKeyMiddleware,
    extract_api_key,
    path_requires_auth,
    verify_api_key,
)
from fancy_llm_router.schemas.config import AppConfig


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/health", False),
        ("/api/v1/complete", True),
        ("/api/v1/models", True),
        ("/analytics", False),
    ],
)
def test_path_requires_auth(path, expected):
    assert path_requires_auth(path) is expected


def test_extract_api_key_bearer():
    class Req:
        headers = {"Authorization": "Bearer secret-token"}

    assert extract_api_key(Req()) == "secret-token"


def test_extract_api_key_header():
    class Req:
        headers = {"X-API-Key": "header-token"}

    assert extract_api_key(Req()) == "header-token"


def test_verify_api_key_constant_time():
    assert verify_api_key("abc", "abc")
    assert not verify_api_key("abc", "def")
    assert not verify_api_key(None, "abc")


@pytest.fixture
def auth_app():
    app = FastAPI()
    app.state.config = AppConfig(api_auth_token="test-secret")
    app.add_middleware(ApiKeyMiddleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/models")
    def models():
        return ["mock@mock"]

    return app


def test_protected_route_requires_key(auth_app):
    client = TestClient(auth_app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/models").status_code == 401
    ok = client.get(
        "/api/v1/models",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert ok.status_code == 200
    assert ok.json() == ["mock@mock"]


def test_protected_route_accepts_x_api_key(auth_app):
    client = TestClient(auth_app)
    ok = client.get("/api/v1/models", headers={"X-API-Key": "test-secret"})
    assert ok.status_code == 200


def test_open_when_no_server_key_configured():
    app = FastAPI()
    app.state.config = AppConfig(api_auth_token=None)
    app.add_middleware(ApiKeyMiddleware)

    @app.get("/api/v1/models")
    def models():
        return []

    client = TestClient(app)
    assert client.get("/api/v1/models").status_code == 200
