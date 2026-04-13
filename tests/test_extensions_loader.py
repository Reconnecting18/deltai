"""Ensure project/extensions discovery loads packages and mounts routes."""

from extensions import load_extensions
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_example_extension_hello_route():
    app = FastAPI()
    load_extensions(app)
    client = TestClient(app)
    r = client.get("/ext/example_extension/hello", params={"name": "pytest"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "example_extension"
    assert "pytest" in body["message"]
