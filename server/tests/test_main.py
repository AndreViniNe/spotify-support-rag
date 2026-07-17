from fastapi.testclient import TestClient

from main import app


def test_invoke_chain_returns_generated_text():
    with TestClient(app) as client:
        response = client.get("/invoke_chain", params={"text": "anything"})

    assert response.status_code == 200
    assert isinstance(response.json(), str)
    assert len(response.json()) > 0
