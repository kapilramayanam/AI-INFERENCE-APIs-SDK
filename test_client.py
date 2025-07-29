import pytest
from ai_sdk.client import InferenceClient

class MockHTTPX:
    def __init__(self):
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append(("POST", url, json))
        return MockResponse({"job_id": "12345"}, 200)

    def get(self, url, headers=None):
        self.calls.append(("GET", url))
        return MockResponse({"status": "completed", "result": {"output": "success"}}, 200)

class MockResponse:
    def __init__(self, data, status):
        self._json = data
        self.status_code = status
        self.text = str(data)

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code != 200:
            raise httpx.HTTPStatusError("", request=None, response=self)

@pytest.fixture
def mock_client(monkeypatch):
    client = InferenceClient("http://fake.api", "test-token")
    mock_http = MockHTTPX()
    monkeypatch.setattr("httpx.post", mock_http.post)
    monkeypatch.setattr("httpx.get", mock_http.get)
    return client

def test_run_model_success(mock_client):
    result = mock_client.run_model("test-model", {"text": "hello"})
    assert result == {"output": "success"}
