import httpx
import time
from typing import Optional

class InferenceClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def run_model(self, model_id: str, input_data: dict, poll: bool = True, poll_interval: int = 2, timeout: int = 60):
        """Send input data to the model and optionally poll for result."""
        try:
            response = httpx.post(
                f"{self.base_url}/models/{model_id}/infer",
                headers=self.headers,
                json={"input": input_data}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise Exception(f"API call failed: {e.response.status_code} - {e.response.text}")

        job_id = response.json().get("job_id")
        if not poll:
            return {"job_id": job_id}

        return self._poll_for_result(job_id, poll_interval, timeout)

    def _poll_for_result(self, job_id: str, poll_interval: int, timeout: int):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                res = httpx.get(f"{self.base_url}/jobs/{job_id}", headers=self.headers)
                res.raise_for_status()
                data = res.json()
                if data.get("status") == "completed":
                    return data.get("result")
                elif data.get("status") == "failed":
                    raise Exception("Model job failed: " + str(data.get("error")))
            except Exception as e:
                raise Exception(f"Polling error: {e}")

            time.sleep(poll_interval)

        raise TimeoutError("Polling timed out")