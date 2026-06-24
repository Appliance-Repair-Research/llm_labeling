import json
import os
import tempfile

from openai import OpenAI
CUR_DIR = os.path.dirname(__file__)

class GPTClient:
    def __init__(self):
        self.load_config_env(os.path.join(CUR_DIR, "config.env"))
        api_key = os.getenv("GPT_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-5.2"

    @staticmethod
    def load_config_env(path: str = "config.env"):
        if not os.path.exists(path):
            return

        with open(path, "r") as config_file:
            for line in config_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                separator = "=" if "=" in line else ":"
                key, value = line.split(separator, 1)
                os.environ.setdefault(key.strip(), value.strip())

    def get_response(self, messages: list, max_tokens: int = 10, temperature: float = 0):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip().lower()

    def create_batch(self, requests: list, metadata: dict | None = None):
        batch_file_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as batch_file:
                batch_file_path = batch_file.name
                for index, request in enumerate(requests):
                    body = request.get("body")
                    if body is None:
                        body = {
                            "model": self.model,
                            "messages": request["messages"],
                            "temperature": request.get("temperature", 0),
                            "max_completion_tokens": request.get("max_tokens", 10),
                        }
                    elif "model" not in body:
                        body = {"model": self.model, **body}

                    batch_file.write(json.dumps({
                        "custom_id": str(request.get("custom_id", index)),
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }) + "\n")

            with open(batch_file_path, "rb") as batch_file:
                uploaded_file = self.client.files.create(file=batch_file, purpose="batch")

            return self.client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata,
            )
        finally:
            if batch_file_path:
                os.remove(batch_file_path)

    def get_batch_status(self, batch_id: str):
        return self.client.batches.retrieve(batch_id)

    def get_batch_results(self, batch_id: str) -> list:
        batch = self.get_batch_status(batch_id)
        if not batch.output_file_id:
            raise RuntimeError(f"Batch is not ready. Current status: {batch.status}")

        response = self.client.files.content(batch.output_file_id)
        text = response.text if hasattr(response, "text") else response.read().decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]
