import json
import os
import sys
import tempfile
import time

from google import genai
from google.genai import types
from prompt import prompt
from google.genai.errors import ServerError

CUR_DIR = os.path.dirname(__file__)


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


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        if api_key is None:
            load_config_env(os.path.join(CUR_DIR, "config.env"))

        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.types = types
        self.model = model

    def _split_messages(self, messages: list) -> tuple[str, str]:
        system_instruction = "\n".join(
            message["content"] for message in messages if message.get("role") == "system"
        )
        contents = "\n\n".join(
            f"{message.get('role', 'user').upper()}: {message['content']}"
            for message in messages
            if message.get("role") != "system"
        )
        return system_instruction, contents

    def _generate_config(self, max_tokens: int, temperature: float):
        return self.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=self.types.ThinkingConfig(thinking_budget=0),
        )

    def _batch_request_body(self, messages: list, max_tokens: int, temperature: float) -> dict:
        system_instruction, contents = self._split_messages(messages)
        body = {
            "contents": [{"role": "user", "parts": [{"text": contents}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        return body

    def get_response(self, messages: list, max_tokens: int = 10, temperature: float = 0):
        system_instruction, contents = self._split_messages(messages)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self.types.GenerateContentConfig(
                system_instruction=system_instruction or None,
                **self._generate_config(max_tokens, temperature).model_dump(exclude_none=True),
            ),
        )
        return response.text.strip().lower()

    def create_batch(self, requests: list, metadata: dict | None = None, display_name: str | None = None):
        batch_file_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as batch_file:
                batch_file_path = batch_file.name
                for index, request in enumerate(requests):
                    body = request.get("body")
                    if body is None:
                        body = self._batch_request_body(
                            request["messages"],
                            request.get("max_tokens", 10),
                            request.get("temperature", 0),
                        )

                    batch_file.write(json.dumps({
                        "key": str(request.get("custom_id", index)),
                        "request": body,
                    }) + "\n")

            uploaded_file = self.client.files.upload(
                file=batch_file_path,
                config=self.types.UploadFileConfig(
                    display_name=display_name or (metadata or {}).get("source", "gemini-batch"),
                    mime_type="jsonl",
                ),
            )

            return self.client.batches.create(
                model=self.model,
                src=uploaded_file.name,
                config={"display_name": display_name or (metadata or {}).get("source", "gemini-batch")},
            )
        finally:
            if batch_file_path:
                os.remove(batch_file_path)

    def get_batch_status(self, batch_name: str):
        return self.client.batches.get(name=batch_name)

    def get_batch_results(self, batch_name: str) -> list:
        batch = self.get_batch_status(batch_name)
        state = batch.state.name if batch.state else "unknown"
        if state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Batch is not ready. Current status: {state}")

        if batch.dest and batch.dest.inlined_responses:
            return [response.model_dump(exclude_none=True) for response in batch.dest.inlined_responses]

        if not batch.dest or not batch.dest.file_name:
            raise RuntimeError("Batch completed without a result file.")

        file_content = self.client.files.download(file=batch.dest.file_name).decode("utf-8")
        results = [json.loads(line) for line in file_content.splitlines() if line.strip()]
        for result in results:
            if "key" in result and "custom_id" not in result:
                result["custom_id"] = result["key"]
        return results


if __name__ == "__main__":
    import pandas as pd

    input_path = os.path.join(CUR_DIR, "data", "call_data_rw.csv")
    result_column = "Gemini Result"
    requests_per_minute = 800
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    df = pd.read_csv(input_path)
    if result_column not in df.columns:
        df[result_column] = ""
    df[result_column] = df[result_column].fillna("").astype(str)

    client = GeminiClient()
    rows = df.iloc[start:] if limit is None else df.iloc[start:start + limit]
    window_started_at = time.monotonic()
    requests_in_window = 0

    for index, row in rows.iterrows():
        if pd.notna(row.get(result_column)) and str(row.get(result_column)).strip():
            continue

        if requests_in_window >= requests_per_minute:
            elapsed = time.monotonic() - window_started_at
            if elapsed < 60:
                sleep_for = 60 - elapsed
                print(f"rate limit reached, sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
            window_started_at = time.monotonic()
            requests_in_window = 0

        messages = [
            {
                "role": "system",
                "content": "You are a classification assistant for appliance repair company.",
            },
            {
                "role": "user",
                "content": f"{prompt}Conversation:\n{row['Script']}",
            },
        ]
        try:
            result = client.get_response(messages)
        except ServerError as e:
            #try one more time
            result = client.get_response(messages)
        requests_in_window += 1
        df.loc[index, result_column] = result
        df.to_csv(input_path, index=False)
        print(f"id={row['ID']} expected={row['Result']} gemini={result}")
