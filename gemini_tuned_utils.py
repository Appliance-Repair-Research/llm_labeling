import os
import sys
import time

import pandas as pd
import requests

from prompt import prompt

CUR_DIR = os.path.dirname(__file__)


class TunedGeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
    ):
        self.load_config_env(os.path.join(CUR_DIR, "config.env"))

        self.api_key = api_key or os.getenv("GOOGLE_CLOUD_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_CLOUD_API_KEY is missing")

        self.endpoint = (
            "https://us-central1-aiplatform.googleapis.com/v1/"
            "projects/1015352592922/"
            "locations/us-central1/"
            "endpoints/5869906102358900736:generateContent"
        )

    def _split_messages(self, messages: list) -> tuple[str, str]:
        system_instruction = "\n".join(
            message["content"]
            for message in messages
            if message.get("role") == "system"
        )

        user_content = "\n\n".join(
            message["content"]
            for message in messages
            if message.get("role") != "system"
        )

        return system_instruction, user_content

    def get_response(
        self,
        messages: list,
        max_tokens: int = 10,
        temperature: float = 0,
    ) -> str:

        system_instruction, user_content = self._split_messages(messages)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": user_content
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {
                    "thinkingBudget": 0
                }
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [
                    {
                        "text": system_instruction
                    }
                ]
            }

        response = requests.post(
            self.endpoint,
            params={
                "key": self.api_key
            },
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        candidate = data["candidates"][0]

        if (
                "content" in candidate
                and "parts" in candidate["content"]
                and len(candidate["content"]["parts"]) > 0
        ):
            return (
                candidate["content"]["parts"][0]
                .get("text", "")
                .strip()
                .lower()
            )

        print("=" * 80)
        print(data)
        print("=" * 80)

        return ""

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

                os.environ.setdefault(
                    key.strip(),
                    value.strip(),
                )


if __name__ == "__main__":

    input_path = os.path.join(CUR_DIR, "data", "call_data_rw.csv")
    result_column = "Gemini Tuned Result"

    requests_per_minute = 800

    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    df = pd.read_csv(input_path)

    if result_column not in df.columns:
        df[result_column] = ""

    df[result_column] = df[result_column].fillna("").astype(str)

    client = TunedGeminiClient()

    rows = (
        df.iloc[start:]
        if limit is None
        else df.iloc[start:start + limit]
    )

    window_started_at = time.monotonic()
    requests_in_window = 0

    for index, row in rows.iterrows():

        if (
            pd.notna(row.get(result_column))
            and str(row.get(result_column)).strip()
        ):
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
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"Conversation:\n{row['Script']}",
            }
        ]

        try:
            result = client.get_response(messages)

        except Exception as e:
            print(f"retrying after error: {e}")

            time.sleep(2)

            result = client.get_response(messages)

        requests_in_window += 1

        df.loc[index, result_column] = result

        df.to_csv(input_path, index=False)

        print(
            f"id={row['ID']} "
            f"expected={row['Result']} "
            f"gemini={result}"
        )