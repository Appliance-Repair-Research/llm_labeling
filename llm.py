import os
import sys
import time

import pandas as pd
from openai import OpenAI

from prompt import prompt

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


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "meta-llama/llama-3.1-8b-instruct",
    ):
        if api_key is None:
            load_config_env(os.path.join(CUR_DIR, "config.env"))

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

        self.model = model

    def get_response(
        self,
        messages: list,
        max_tokens: int = 10,
        temperature: float = 0,
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content.strip().lower()


if __name__ == "__main__":

    input_path = os.path.join(CUR_DIR, "data", "call_data_rw.csv")
    result_column = "LLAMA Result"

    requests_per_minute = 1000

    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    df = pd.read_csv(input_path)

    if result_column not in df.columns:
        df[result_column] = ""

    df[result_column] = df[result_column].fillna("").astype(str)

    client = OpenRouterClient(
        model="meta-llama/llama-3.3-70b-instruct"
        # model="meta-llama/llama-3.1-8b-instruct"
    )

    rows = df.iloc[start:] if limit is None else df.iloc[start:start + limit]

    window_started_at = time.monotonic()
    requests_in_window = 0

    for index, row in rows.iterrows():

        if str(row.get(result_column, "")).strip():
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
            result = client.get_response(
                messages,
                max_tokens=10,
                temperature=0,
            )
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

            try:
                result = client.get_response(
                    messages,
                    max_tokens=10,
                    temperature=0,
                )
            except Exception as e:
                print(f"Retry failed: {e}")
                continue

        requests_in_window += 1

        df.loc[index, result_column] = result

        df.to_csv(input_path, index=False)

        print(
            f"id={row['ID']} "
            f"expected={row['Result']} "
            f"openrouter={result}"
        )