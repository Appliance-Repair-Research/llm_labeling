import os
import sys
import time
import pandas as pd
from prompt import prompt
from openai_utils import GPTClient

CUR_DIR = os.path.dirname(__file__)

if __name__ == "__main__":
    input_path = os.path.join(CUR_DIR, "data", "call_data_rw.csv")
    result_column = "GPT Result"

    # Rate limit: 1000 requests per minute as a safe default
    requests_per_minute = 1000

    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    df = pd.read_csv(input_path)

    if result_column not in df.columns:
        df[result_column] = ""

    df[result_column] = df[result_column].fillna("").astype(str)

    client = GPTClient()

    rows = df.iloc[start:] if limit is None else df.iloc[start:start + limit]

    window_started_at = time.monotonic()
    requests_in_window = 0

    for index, row in rows.iterrows():
        # Skip if we already have a result
        if str(row.get(result_column, "")).strip():
            continue

        # Simple rate limiting
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
                "content": "You are a classification assistant for appliance repair company. "
                           "Almost all calls with potential customers starts with asking customer their zip code,"
                           " then asking if appliance is under warranty (we don't do warranty service, "
                           "then asking about appliance type, then asking about appliance issue, these questions"
                           "should be answered directly or indirectly during the call to identify potential customer ",
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
            print(f"Error for ID {row['ID']}: {e}")
            time.sleep(2)
            try:
                # Retry once
                result = client.get_response(
                    messages,
                    max_tokens=10,
                    temperature=0,
                )
            except Exception as e:
                print(f"Retry failed for ID {row['ID']}: {e}")
                continue

        requests_in_window += 1

        df.loc[index, result_column] = result

        # Save after each row to avoid losing progress
        df.to_csv(input_path, index=False)

        print(
            f"id={row['ID']} "
            f"expected={row['Result']} "
            f"gpt={result}"
        )
