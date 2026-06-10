import json
import os

import pandas as pd

from google_utils import GeminiClient
from prompt import prompt


INPUT_PATH = "data/call_data_rw.csv"
BATCH_STATE_PATH = "data/call_data_gemini_batches.json"
RESULT_COLUMN = "Gemini Result"
MAX_BATCH_ROWS = 100
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def has_result(value) -> bool:
    return pd.notna(value) and bool(str(value).strip())


def load_state() -> dict:
    if os.path.exists(BATCH_STATE_PATH):
        with open(BATCH_STATE_PATH, "r") as f:
            return json.load(f)
    return {"batches": []}


def save_state(state: dict):
    with open(BATCH_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def build_messages(transcript: str) -> list:
    return [
        {
            "role": "system",
            "content": "You are a classification assistant for appliance repair company.",
        },
        {
            "role": "user",
            "content": f"{prompt}Conversation:\n{transcript}",
        },
    ]


def extract_text(result: dict) -> str:
    response = result.get("response", result)
    candidates = response.get("candidates") or []
    if not candidates:
        return ""

    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts).strip().lower()


def submit_batch(input_path: str) -> dict | None:
    df = pd.read_csv(input_path)
    if RESULT_COLUMN not in df.columns:
        df[RESULT_COLUMN] = ""

    requests = []
    for _, row in df.iterrows():
        if len(requests) >= MAX_BATCH_ROWS:
            break
        if has_result(row.get(RESULT_COLUMN, "")):
            continue

        script = row.get("Script")
        if pd.isna(script) or not str(script).strip():
            continue

        requests.append({
            "custom_id": str(row["ID"]),
            "messages": build_messages(str(script)),
            "max_tokens": 10,
            "temperature": 0,
        })

    if not requests:
        print("all rows are done")
        return None

    batch = GeminiClient().create_batch(
        requests,
        metadata={"source": os.path.basename(input_path)},
        display_name="call-labeling-gemini",
    )
    df.to_csv(input_path, index=False)
    print(f"batch_name={batch.name}")
    print(f"status={batch.state.name if batch.state else 'unknown'}")
    print(f"requests={len(requests)}")
    return {
        "batch_name": batch.name,
        "status": batch.state.name if batch.state else "unknown",
        "requests": len(requests),
        "processed": False,
    }


def handle_batch(batch_record: dict, input_path: str) -> bool:
    client = GeminiClient()
    batch = client.get_batch_status(batch_record["batch_name"])
    status = batch.state.name if batch.state else "unknown"
    batch_record["status"] = status
    print(f"status={status}")

    if status not in TERMINAL_STATES:
        return False
    if status in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
        if batch.error:
            print(batch.error)
        batch_record["processed"] = True
        return True

    df = pd.read_csv(input_path)
    if RESULT_COLUMN not in df.columns:
        df[RESULT_COLUMN] = ""

    df["ID"] = df["ID"].astype(str)
    id_to_index = df.reset_index().set_index("ID")["index"]

    updated = 0
    for result in client.get_batch_results(batch_record["batch_name"]):
        custom_id = str(result.get("custom_id") or result.get("key") or "")
        if not custom_id or custom_id not in id_to_index.index:
            continue
        if result.get("error"):
            continue

        label = extract_text(result)
        if not label:
            continue

        df.loc[int(id_to_index.loc[custom_id]), RESULT_COLUMN] = label
        updated += 1

    df.to_csv(input_path, index=False)
    batch_record["processed"] = True
    print(f"updated={updated}")
    print(f"output={input_path}")
    return True


def run_next_step() -> bool:
    state = load_state()

    active_batch = next(
        (batch for batch in state["batches"] if not batch.get("processed")),
        None,
    )
    if active_batch:
        completed = handle_batch(active_batch, INPUT_PATH)
        save_state(state)
        if not completed:
            return False

    next_batch = submit_batch(INPUT_PATH)
    if next_batch:
        state["batches"].append(next_batch)
        save_state(state)
        return False

    return True


if __name__ == "__main__":
    # client = GeminiClient()
    #
    # batch_name = "batches/qw1pnnx6zj9tst7my20r82j2d1saehew14ay"
    #
    # client.client.batches.cancel(name=batch_name)
    #
    # batch = client.client.batches.get(name=batch_name)
    # print(batch.state)
    run_next_step()
