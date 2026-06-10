import json
import os

import pandas as pd

from openai_utils import GPTClient


INPUT_PATH = "data/call_data_rw.csv"
BATCH_STATE_PATH = "data/call_data_ai_batches.json"
BATCH_ID = "batch_69f28fd9891c81909070b1dcd1020d29"
MAX_BATCH_TOKENS = 800_000


def has_ai_result(value) -> bool:
    return pd.notna(value) and bool(str(value).strip())


def load_state() -> dict:
    if os.path.exists(BATCH_STATE_PATH):
        with open(BATCH_STATE_PATH, "r") as f:
            return json.load(f)
    state = {"batches": []}
    if BATCH_ID:
        state["batches"].append({"batch_id": BATCH_ID, "status": "unknown"})
    return state


def save_state(state: dict):
    with open(BATCH_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def build_messages(transcript: str) -> list:
    prompt = (
        "Classify the outcome of this conversation in one of these categories in lower case, your response should only"
        "include classification category, but nothing else:\n"
        "book, no book, wrong, misc, recall, business. When you analyze context of text, beware that text contain typos.\n\n"
        "Definitions of categories you should strictly use in your answer:\n"
        "- book: potential customer books appointment with us, appointment date and time is set and customer approves "
        "it verbally during the call. If an appointment is clearly scheduled and accepted, classify as book even if there "
        "are minor unresolved details, \n"
        "- no book: potential customer that doesn’t decide to book with us during the call mainly due to following reasons"
        " 1)we don't have available slot, 2)customer is willing to pay service call fee but finds it expensive, "
        "3)agent doesn't handle call in a good manner and customer decides not to book with us, 4) customer wants to call "
        "other companies before deciding 5) potential customer receives information and decides to call back later "
        "6)customer schedules appointment but changes their mind about repair after learning cost of the service" 
        "- wrong: not a potential customer, definition of not a potential customer is following:\n"
        "  1) customer is outside our coverage area\n"
        "  2) customer is looking for service we don't provide, such as but not limited to tv repair, air conditioner etc\n"
        "  3) appliance is under warranty even if customer is unsure, as long as agent says it is very likely or definitely in warranty\n"
        "  4) wants to bring appliance to store\n"
        "  5) customer clearly refuses any paid visit/diagnostic and only wants free phone troubleshooting help, classify as wrong.\n"
        "  6) wants phone support over the phone instead of paying for coming and checking appliance\n"
        "  7) Important:-agent explicitly says that customer has called the wrong number or we do not provide service or we don't provide phone support\n"
        "  8) or any other reason clearly not a potential customer. "
        "  9) caller is from another business trying to sell something to us\n"
        "  10) caller prefers to speak in language other than English\n"
        "  11) caller is not decision maker"
        "  12) customer ends up calling back again accidentally and informs that they have already talked."
        "- misc: cannot be categorized or conversation is too short/insufficient. If the call ends before enough information "
        "is collected to know whether the caller is a valid lead, classify as misc. This includes when customer abruptly ends the call before answering "
        "necessary questions to identify if they are potential customer unless customer hangs up because they are no longer interested in service\n\n"
        "Conversation:\n"
        f"{transcript}"
    )
    return [
        {"role": "system", "content": "You are a classification assistant for appliance repair company. "
                                      "Almost all calls with potential customers starts with asking customer their zip code,"
                                      " then asking if appliance is under warranty (we don't do warranty service, "
                                      "then asking about appliance type, then asking about appliance issue, these questions"
                                      "should be answered directly or indirectly during the call to identify potential customer "},
        {"role": "user", "content": prompt},
    ]


def submit_batch(input_path: str) -> dict | None:
    df = pd.read_csv(input_path)
    if "GPT Result" not in df.columns:
        df["GPT Result"] = ""

    requests = []
    estimated_tokens = 0
    for index, row in df.iterrows():
        if has_ai_result(row.get("GPT Result", "")):
            continue
        script = row.get("Script")
        if pd.isna(script) or not str(script).strip():
            continue

        request_tokens = sum(len(message["content"]) for message in build_messages(str(script))) / 4 + 10
        if requests and estimated_tokens + request_tokens > MAX_BATCH_TOKENS:
            break
        estimated_tokens += request_tokens

        custom_id = str(row["ID"])
        requests.append({
            "custom_id": custom_id,
            "messages": build_messages(str(script)),
            "max_tokens": 10,
            "temperature": 0,
        })

    if not requests:
        print("all rows are done")
        return None

    batch = GPTClient().create_batch(requests, metadata={"source": os.path.basename(input_path)})
    df.to_csv(input_path, index=False)
    print(f"batch_id={batch.id}")
    print(f"requests={len(requests)}")
    print(f"estimated_tokens={round(estimated_tokens)}")
    print(f"working_file={input_path}")
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "requests": len(requests),
        "estimated_tokens": round(estimated_tokens),
    }


def handle_batch(batch_record: dict, input_path: str) -> bool:
    gpt_client = GPTClient()
    batch_id = batch_record["batch_id"]
    batch = gpt_client.get_batch_status(batch_id)
    batch_record["status"] = batch.status
    print(f"status={batch.status}")
    if batch.status != "completed":
        if batch.status == "failed":
            print(batch.model_dump_json(indent=2))
        return False

    df = pd.read_csv(input_path)
    if "GPT Result" not in df.columns:
        df["GPT Result"] = ""

    df["ID"] = df["ID"].astype(str)
    id_to_index = df.reset_index().set_index("ID")["index"]
    results = gpt_client.get_batch_results(batch_id)

    updated = 0
    for result in results:
        custom_id = result["custom_id"]
        if custom_id not in id_to_index.index:
            continue
        if result.get("error"):
            continue
        ai_result = result["response"]["body"]["choices"][0]["message"]["content"].strip().lower()
        if not ai_result:
            continue
        df.loc[int(id_to_index.loc[custom_id]), "GPT Result"] = ai_result
        updated += 1

    df.to_csv(input_path, index=False)
    print(f"updated={updated}")
    print(f"output={input_path}")
    return True


def run_next_step() -> bool:
    state = load_state()

    active_batch = next(
        (batch for batch in state["batches"] if batch.get("status") not in {"completed", "failed", "cancelled", "expired"}),
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


import os



if __name__ == "__main__":
    run_next_step()
