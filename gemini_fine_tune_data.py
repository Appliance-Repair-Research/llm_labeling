import pandas as pd
import json
from prompt import prompt
import os

CUR_DIR = os.path.dirname(__file__)

df = pd.read_csv(
    os.path.join(CUR_DIR, "data", "call_data_rw.csv")
)

output_file = os.path.join(
    CUR_DIR,
    "data",
    "train.jsonl"
)

train_data = (
    df[df["split"] == "train"]
    .dropna(subset=["Script", "Result"])
    .copy()
)

with open(output_file, "w", encoding="utf-8") as f:
    for _, row in train_data.iterrows():

        record = {
            "systemInstruction": {
                "role": "system",
                "parts": [
                    {
                        "text": prompt.strip()
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"Conversation:\n{str(row['Script']).strip()}"
                        }
                    ]
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": str(row["Result"]).strip().lower()
                        }
                    ]
                }
            ]
        }

        f.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )

print(f"Saved {len(train_data):,} examples to {output_file}")