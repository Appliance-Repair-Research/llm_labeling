import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report
from transformers import (
    Trainer,
    TrainingArguments,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CUR_DIR = os.path.dirname(__file__)

TEXT_COL = "Script"
TARGET_COL = "Result"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(os.path.join(CUR_DIR, "data", "call_data_rw.csv"))

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

label_map = (
    df[["label", "Result"]]
    .drop_duplicates()
    .set_index("label")["Result"]
    .to_dict()
)

train_texts = train_df[TEXT_COL].tolist()
train_labels = train_df["label"].tolist()

test_texts = test_df[TEXT_COL].tolist()
test_labels = test_df["label"].tolist()

# =========================
# TOKENIZATION
# =========================

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = Dataset(train_encodings, train_labels)
test_dataset = Dataset(test_encodings, test_labels)

# =========================
# MODEL
# =========================

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=df["label"].nunique()
)

training_args = TrainingArguments(
    output_dir="./results",
    seed=SEED,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# =========================
# TRAIN
# =========================

trainer.train()

# =========================
# TEST EVALUATION
# =========================

test_predictions = trainer.predict(test_dataset)
test_pred = np.argmax(test_predictions.predictions, axis=1)

print(classification_report(test_labels, test_pred))

# =========================
# TRAIN PREDICTIONS
# =========================

train_predictions = trainer.predict(train_dataset)
train_pred = np.argmax(train_predictions.predictions, axis=1)

# =========================
# SAVE PREDICTIONS
# =========================

pred_df = pd.concat([
    pd.DataFrame({
        "ID": train_df["ID"],
        "Roberta Result": train_pred
    }),
    pd.DataFrame({
        "ID": test_df["ID"],
        "Roberta Result": test_pred
    })
])

pred_df["Roberta Result"] = pred_df["Roberta Result"].map(label_map)

df = df.merge(pred_df, on="ID", how="left")

df.to_csv(
    os.path.join(CUR_DIR, "data", "call_data_rw.csv"),
    index=False
)

print(f"Saved {len(pred_df)} predictions.")