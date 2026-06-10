
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os


SEED = 42
np.random.seed(SEED)


CUR_DIR = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(CUR_DIR, "data", "call_data_rw.csv"))

TEXT_COL = "Script"
TARGET_COL = "Result"


df = df.dropna(subset=[TEXT_COL, TARGET_COL])

df[TEXT_COL] = df[TEXT_COL].astype(str).str.lower()

# df[TEXT_COL] = df[TEXT_COL].str.replace("agent:", "agent_says ", regex=False)
# df[TEXT_COL] = df[TEXT_COL].str.replace("customer:", "customer_says ", regex=False)

df[TEXT_COL] = df[TEXT_COL].str.replace(r"[^a-z0-9\s_]", " ", regex=True)
df[TEXT_COL] = df[TEXT_COL].str.replace(r"\s+", " ", regex=True).str.strip()
# labels a números
labels = df[TARGET_COL].astype("category")
df["label"] = labels.cat.codes

label_map = dict(enumerate(labels.cat.categories))

print(label_map)

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=SEED,
    stratify=df["label"]
)


df["split"] = None
df.loc[df["ID"].isin(train_df["ID"]), "split"] = "train"
df.loc[df["ID"].isin(test_df["ID"]), "split"] = "test"


df.to_csv(os.path.join(CUR_DIR, "data", "call_data_rw.csv"), index=False)
