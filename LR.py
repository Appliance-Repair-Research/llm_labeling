import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# =========================
# 1. LOAD DATA
# =========================

SEED = 42

CUR_DIR = os.path.dirname(__file__)

TEXT_COL = "Script"
TARGET_COL = "Result"

df = pd.read_csv(os.path.join(CUR_DIR, "data", "call_data_rw.csv"))

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

print(df.columns)


# =========================
# 2. TRAIN / TEST
# =========================

X_train = train_df[TEXT_COL].tolist()
y_train = train_df["label"].tolist()

X_test = test_df[TEXT_COL].tolist()
y_test = test_df["label"].tolist()


# =========================
# 3. MODEL
# =========================

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
])


# =========================
# 4. CROSS VALIDATION
# =========================

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print(f"CV Accuracy Mean: {scores.mean():.4f}")
print("CV Scores:", scores)


# =========================
# 5. TRAIN
# =========================

model.fit(X_train, y_train)


# =========================
# 6. SAMPLE PREDICTIONS
# =========================

y_pred_sample = model.predict(X_test[:5])

print("Predictions:", y_pred_sample)
print("Actual:", y_test[:5])


# =========================
# 7. FULL TEST EVALUATION
# =========================

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(
    pd.DataFrame(
        cm,
        index=[f"Actual_{i}" for i in sorted(set(y_test))],
        columns=[f"Pred_{i}" for i in sorted(set(y_test))]
    )
)

# =========================
# 8. SAVE PREDICTIONS
# =========================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

label_map = (
    df[["label", "Result"]]
    .drop_duplicates()
    .set_index("label")["Result"]
    .to_dict()
)

pred_df = pd.concat([
    pd.DataFrame({
        "ID": train_df["ID"],
        "LR Result": train_pred
    }),
    pd.DataFrame({
        "ID": test_df["ID"],
        "LR Result": test_pred
    })
])

pred_df["LR Result"] = pred_df["LR Result"].map(label_map)

df = df.merge(pred_df, on="ID", how="left")

df.to_csv(
    os.path.join(CUR_DIR, "data", "call_data_rw.csv"),
    index=False
)

print(f"Saved {len(pred_df)} predictions.")