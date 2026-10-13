
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report
)

DATA_DIR = Path("data") / "elliptic_bitcoin_dataset"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
classes = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv")
graph_features = pd.read_csv(OUTPUT_DIR / "temporal_graph_features.csv")

features = features.rename(columns={0: "txId", 1: "time_step"})
features["txId"] = features["txId"].astype(str)
classes["txId"] = classes["txId"].astype(str)
graph_features["txId"] = graph_features["txId"].astype(str)

data = features.merge(classes, on="txId", how="inner")
data = data[data["class"].astype(str).isin(["1", "2"])].copy()
data["target"] = data["class"].astype(str).map({"1": 1, "2": 0})

graph_columns = [
    "in_degree",
    "out_degree",
    "total_degree",
    "pagerank"
]

data = data.merge(
    graph_features[["txId"] + graph_columns],
    on="txId",
    how="left",
    validate="one_to_one"
)

feature_columns = [
    column for column in features.columns
    if column not in ["txId", "time_step"]
]

for column in feature_columns + graph_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data[feature_columns + graph_columns] = (
    data[feature_columns + graph_columns].fillna(0)
)

steps = sorted(data["time_step"].unique())
cutoff_index = int(len(steps) * 0.8)
cutoff = steps[cutoff_index]

train = data[data["time_step"] < cutoff].copy()
test = data[data["time_step"] >= cutoff].copy()

print("Training time steps:", train["time_step"].min(), "to", train["time_step"].max())
print("Testing time steps:", test["time_step"].min(), "to", test["time_step"].max())
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training class counts:")
print(train["target"].value_counts())
print("Testing class counts:")
print(test["target"].value_counts())

if train.empty or test.empty or train["target"].nunique() < 2 or test["target"].nunique() < 2:
    raise ValueError("The temporal split does not contain both classes in train and test.")

y_train = train["target"]
y_test = test["target"]

def evaluate_model(name, columns):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    )

    model.fit(train[columns], y_train)

    predictions = model.predict(test[columns])
    probabilities = model.predict_proba(test[columns])[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_test, probabilities)
    }

    print(f"\n--- {name} ---")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Licit", "Illicit"],
        zero_division=0
    ))

    return model, metrics

baseline_model, baseline_metrics = evaluate_model(
    "Temporal baseline",
    feature_columns
)

graph_model, graph_metrics = evaluate_model(
    "Temporal graph-enhanced",
    feature_columns + graph_columns
)

results = pd.DataFrame([baseline_metrics, graph_metrics])
results.to_csv(OUTPUT_DIR / "temporal_final_comparison.csv", index=False)

print("\n--- FINAL COMPARISON ---")
print(results.to_string(index=False))

print("\n--- GRAPH FEATURE IMPORTANCE ---")
importance = pd.Series(
    graph_model.feature_importances_,
    index=feature_columns + graph_columns
)

print(importance[graph_columns].sort_values(ascending=False).to_string())

print("\nSaved to:", (OUTPUT_DIR / "temporal_final_comparison.csv").resolve())
