
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
graph_features = pd.read_csv(OUTPUT_DIR / "graph_features.csv")

features = features.rename(columns={0: "txId", 1: "time_step"})
features["txId"] = features["txId"].astype(str)
classes["txId"] = classes["txId"].astype(str)
graph_features["txId"] = graph_features["txId"].astype(str)

data = features.merge(classes, on="txId", how="inner")
data = data[data["class"].isin(["1", "2", 1, 2])].copy()
data["target"] = data["class"].astype(int).map({1: 1, 2: 0})

graph_columns = [
    "in_degree",
    "out_degree",
    "total_degree",
    "pagerank"
]

data = data.merge(
    graph_features[["txId"] + graph_columns],
    on="txId",
    how="left"
)

data[graph_columns] = data[graph_columns].fillna(0)

feature_columns = [
    column for column in features.columns
    if column not in ["txId", "time_step"]
]

for column in feature_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data[feature_columns + graph_columns] = (
    data[feature_columns + graph_columns].fillna(0)
)

steps = sorted(data["time_step"].unique())

if len(steps) < 5:
    raise ValueError("Not enough time steps for temporal evaluation.")

split_position = int(len(steps) * 0.8)
train_steps = steps[:split_position]
test_steps = steps[split_position:]

train = data[data["time_step"].isin(train_steps)]
test = data[data["time_step"].isin(test_steps)]

print("Training time steps:", train_steps[0], "to", train_steps[-1])
print("Testing time steps:", test_steps[0], "to", test_steps[-1])
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training labels:")
print(train["target"].value_counts())
print("Testing labels:")
print(test["target"].value_counts())

if train["target"].nunique() < 2 or test["target"].nunique() < 2:
    raise ValueError(
        "Train or test contains only one class. "
        "Adjust the temporal cutoff after inspecting class counts."
    )

X_base_train = train[feature_columns]
X_base_test = test[feature_columns]

X_graph_train = train[feature_columns + graph_columns]
X_graph_test = test[feature_columns + graph_columns]

y_train = train["target"]
y_test = test["target"]

def evaluate(name, X_train, X_test):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_test, probabilities)
    }

    print(f"\n{name}")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Licit", "Illicit"],
        zero_division=0
    ))

    return model, metrics

baseline_model, baseline_metrics = evaluate(
    "Temporal baseline",
    X_base_train,
    X_base_test
)

graph_model, graph_metrics = evaluate(
    "Temporal graph-enhanced",
    X_graph_train,
    X_graph_test
)

results = pd.DataFrame([baseline_metrics, graph_metrics])
results.to_csv(OUTPUT_DIR / "temporal_comparison.csv", index=False)

print("\nTemporal comparison:")
print(results.to_string(index=False))
print("\nSaved to:", (OUTPUT_DIR / "temporal_comparison.csv").resolve())
