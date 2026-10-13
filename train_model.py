
from pathlib import Path

import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data") / "elliptic_bitcoin_dataset"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

features_path = DATA_DIR / "elliptic_txs_features.csv"
classes_path = DATA_DIR / "elliptic_txs_classes.csv"
graph_path = OUTPUT_DIR / "graph_features.csv"

features = pd.read_csv(features_path, header=None)
classes = pd.read_csv(classes_path)
graph_features = pd.read_csv(graph_path)

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

X_base = data[feature_columns].apply(pd.to_numeric, errors="coerce")
X_graph = data[feature_columns + graph_columns].apply(
    pd.to_numeric, errors="coerce"
)
y = data["target"]
time_steps = data["time_step"]

train_idx, test_idx = train_test_split(
    data.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_base_train = X_base.loc[train_idx]
X_base_test = X_base.loc[test_idx]
X_graph_train = X_graph.loc[train_idx]
X_graph_test = X_graph.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

def evaluate_model(name, X_train, X_test):
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

    results = {
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

    print(f"\n--- {name} ---")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Licit", "Illicit"],
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    return model, results

baseline_model, baseline_results = evaluate_model(
    "Baseline",
    X_base_train,
    X_base_test
)

graph_model, graph_results = evaluate_model(
    "Graph-enhanced",
    X_graph_train,
    X_graph_test
)

results = pd.DataFrame([baseline_results, graph_results])
results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\n--- MODEL COMPARISON ---")
print(results.to_string(index=False))

results.set_index("model")[["precision", "recall", "f1", "pr_auc"]].T.plot(
    kind="bar",
    figsize=(10, 6)
)
plt.title("Baseline vs Graph-Enhanced LightGBM")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
plt.show()

importance = pd.Series(
    graph_model.feature_importances_,
    index=X_graph.columns
).sort_values(ascending=False)

print("\n--- TOP 15 FEATURES ---")
print(importance.head(15).to_string())

importance.head(15).sort_values().plot(kind="barh", figsize=(9, 6))
plt.title("Top 15 Features — Graph-Enhanced Model")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
plt.show()

print("\nResults saved in:", OUTPUT_DIR.resolve())
