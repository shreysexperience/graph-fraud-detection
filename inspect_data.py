
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data") / "elliptic_bitcoin_dataset"

features_path = DATA_DIR / "elliptic_txs_features.csv"
classes_path = DATA_DIR / "elliptic_txs_classes.csv"
edges_path = DATA_DIR / "elliptic_txs_edgelist.csv"

for path in [features_path, classes_path, edges_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")

features = pd.read_csv(features_path, header=None)
classes = pd.read_csv(classes_path)
edges = pd.read_csv(edges_path)

print("Original feature shape:", features.shape)

if features.shape[1] < 3:
    raise ValueError("Feature file has fewer than 3 columns.")

features.columns = (
    ["txId", "time_step"]
    + [f"feature_{i}" for i in range(1, features.shape[1] - 1)]
)

print("\n--- DATASET SHAPES ---")
print("Features:", features.shape)
print("Classes:", classes.shape)
print("Edges:", edges.shape)

print("\n--- FEATURE SAMPLE ---")
print(features.iloc[:5, :7].to_string(index=False))

print("\n--- CLASS COUNTS ---")
print(classes["class"].value_counts(dropna=False))

print("\n--- TIME STEPS ---")
print("Minimum:", features["time_step"].min())
print("Maximum:", features["time_step"].max())
print("Unique steps:", features["time_step"].nunique())

print("\n--- MISSING VALUES ---")
print("Feature cells missing:", int(features.isna().sum().sum()))
print("Class cells missing:", int(classes.isna().sum().sum()))
print("Edge cells missing:", int(edges.isna().sum().sum()))

print("\n--- GRAPH CONNECTIVITY INPUT ---")
print("Unique feature transaction IDs:", features["txId"].nunique())

if {"txId1", "txId2"}.issubset(edges.columns):
    print("Unique edge source IDs:", edges["txId1"].nunique())
    print("Unique edge destination IDs:", edges["txId2"].nunique())
else:
    print("Actual edge columns:", edges.columns.tolist())
    print(edges.head().to_string(index=False))

print("\n--- CHECK COMPLETE ---")