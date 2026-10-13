
from pathlib import Path

import pandas as pd
import networkx as nx

DATA_DIR = Path("data") / "elliptic_bitcoin_dataset"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
edges = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")

features = features.rename(columns={0: "txId", 1: "time_step"})
features["txId"] = features["txId"].astype(str)
edges["txId1"] = edges["txId1"].astype(str)
edges["txId2"] = edges["txId2"].astype(str)

time_lookup = features.set_index("txId")["time_step"].to_dict()
all_nodes = set(features["txId"])

edges["source_time"] = edges["txId1"].map(time_lookup)
edges["target_time"] = edges["txId2"].map(time_lookup)

if edges[["source_time", "target_time"]].isna().any().any():
    raise ValueError("Some edge endpoints are missing from the feature file.")

records = []

for step in sorted(features["time_step"].unique()):
    available_nodes = set(
        features.loc[features["time_step"] <= step, "txId"]
    )

    eligible_edges = edges[
        (edges["source_time"] <= step)
        & (edges["target_time"] <= step)
    ]

    graph = nx.DiGraph()
    graph.add_nodes_from(available_nodes)
    graph.add_edges_from(
        eligible_edges[["txId1", "txId2"]].itertuples(
            index=False, name=None
        )
    )

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    pagerank = nx.pagerank(graph, alpha=0.85, max_iter=200)

    for tx_id in features.loc[
        features["time_step"] == step, "txId"
    ]:
        records.append({
            "txId": tx_id,
            "in_degree": in_degree.get(tx_id, 0),
            "out_degree": out_degree.get(tx_id, 0),
            "total_degree": (
                in_degree.get(tx_id, 0) + out_degree.get(tx_id, 0)
            ),
            "pagerank": pagerank.get(tx_id, 0.0)
        })

    print(
        f"Processed time step {step}: "
        f"{len(available_nodes)} available nodes, "
        f"{graph.number_of_edges()} available edges"
    )

graph_features = pd.DataFrame(records)
output_path = OUTPUT_DIR / "temporal_graph_features.csv"
graph_features.to_csv(output_path, index=False)

print("\nTemporal graph feature generation complete.")
print("Rows:", len(graph_features))
print("Columns:", graph_features.columns.tolist())
print("Saved to:", output_path.resolve())
print(graph_features.head().to_string(index=False))
