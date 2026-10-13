
from pathlib import Path
import pandas as pd
import networkx as nx

DATA_DIR = Path("data") / "elliptic_bitcoin_dataset"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

features_path = DATA_DIR / "elliptic_txs_features.csv"
edges_path = DATA_DIR / "elliptic_txs_edgelist.csv"

features = pd.read_csv(features_path, header=None)
edges = pd.read_csv(edges_path)

features = features.rename(columns={0: "txId", 1: "time_step"})
features["txId"] = features["txId"].astype(str)
edges["txId1"] = edges["txId1"].astype(str)
edges["txId2"] = edges["txId2"].astype(str)

graph = nx.DiGraph()

graph.add_nodes_from(features["txId"].unique())
graph.add_edges_from(
    edges[["txId1", "txId2"]].itertuples(index=False, name=None)
)

print("Graph created successfully")
print("Number of nodes:", graph.number_of_nodes())
print("Number of edges:", graph.number_of_edges())
print("Is directed:", graph.is_directed())

in_degree = dict(graph.in_degree())
out_degree = dict(graph.out_degree())
pagerank = nx.pagerank(graph, alpha=0.85, max_iter=200)

weak_components = {}
for component_id, component in enumerate(nx.weakly_connected_components(graph)):
    for node in component:
        weak_components[node] = component_id

graph_features = pd.DataFrame({
    "txId": list(graph.nodes()),
    "in_degree": [in_degree[node] for node in graph.nodes()],
    "out_degree": [out_degree[node] for node in graph.nodes()],
    "pagerank": [pagerank[node] for node in graph.nodes()],
    "weak_component": [weak_components[node] for node in graph.nodes()]
})

graph_features["total_degree"] = (
    graph_features["in_degree"] + graph_features["out_degree"]
)

output_path = OUTPUT_DIR / "graph_features.csv"
graph_features.to_csv(output_path, index=False)

print("\n--- GRAPH SUMMARY ---")
print("Weakly connected components:", nx.number_weakly_connected_components(graph))
print("Isolated nodes:", len(list(nx.isolates(graph))))
print("Average in-degree:", graph_features["in_degree"].mean())
print("Average out-degree:", graph_features["out_degree"].mean())
print("Maximum total degree:", graph_features["total_degree"].max())
print("Graph features saved to:", output_path.resolve())
print("\nFirst five rows:")
print(graph_features.head().to_string(index=False))
