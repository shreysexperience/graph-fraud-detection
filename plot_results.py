
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

output_dir = Path("outputs")
results_path = output_dir / "temporal_final_comparison.csv"

results = pd.read_csv(results_path).set_index("model")

metrics = ["precision", "recall", "f1", "pr_auc"]
labels = ["Precision", "Recall", "F1-score", "PR-AUC"]

baseline = results.loc["Temporal baseline", metrics]
graph = results.loc["Temporal graph-enhanced", metrics]

x = range(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    [i - width / 2 for i in x],
    baseline.values,
    width,
    label="Baseline"
)

ax.bar(
    [i + width / 2 for i in x],
    graph.values,
    width,
    label="Graph-enhanced"
)

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("Baseline vs Graph-Enhanced LightGBM")
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(output_dir / "temporal_model_comparison.png", dpi=180)
plt.show()

print("Chart saved to outputs/temporal_model_comparison.png")
