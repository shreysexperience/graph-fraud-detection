
# Graph-Based Fraud Detection

## Overview

This project investigates whether transaction graph features improve illicit transaction detection compared with transaction-level features alone.

It uses the Elliptic Bitcoin transaction dataset, NetworkX for graph analysis, and LightGBM for supervised classification.

## Objectives

- Construct a directed transaction graph.
- Calculate degree-based features and PageRank.
- Compare a baseline LightGBM model with a graph-enhanced model.
- Evaluate performance on later transaction time steps.
- Analyze precision, recall, F1-score, and PR-AUC.

## Tech Stack

- Python
- Pandas and NumPy
- NetworkX
- LightGBM
- Scikit-learn
- Matplotlib

## Project Structure

```text
graph-fraud-detection/
├── data/
│   └── elliptic_bitcoin_dataset/
├── outputs/
├── inspect_data.py
├── build_graph.py
├── build_temporal_graph.py
├── train_model.py
├── train_temporal.py
├── train_temporal_final.py
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data preparation

Load transaction attributes, class labels, and transaction relationships. Exclude unlabeled transactions from supervised training.

### 2. Graph construction

Use NetworkX to construct a directed graph from the supplied edge list.

### 3. Graph features

Calculate:

- In-degree
- Out-degree
- Total degree
- PageRank

### 4. Model comparison

Train two LightGBM classifiers using the same temporal training and test partitions:

- Baseline: original transaction attributes.
- Graph-enhanced: original attributes plus graph features.

### 5. Evaluation

Report accuracy, precision, recall, F1-score, and average precision (PR-AUC).


## Results

The baseline LightGBM model was compared with a graph-enhanced LightGBM model using a temporal train/test split on the labeled Elliptic Bitcoin transaction dataset.

| Metric | Temporal Baseline | Graph-Enhanced |
|---|---:|---:|
| Accuracy | 97.505% | 97.559% |
| Precision | 92.000% | 93.112% |
| Recall | 61.478% | 61.635% |
| F1-score | 73.704% | 74.172% |
| Average Precision (PR-AUC) | 78.699% | 77.926% |

### Interpretation

- The graph-enhanced model achieved a modest increase in precision and F1-score.
- Recall changed only slightly.
- Average precision decreased, indicating weaker overall ranking performance across classification thresholds.

These results do not establish that graph features consistently improve fraud detection.

## Evaluation Limitations

- Transaction relationships must be checked for availability at prediction time.
- Graph construction may introduce temporal information leakage if future relationships influence features.
- The experiment is a preliminary benchmark and requires further validation before production use.
- Results on the Elliptic Bitcoin dataset may not generalize to other fraud domains.


## Limitations

- Dataset labels are incomplete.
- Historical transaction relationships may not represent real-time availability.
- Temporal graph construction assumptions require validation.
- Results on Bitcoin transactions may not generalize to payment-card fraud.
- This is a research prototype, not a production fraud detection system.

## Reproducibility

Create a virtual environment, install the packages in `requirements.txt`, obtain the dataset from its original source, and run the scripts in the order documented in this repository.

## Author

Shreyansh
