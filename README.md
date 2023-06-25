# stacklp
Stacking model for link prediction for networks. 


# stacklp

A Python implementation of the stacking model for link prediction proposed in [the following paper](https://www.pnas.org/doi/abs/10.1073/pnas.1914950117):
```
Ghasemian, Amir, et al. "Stacking models for nearly optimal link prediction in complex networks." Proceedings of the National Academy of Sciences 117.38 (2020): 23393-23400.
```

There is also [the original implementation by the authors](https://github.com/Aghasemian/OptimalLinkPrediction), and the implementation in this repository has several differences below:

1. Some redundant variables are dropped for the sake of speed. These variables are:
  - Load centrality, `load_cents` (due to a highly correlation to the betweenness centrality)
  - Variables based on the full SVD decomposition, `svd_edges`, `svd_edges_dot`,`svd_edges_mean` (since the full SVD decomposition recovers the given adjacency matrix!!).
2. Train/Test data split.
  - In the initial implementation by Aghasemian, the feature matrix, denoted as $X$, was computed using a provided network, without any held-out edges. Subsequently, this matrix was divided into separate train and test feature matrices to facilitate model selection. However, a potential concern with this approach is that the feature matrix $X$ is calculated using all edges in the given network, which means that the train features are based on the ground-truth links that are used for evaluating the mdoel. In other words, information about the ground-truth can *leak* to the train set.
  - To prevent this, in this implementation, the given network is split into test and train edges. Then, the feature matrix $X$ is computed based on the train edges. This way, the model only learns the given train edges, and is evaluated based on the unseen test edges.

# Usage

```python
import stacklp
import networkx as nx
import numpy as np

# Load network
A = sparse.csr_matrix(nx.adjacency_matrix(G = nx.karate_club_graph()))

# Create/Fit the model
model = StackingLinkPredictionModel()
model.fit(A)

# Prediction
src_nodes = np.array([0, 1, 5, 9])
trg_nodes = np.array([33, 32, 31, 20])
prob = model.predict(A, src_nodes, trg_nodes)

# Get feature importance
model.get_feature_importance()
```

The `.fit` function performs the model selection based on the cross validation. You can change the fraction of test edges and the number of validations. See [here](./stacklp/stacking_model.py) for the arguments of `StackingLinkPredictionModel`

