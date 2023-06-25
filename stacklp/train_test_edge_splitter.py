# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-24 16:19:13
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-24 16:19:33
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from scipy.sparse import csgraph


class TrainTestEdgeSplitter:
    def __init__(self, fraction=0.5):
        """Only support undirected Network.

        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """
        self.fraction = fraction

    def fit(self, A):
        """Split train and test edges with MST.

        Train network should have a one weakly connected component.
        """
        r, c, _ = sparse.find(A)
        edges = np.unique(pairing(r, c))

        MST = csgraph.minimum_spanning_tree(A + A.T)
        r, c, _ = sparse.find(MST)
        mst_edges = np.unique(pairing(r, c))
        remained_edge_set = np.array(
            list(set(list(edges)).difference(set(list(mst_edges))))
        )
        n_edge_removal = int(len(edges) * self.fraction)
        if len(remained_edge_set) < n_edge_removal:
            raise Exception(
                "Cannot remove edges by keeping the connectedness. Decrease the `fraction` parameter"
            )

        test_edge_set = np.random.choice(
            remained_edge_set, n_edge_removal, replace=False
        )

        train_edge_set = np.array(
            list(set(list(edges)).difference(set(list(test_edge_set))))
        )

        self.test_edges_ = depairing(test_edge_set)
        self.train_edges_ = depairing(train_edge_set)
        self.n = A.shape[0]

    def transform(self):
        return self.train_edges_, self.test_edges_


def pairing(r, c):
    return np.minimum(r, c) + 1j * np.maximum(r, c)


def depairing(v):
    return np.real(v).astype(int), np.imag(v).astype(int)
