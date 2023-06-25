# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-27 16:40:11
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-25 06:44:57
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from scipy.sparse import csgraph

from .negative_edge_sampler import NegativeEdgeSampler
from .train_test_edge_splitter import TrainTestEdgeSplitter


class LinkPredictionDataset:
    """
    Generates a link prediction dataset for evaluating link prediction models.

    :param testEdgeFraction: Fraction of edges to be removed from the given network for testing.
    :type testEdgeFraction: float
    :param negative_edge_sampler: Type of negative edge sampler. Can be "uniform" for conventional link prediction evaluation or "degreeBiased" for degree-biased sampling.
    :type negative_edge_sampler: str
    :param negatives_per_positive: Number of negative edges to sample per positive edge. Defaults to 1.
    :type negatives_per_positive: int, optional

    Example usage:
    >> model = LinkPredictionDataset(testEdgeFraction=0.5, negative_edge_sampler="degreeBiased")
    >> model.fit(net)
    >> train_net, target_edge_table = model.transform()
    """

    def __init__(
        self,
        testEdgeFraction,
        negative_edge_sampler,
        negatives_per_positive=1,
        all_negatives=False,
    ):
        """
        Initializer

        :param testEdgeFraction: Fraction of edges to be removed from the given network for testing.
        :type testEdgeFraction: float
        :param negative_edge_sampler: Type of negative edge sampler. Can be "uniform" for conventional link prediction evaluation or "degreeBiased" for degree-biased sampling.
        :type negative_edge_sampler: str
        :param negatives_per_positive: Number of negative edges to sample per positive edge. Defaults to 1.
        :type negatives_per_positive: int, optional
        """
        self.sampler = NegativeEdgeSampler(
            negative_edge_sampler=negative_edge_sampler,
        )
        self.splitter = TrainTestEdgeSplitter(fraction=testEdgeFraction)
        self.testEdgeFraction = testEdgeFraction
        self.negatives_per_positive = negatives_per_positive
        self.all_negatives = all_negatives

    def fit(self, net):
        self.n_nodes = net.shape[0]

        # Train-test edge split
        self.splitter.fit(net)

        # Sampling negative edges
        self.sampler.fit(net)

        self.net = net

    def transform(self):
        test_src, test_trg = self.splitter.test_edges_
        train_src, train_trg = self.splitter.train_edges_

        # Ensure that the network is undirected and unweighted
        self.train_net = sparse.csr_matrix(
            (np.ones_like(train_src), (train_src, train_trg)),
            shape=(self.n_nodes, self.n_nodes),
        )
        self.train_net = sparse.csr_matrix(self.train_net + self.train_net.T)
        self.train_net.data = self.train_net.data * 0 + 1

        if self.all_negatives:
            # We evaluate the all positives and all negatives
            neg_src, neg_trg = np.triu_indices(self.n_nodes, k=1)
            y = np.array(self.net[(neg_src, neg_trg)]).reshape(-1)
            s = y == 0
            neg_src, neg_trg, y = neg_src[s], neg_trg[s], y[s]

            self.target_edge_table = pd.DataFrame(
                {
                    "src": np.concatenate([test_src, neg_src]),
                    "trg": np.concatenate([test_trg, neg_trg]),
                    "isPositiveEdge": np.concatenate(
                        [np.ones_like(test_src), np.zeros_like(neg_trg)]
                    ),
                }
            )
            return self.train_net, self.target_edge_table

        n_test_edges = np.int(len(test_src))
        neg_src, neg_trg = [], []
        for _ in range(self.negatives_per_positive):
            _neg_src, _neg_trg = self.sampler.sampling(size=n_test_edges)
            neg_src.append(_neg_src)
            neg_trg.append(_neg_trg)
        neg_src, neg_trg = np.concatenate(neg_src), np.concatenate(neg_trg)

        self.target_edge_table = pd.DataFrame(
            {
                "src": np.concatenate([test_src, neg_src]),
                "trg": np.concatenate([test_trg, neg_trg]),
                "isPositiveEdge": np.concatenate(
                    [np.ones_like(test_src), np.zeros_like(neg_trg)]
                ),
            }
        )
        return self.train_net, self.target_edge_table
