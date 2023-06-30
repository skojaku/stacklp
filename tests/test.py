# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-25 06:39:44
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-25 06:45:45
import unittest
import networkx as nx
import numpy as np
import itertools
from scipy import sparse
from stacklp import StackingLinkPredictionModel


class TestOptimalStacking(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = sparse.csr_matrix(nx.adjacency_matrix(self.G))

    def test_model(self):
        model = StackingLinkPredictionModel(
            val_edge_frac=0.2, negative_edge_sampler="uniform"
        )

        model.fit(self.A)
        inds = np.array(
            list(
                itertools.product(
                    np.arange(self.A.shape[0]), np.arange(self.A.shape[0])
                )
            )
        )
        s = model.predict(self.A, inds[:, 0], inds[:, 1])
        model.get_feature_importance()
