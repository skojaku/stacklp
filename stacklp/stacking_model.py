# /*
#  * @Author: Rachith Aiyappa
#  * @Date: 2023-04-03 09:43:13
#  * @Last Modified by:   Sadamori Kojaku
#  * @Last Modified time: 2023-06-30 16:25:46
#  */
# %%
from tqdm.auto import tqdm
from collections import OrderedDict
import igraph
import itertools
import pickle
import numpy as np
import pandas as pd
from scipy import sparse, stats
from numba import njit
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from .negative_edge_sampler import NegativeEdgeSampler
from .utils import pairing, depairing
from .link_prediction_dataset import LinkPredictionDataset


class StackingLinkPredictionModel:
    """
    A class that performs optimal stacking link prediction.

    Attributes
    ----------
    val_edge_frac : float
        The fraction of edges used for validation.
    params : dict
        Additional parameters passed to the predictive model.
    negative_edge_sampler : str
        Type of the negative sampler.
    pred_model : object or None
        An optional object representing the predictive model.

    Returns
    -------
    StackingLinkPredictionModel
    """

    def __init__(
        self,
        filename=None,
        val_edge_frac=0.2,
        negative_edge_sampler="uniform",
        n_train_samples=10000,
        n_cv=5,
        **params,
    ):
        """
        Initialize a StackingModel object.

        Parameters
        ----------
        filename : str or None, optional
            A string representing the path to a file containing the graph data.
        val_edge_frac : float, optional
            The fraction of edges used for validation. Default value is 0.2.
        negative_edge_sampler : str, optional
            Type of the negative sampler. Available negative samples are ["uniform", "degreeBiased"].
            "uniform" samples the unconnected node pairs uniformly at random and is a faithful implementation of the original stacking model.
            "degreeBiased" samples unconnected node pairs with probability proportional to the product of the degrees of the nodes.
        n_train_samples : int, optional (default=10000)
            The number of samples used for training.
        n_cv : int, optional (default=5)
            The number of cross-validation folds.
        **params : dict, optional
            Additional parameters passed to the predictive model.

        Attributes
        ----------
        pred_model : object or None
            An optional object representing the predictive model.
        var_name : object or None
            List of the names of predictive variables

        Returns
        -------
        None
        """
        if filename is not None:
            self.load(filename)
            return
        self.val_edge_frac = val_edge_frac
        self.params = params
        self.negative_edge_sampler = negative_edge_sampler
        self.pred_model = None
        self.var_names = None
        self.n_cv = n_cv
        self.n_train_samples = n_train_samples

    def fit(self, network, n_depths=None, n_ests=None):
        """Fits the model and searches over hyperparameters.

        Parameters
        ----------
        network : sparse matrix
            The adjacency matrix of the graph.
        n_depths : list of int, optional (default=None)
            The maximum depth of each decision tree in the random forest.
        n_ests : list of int, optional (default=None)
            The number of trees in the random forest.

        Returns
        -------
        OptimalStackingLinkPrediction
            The fitted `OptimalStackingLinkPrediction` instance.
        """
        # If n_depths is not passed, set a sample search space
        if n_depths is None:
            n_depths = [2, 4, 8]

        # If n_ests is not passed, set a sample search space
        if n_ests is None:
            n_ests = [25, 50, 100]

        datasets = self.generate_train_test_datasets(
            network, n_cv=self.n_cv, n_train_samples=self.n_train_samples
        )

        # Create an empty list to store the cross-validation results
        cv_results = []

        # Loop over all combinations of train-test indices, maximum depth (n_depth), and number of trees (n_est)
        for dataset, n_depth, n_est in tqdm(
            list(itertools.product(datasets, n_depths, n_ests))
        ):
            X_train, Y_train = dataset["train"]
            X_test, Y_test = dataset["test"]

            # Train a random forest classifier with the given parameters
            dtree_model = RandomForestClassifier(
                max_depth=n_depth, n_estimators=n_est
            ).fit(X_train, Y_train)

            # Predict on the test set using the trained model
            dtree_predictions = dtree_model.predict(X_test)

            # Compute the F-measure for each class (positive and negative) in Y_test and dtree_predictions
            f_measure_aux = precision_recall_fscore_support(
                Y_test, dtree_predictions, average=None, zero_division=True
            )[2]

            # Append the results to the cv_results list
            cv_results.append(
                {
                    "n_depth": n_depth,
                    "n_est": n_est,
                    "f_measure": np.mean(f_measure_aux),
                }
            )

        # Create a pandas DataFrame from the cv_results list, group by n_depth and n_est, compute the mean, sort by f_measure, and get the best combination
        df = (
            pd.DataFrame(cv_results)
            .groupby(["n_depth", "n_est"])
            .mean()
            .reset_index()
            .sort_values(by="f_measure")
            .tail(1)
        )
        n_depth, n_est = df["n_depth"].values[0], df["n_est"].values[0]

        # Train with the whole network
        # Todo: This chunk of the code is repetitive. Consider creating a function.
        neg_edge_sampler = NegativeEdgeSampler(
            negative_edge_sampler=self.negative_edge_sampler
        )
        neg_edge_sampler.fit(network)
        pos_src, pos_trg, _ = sparse.find(network)
        neg_src, neg_trg = neg_edge_sampler.sampling(len(pos_src))
        Y, src, trg = (
            np.concatenate([np.ones_like(pos_src), np.zeros_like(neg_src)]),
            np.concatenate([pos_src, neg_src]),
            np.concatenate([pos_trg, neg_trg]),
        )

        feature_dframe = calc_feature_set(network, src, trg)
        X = self.imputing_missing_values(feature_dframe.values)
        self.pred_model = RandomForestClassifier(
            max_depth=n_depth, n_estimators=n_est
        ).fit(X, Y)
        return self

    def generate_train_test_datasets(self, network, n_cv, n_train_samples):
        # Prep negative edge sampler
        neg_edge_sampler = NegativeEdgeSampler(
            negative_edge_sampler=self.negative_edge_sampler
        )
        neg_edge_sampler.fit(network)

        datasets = []
        for _ in range(n_cv):
            lpdataset = LinkPredictionDataset(
                testEdgeFraction=self.val_edge_frac,
                negative_edge_sampler=self.negative_edge_sampler,
            )
            lpdataset.fit(network)
            train_net, eval_edge_table = lpdataset.transform()
            Y_eval, eval_src, eval_trg = tuple(
                eval_edge_table[["isPositiveEdge", "src", "trg"]].values.astype(int).T
            )
            feature_dframe = calc_feature_set(train_net, eval_src, eval_trg)
            self.var_names = list(feature_dframe.columns)
            X_eval = self.imputing_missing_values(feature_dframe.values)

            # Compute the feature matrix. If the number of edges is less than a prescribed
            # number of training samples, we perform random up-sampling, and otherwise
            # down-sampling them.
            pos_src, pos_trg, _ = sparse.find(train_net)
            neg_src, neg_trg = neg_edge_sampler.sampling(len(pos_src))
            Y_train, train_src, train_trg = (
                np.concatenate([np.ones_like(pos_src), np.zeros_like(neg_src)]),
                np.concatenate([pos_src, neg_src]),
                np.concatenate([pos_trg, neg_trg]),
            )
            if len(train_src) < n_train_samples:
                feature_dframe = calc_feature_set(train_net, train_src, train_trg)
                X_train = self.imputing_missing_values(feature_dframe.values)
                ind = np.random.randint(0, X_train.shape[0], size=n_train_samples)
                Y_train, X_train = Y_train[ind], X_train[ind, :]
            elif len(eval_trg) >= n_train_samples:
                ind = np.random.randint(0, len(eval_trg), size=n_train_samples)
                feature_dframe = calc_feature_set(
                    train_net, train_src[ind], train_trg[ind]
                )
                X_train = self.imputing_missing_values(feature_dframe.values)
            datasets.append({"train": (X_train, Y_train), "test": (X_eval, Y_eval)})
        return datasets

    def predict(self, network, src, trg):
        """Predicts the probability of a link between two nodes.

        Parameters
        ----------
        network : sparse matrix
            The adjacency matrix of the graph.
        src : array-like
            The source nodes.
        trg : array-like
            The target nodes.
        Returns
        -------
        array-like
            The predicted probabilities of a link between each pair of nodes.
        """
        feature_dframe = calc_feature_set(network, src, trg)
        X = self.imputing_missing_values(feature_dframe.values)
        return self.pred_model.predict_proba(X)[:, 1]

    def imputing_missing_values(self, X):
        """Impute missing values in input data using column mean.

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Data with the same shape as X but with missing values replaced by their respective column means.
        """
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        return X

    def load(self, filename):
        """
        Load the object's attributes from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the object's attributes from.

        Returns
        -------
        None
        """

        with open(filename, "rb") as f:
            dump = pickle.load(f)
        for k, v in dump.items():
            setattr(self, k, v)

    def save(self, filename):
        """
        Save the object's attributes to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the object's attributes to.

        Returns
        -------
        None
        """

        dump = vars(self)
        with open(filename, "wb") as f:
            pickle.dump(dump, f)

    def get_feature_importance(self):
        """Get feature importance"""
        return dict(zip(self.var_names, self.pred_model.feature_importances_))


def calc_feature_set(A, target_src, target_trg):
    # ------------------------------------------
    # stats
    # ------------------------------------------
    _src, _trg, _weight = sparse.find(A)
    n_nodes = A.shape[0]
    n_edges = len(_src)

    retvals = OrderedDict(
        {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
        }
    )
    # ------------------------------------------
    # Degree
    # ------------------------------------------
    deg = np.array(A.sum(axis=1)).reshape(-1)
    ave_deg_net = np.sum(deg) / A.shape[0]
    var_deg_net = np.std(deg)
    ave_neigh_deg = np.array(A @ deg).reshape(-1) / deg
    retvals.update(
        {
            "ave_deg_net": ave_deg_net,
            "var_deg_net": var_deg_net,
            "ave_neigh_deg1": ave_neigh_deg[target_src],
            "ave_neigh_deg2": ave_neigh_deg[target_trg],
            "deg_cent1": deg[target_src],
            "deg_cent2": deg[target_trg],
            "deg_ass_net": stats.pearsonr(deg[_src], deg[_trg])[0],
        }
    )

    # ------------------------------------------
    # Triangles & Clustering
    # ------------------------------------------
    # average (local) clustering coefficient (ACC)
    Atruss = A.copy()
    Atruss.data = calc_edge_truss_number(A.indptr, A.indices, A.shape[0])
    n_local_triangles = np.array(Atruss.sum(axis=1)).reshape(-1) / 2
    local_clust_coef = 2 * n_local_triangles / (deg * np.maximum(deg - 1, 1))
    ave_clust_net = np.mean(local_clust_coef)
    transitivity = 2 * np.sum(n_local_triangles) / np.sum(deg * np.maximum(deg - 1, 1))
    retvals.update(
        {
            "ave_clust_net": ave_clust_net,
            "num_triangles_1": n_local_triangles[target_src],
            "num_triangles_2": n_local_triangles[target_trg],
            "clust_coeff1": local_clust_coef[target_src],
            "clust_coeff2": local_clust_coef[target_trg],
            "transit_net": transitivity,
        }
    )
    # ------------------------------------------
    # Jaccard’s coefficient of neighbor sets of i, j (JC)
    # ------------------------------------------
    #
    common_neighbors = np.array(
        (A[target_src, :].multiply(A[target_trg, :])).sum(axis=1)
    ).reshape(-1)
    jacc_coeff = common_neighbors / np.maximum(
        deg[target_src] + deg[target_trg] - common_neighbors, 1
    )
    retvals.update(
        {
            "com_ne": common_neighbors,
            "jacc_coeff": jacc_coeff,
        }
    )
    # -----------------------------------------------
    # Leicht-Holme-Newman index of neighbor sets of i, j (LHN)
    # -----------------------------------------------
    retvals.update(
        {
            "LHN": common_neighbors / (deg[target_src] * deg[target_trg]),
        }
    )

    # ------------------------------------------
    # resource allocation index of i, j (RA)
    # ------------------------------------------
    deg_inv = 1 / np.maximum(deg, 1)
    deg_inv[deg == 0] = 0
    res_alloc_ind = np.array(
        ((A[target_src, :] @ sparse.diags(deg_inv)).multiply(A[target_trg, :])).sum(
            axis=1
        )
    ).reshape(-1)
    retvals.update(
        {
            "res_alloc_ind": res_alloc_ind,
        }
    )

    # ------------------------------------------
    # Adamic/Adar index of i, j (AA)
    # ------------------------------------------
    log_deg_inv = 1 / np.maximum(np.log(np.maximum(deg, 1)), 1)
    log_deg_inv[deg == 0] = 0
    adam_adar = np.array(
        ((A[target_src, :] @ sparse.diags(log_deg_inv)).multiply(A[target_trg, :])).sum(
            axis=1
        )
    ).reshape(-1)
    retvals.update(
        {
            "adam_adar": adam_adar,
        }
    )

    # ------------------------------------------
    # preferential attachment (degree product) of i, j (PA)
    # ------------------------------------------
    retvals.update(
        {
            "pref_attach": deg[target_src] * deg[target_trg],
        }
    )
    # ------------------------------------------
    # Page rank values for i and j (PR_i, PR_j)
    # ------------------------------------------
    P = sparse.diags(1.0 / deg) @ A
    b = np.ones((1, P.shape[0]))
    page_rank = calc_page_rank(P, b).reshape(-1)
    personalized_page_rank = calc_page_rank(P, np.eye(P.shape[0]))
    retvals.update(
        {
            "pag_rank1": page_rank[target_src],
            "pag_rank2": page_rank[target_trg],
            "page_rank_pers_edges": np.array(
                personalized_page_rank[(target_src, target_trg)]
            ).reshape(-1),
        }
    )
    # ------------------------------------------
    # Eigenvector centrality
    # ------------------------------------------
    eig_cent = deg / np.linalg.norm(deg)
    retvals.update(
        {
            "eig_cent1": eig_cent[target_src],
            "eig_cent2": eig_cent[target_trg],
        }
    )

    # ------------------------------------------
    # Katz centralities for i and j (KC_i, KC_j)
    # ------------------------------------------
    alpha, beta = 0.1, 1.0
    ktz_cent = beta * np.array(
        sparse.linalg.inv(sparse.eye(A.shape[0]) - alpha * A).sum(axis=1)
    ).reshape(-1)
    retvals.update(
        {
            "ktz_cent1": ktz_cent[target_src],
            "ktz_cent2": ktz_cent[target_trg],
        }
    )

    # -----------------------------------------------
    # closeness centralities for i and j (CC_i, CC_j)
    # -----------------------------------------------
    # closeness centralities for i and j (CC_i, CC_j)
    D = sparse.csgraph.shortest_path(A, directed=False)
    D = np.ma.masked_invalid(D)
    closn_cent = 1.0 / np.array(D.sum(axis=1)).reshape(-1) * (D.shape[0] - 1)
    retvals.update(
        {
            "clos_cent1": closn_cent[target_src],
            "clos_cent2": closn_cent[target_trg],
            "short_path": np.array(D[(target_src, target_trg)]).reshape(-1),
            "diam_net": float(D.max()),
        }
    )
    # -----------------------------------------------
    # Betwenness
    # -----------------------------------------------

    sources, targets = A.nonzero()
    g = igraph.Graph(zip(sources.tolist(), targets.tolist()))
    betw_cent = np.array(g.betweenness())
    retvals.update(
        {
            "betw_cent1": betw_cent[target_src],
            "betw_cent2": betw_cent[target_trg],
        }
    )

    # -----------------------------------------------
    # SVDs
    # -----------------------------------------------

    U, sig, V = sparse.linalg.svds(
        A.astype(float), k=int(np.ceil(np.sqrt(A.shape[0]))) + 1, which="LM"
    )
    Usig = np.einsum("ij,j->ij", U, sig)
    svd_edge_approx = np.sum(Usig[target_src] * V[:, target_trg].T, axis=1)

    # (U @ sig @ V) @ (V.T @ sig @ U.T)
    VV = V @ V.T
    UsigVVsig = U @ np.einsum("ij,j->ij", np.einsum("ij,i->ij", VV, sig), sig)
    svd_edge_approx_inner = np.sum(UsigVVsig[target_src] * U[target_trg, :], axis=1)

    retvals.update(
        {
            "betw_cent1": betw_cent[target_src],
            "betw_cent2": betw_cent[target_trg],
            "svd_edges_approx": svd_edge_approx,
            "svd_edges_dot_approx": svd_edge_approx_inner,
        }
    )
    return pd.DataFrame(retvals)


@njit(nogil=True)
def calc_edge_truss_number(indptr, indices, n_nodes):
    n_edges = len(indices)
    retval = np.zeros(n_edges, dtype=np.float64)
    for i in range(n_nodes):
        neighbors = indices[indptr[i] : indptr[i + 1]]
        deg = len(neighbors)

        if deg <= 1:
            continue
        # Count the common neighbors
        for j, nei in enumerate(neighbors):
            neighbor_neighbors = indices[indptr[nei] : indptr[nei + 1]]
            common_neighbors = np.intersect1d(neighbors, neighbor_neighbors) - 1
            retval[indptr[i] + j] = len(common_neighbors)
    return retval


def calc_page_rank(P, b, alpha=0.85, tol=1.0e-6):
    x = np.random.rand(*b.shape)
    x = np.einsum("ij,i->ij", x, 1 / np.sum(x, axis=1))
    b = np.einsum("ij,i->ij", b, 1 / np.sum(b, axis=1))
    err = 100
    while err > tol:
        xnew = alpha * x @ P + (1 - alpha) * b
        err = np.mean(np.abs(xnew - x))
        x = xnew.copy()
    return x
