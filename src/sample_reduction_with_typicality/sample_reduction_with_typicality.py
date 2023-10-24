import logging
import sys
from collections import defaultdict
from typing import Callable, Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as sp
from scipy.spatial.distance import cdist
from scipy.special import digamma, gamma
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class SampleReductionWithTypicality(object):
    """
    Reduce/downsize a matrix based on entropy.
    Main method: reduce
    Given a matrix of n samples and m features, reduce/downsize it to n_ samples, a subset of n. The samples are chosen in the way that minimizes the loss of entropy, but still allow to best describe the original distribution.
    We sample amongst the most typical and most atypical samples
    The class contains several static methods around the concept of typicality

    Parameters
    ----------
    batch_size: int, default: 5000
                Size of the batch to run. When the input matrix is very large, we build the clusters on a random subset defined by the batch_size, then assign each remaining points to their cluster, by batch.

    final_nb_rows: int, default: -1
                   Final number of rows desired. The algorithm will not take into account the entropy it will just try to get as close as possible to that number. Note that the number cannot be exactly matched due to how the data is sampled, but it will be close. The sampling increases by 1% data from the largest cluster.
                   When The default is used than that parameter is ignored.

    nb_neighbors: int, default: 1
                  Number of neighbors to use with KNN to build the entropy we recommend leaving that value at 1.

    max_pct_ent_error: int, default: 5
                       Maximum error allowed on the entropy from the reduced dataset compared to the full entropy. final_nb_rows takes precedence

    min_nb_rows: int, default: 0
                 Minimum number of rows desired. Once that threshold is met, the algorithm will either continue or stop based on max_pct_ent_error

    max_nb_rows: int, default: -1
                 Maximum number of rows desired. The algorithm will stop once the threshold is met, even if max_pct_ent_error is larger than desired

    verbose: [int, bool], default: 1
             Whether to print information when the method is running or not

    return_clusters: [int, bool], default: 0
                     Whether the main method `reduce`, should return the clusters, in addition to the reduced matrix or not

    Example
    -------
    # Construct a dataset
    n = 1000
    Rho = [
        [1.35071688, 0.15321558, 0.84785951, 0.82255503, -0.33551541, 0.62205449, 0.42880575],
        [0.15321558, 1.35113273, -0.66183342, 0.74442862, 0.67287063, -0.28934146, 0.34474363],
        [0.84785951, -0.66183342, 1.16071755, 0.21553483, -0.54921448, 0.55342434, 0.42030557],
        [0.82255503, 0.74442862, 0.21553483, 1.27186731, 0.80719934, 0.81152044, 0.89989037],
        [-0.33551541, 0.67287063, -0.54921448, 0.80719934, 1.46557044, 0.58029024, 0.74410743],
        [0.62205449, -0.28934146, 0.55342434, 0.81152044, 0.58029024, 1.32526075, 0.60227779],
        [0.42880575, 0.34474363, 0.42030557, 0.89989037, 0.74410743, 0.60227779, 1.09473434],
    ]
    Z = np.random.multivariate_normal([0] * 7, Rho, n)
    U = sp.norm.cdf(Z, 0, 1)
    X_large_sample = [
        sp.gamma.ppf(U[:, 0], 2, scale=1),
        sp.beta.ppf(U[:, 1], 2, 2),
        sp.t.ppf(U[:, 2], 5),
        sp.gamma.ppf(U[:, 3], 2, scale=1),
        sp.t.ppf(U[:, 4], 2),
        sp.t.ppf(U[:, 5], 3),
        sp.t.ppf(U[:, 6], 15),
    ]

    beta = [0.1, 2, -0.5, 2, 0.3, 0.4, 10]
    sigma = np.random.normal(0, 10, len(X_large_sample[0]))
    y = (
        beta[0] * X_large_sample[0]
        + beta[1] * X_large_sample[1]
        + beta[2] * X_large_sample[2]
        + beta[3] * X_large_sample[3] ** 2
        + beta[4] * X_large_sample[4]
        + beta[5] * X_large_sample[5]
        + beta[6] * X_large_sample[6]
        + sigma
    )
    X_large = [
        X_large_sample[0],
        X_large_sample[1],
        X_large_sample[2],
        X_large_sample[3],
        X_large_sample[4],
        X_large_sample[5],
        X_large_sample[6],
        y,
    ]
    data = pd.DataFrame(
        {
            "x0": X_large[0],
            "x1": X_large[1],
            "x2": X_large[2],
            "x3": X_large[3],
            "x4": X_large[4],
            "x5": X_large[5],
            "x6": X_large[6],
            "y": X_large[7],
        }
    )
    # Reduce
    srwt = SampleReductionWithTypicality(batch_size=200, verbose=True)
    X_final_large = srwt.reduce(data.to_numpy())
    log.info(f"Start number of rows: {n}. End number of rows: {X_final_large.shape}")
    log.info("Testing SampleReductionWithTypicality class")
    """

    def __init__(
        self,
        batch_size: int = 5000,
        final_nb_rows: int = -1,
        nb_neighbors: int = 1,
        max_pct_ent_error: int = 5,
        min_nb_rows: int = 0,
        max_nb_rows: int = -1,
        verbose: Union[int, bool] = 1,
        return_clusters: Union[int, bool] = 0,
    ):
        self.batch_size = batch_size
        self.final_nb_rows = final_nb_rows
        self.nb_neighbors = nb_neighbors
        self.max_pct_ent_error = max_pct_ent_error
        self.min_nb_rows = min_nb_rows
        self.max_nb_rows = max_nb_rows
        self.verbose = verbose
        self.return_clusters = return_clusters

    def reduce(self, X_original: npt.NDArray) -> Union[npt.NDArray, Tuple[npt.NDArray, Dict]]:
        """
        Main method used to downsample a dataset.

        Parameters
        ----------

        X_original: npt.NDArray
                    Your dataset
        """
        cluster_flag = True
        batch_size = self.batch_size
        final_nb_rows = self.final_nb_rows
        nb_neighbors = self.nb_neighbors
        max_pct_ent_error = self.max_pct_ent_error
        min_nb_rows = self.min_nb_rows
        max_nb_rows = self.max_nb_rows
        verbose = self.verbose
        indices = next(self.get_indices(X_original, batch_size))
        if verbose:
            log.info("Get the full entropy")
        if cluster_flag:
            ml, d = X_original.shape
            H_search_inc = max(1, int(ml * 0.01))
            k = nb_neighbors
            _, distances = self.k_nearest_neighbors(X_original, k=nb_neighbors)
            H_full = self.get_entropy(ml, k, d, distances)
            if verbose:
                log.info(f"Full entropy: {H_full:.3f}")
        if max_nb_rows == -1:
            max_nb_rows = X_original.shape[0]
        X = X_original[indices, :]

        if cluster_flag:
            if verbose:
                log.info("Generating the clusters")
            clusters: dict = defaultdict(int)
            mu = X.mean(axis=0).reshape(1, -1)
            nb_features = X.shape[1]
            cov = np.cov(X.T)
            distances_sq = self.vec_vec_distances_sq_fcn(X)
            distances_sq_sum = distances_sq.sum()
            K = X.shape[0]
            data_density_mat_all = self.data_density_mat_fcn_all(distances_sq, distances_sq_sum, K)
            dd_mat = 1 / (1 + (np.linalg.norm(X - mu, axis=1) ** 2) / (np.sum(np.diag(cov))))
            dd_epsilon = np.std(dd_mat) * 0.0
            clusters[0] = [
                X[[0], :].mean(axis=0).reshape(1, -1),
                np.ones((nb_features, nb_features)),
                1,
                dd_mat[0],
            ]
            last_cluster_id = 0
            nb_features = X.shape[1]
            mu_clusters = np.array(clusters[0][0])
            min_dd, max_dd = dd_mat[0] - dd_epsilon, dd_mat[0] + dd_epsilon
            for i in range(1, X.shape[0]):
                new_cluster_flag = False
                k = i
                x = X[[i], :]
                dd = data_density_mat_all[i]
                if dd < min_dd:
                    new_cluster_flag = True
                    min_dd = dd - dd_epsilon
                if dd > max_dd:
                    new_cluster_flag = True
                    max_dd = dd + dd_epsilon
                if new_cluster_flag:
                    last_cluster_id += 1
                    mu = x.reshape(1, -1)
                    cov = np.ones((nb_features, nb_features))
                    clusters[last_cluster_id] = [mu, cov, 1, dd_mat[i]]
                    mu_clusters = np.concatenate((mu_clusters, mu), axis=0)
                else:
                    cluster_id_to_update = np.argmin(cdist(x, mu_clusters))
                    mu, cov, nb_pts, dd = clusters[cluster_id_to_update]
                    mu_old = mu
                    mu = self.update_mu(
                        mu, nb_pts, 1, x
                    )  # 1 cause x is 1 point only of dimension (1, nb_features)
                    cov = self.update_cov(
                        cov, mu, mu_old, nb_pts, 1, x
                    )  # 1 cause x is 1 point only of dimension (1, nb_features)
                    var = np.sum(np.diag(cov))
                    dd = 1 / (1 + np.linalg.norm(x - mu) ** 2 / var)
                    nb_pts += 1
                    clusters[cluster_id_to_update] = [mu, cov, nb_pts, dd]

            flag = False
            for ind in clusters:
                mu = clusters[ind][0]
                mus: npt.NDArray = np.concatenate((mus, mu), axis=0) if flag else mu
                flag = True

        X = X_original
        if verbose:
            log.info("Assign points to clusters and reduce")
        cluster_assignment = np.empty(0)
        ind = 0
        for indices in self.get_indices(X_original, batch_size):
            cluster_assignment = np.concatenate(
                (cluster_assignment, np.argmin(cdist(X[indices, :], mus), axis=1))
            )
            ind += 1
        if verbose:
            log.info("Assignment done")
        cloud_dict = {}
        try:
            assert len(clusters) >= len(np.unique(cluster_assignment))
        except AssertionError as a:
            err_msg = f'Number of clusters is less than the number of unique cluster assignment: {len(clusters)} <> {len(np.unique(cluster_assignment))}'
            raise AssertionError(err_msg)
        sorted_inds = {}
        cluster_sizes = {}
        nb_clusters = len(clusters)
        if verbose:
            log.info("Look for the nearest neighbors")
        loop = 0
        valid_cluster_ids = []
        for ind in range(len(clusters)):
            cloud_inds = np.where(cluster_assignment == ind)[0]
            cloud_dict[ind] = X[cloud_inds]
            # because we reassign the points to the nearest clusters, some might lose all their points
            # we need to remove such cases
            if cloud_dict[ind].shape[0] > 0:
                valid_cluster_ids.append(ind)
                if verbose:
                    log.info(
                        f"Cluster {ind} out of {nb_clusters}, with {cloud_dict[ind].shape[0]} points"
                    )
                neighbours_inds, distances = self.k_nearest_neighbors(
                    cloud_dict[ind], k=min(nb_neighbors, cloud_dict[ind].shape[0] - 1)
                )
                sorted_inds[ind] = np.argsort(distances.sum(axis=1))
                cluster_sizes[ind] = cloud_dict[ind].shape[0]
            loop += 1
        if verbose:
            log.info("done")

        if final_nb_rows > 0:
            nb_pts = self.binary_search(
                self.cluster_pts_func,
                final_nb_rows,
                cluster_sizes,
                X_original.shape[0],
                0,
                max_iterations=500,
            )
            X_transient = np.empty((0, d))
            k = nb_neighbors
            for ind in valid_cluster_ids:
                X_transient = np.concatenate(
                    (
                        X_transient,
                        self.get_top_bottom_entropy_pts(
                            cloud_dict[ind][sorted_inds[ind], :], nb_pts
                        ),
                    )
                )
            _, distances = self.k_nearest_neighbors(X_transient, k=nb_neighbors)
            ml, d = distances.shape[0], nb_features
            H_transient = self.get_entropy(ml, k, d, distances)
            pct_ent_error = abs(H_transient - H_full) / H_full * 100
            if verbose:
                log.info(
                    f"pct error: {pct_ent_error}; Reduced matrix number of rows: {X_transient.shape[0]}; reduced sample entropy: {H_transient:0.3f}; full entropy: {H_full:0.3f}"
                )
        else:
            if verbose:
                log.info("Search for the points to pick")
            pct_ent_error = 100
            H_search_inc = int(max([cluster_sizes[s] for s in cluster_sizes]) * 0.01)
            nb_pts = H_search_inc
            ind = 0
            nb_transient_rows = min_nb_rows
            while (
                not ((pct_ent_error < max_pct_ent_error) and (min_nb_rows <= nb_transient_rows))
                and (ind < 500)
                and (nb_transient_rows <= max_nb_rows)
            ):
                ind += 1
                H_transient = 0
                k = nb_neighbors
                X_transient = np.empty((0, d))
#                 for ind in range(len(clusters)):
                for ind in valid_cluster_ids:
                    X_transient = np.concatenate(
                        (
                            X_transient,
                            self.get_top_bottom_entropy_pts(
                                cloud_dict[ind][sorted_inds[ind], :], nb_pts
                            ),
                        )
                    )
                _, distances = self.k_nearest_neighbors(X_transient, k=nb_neighbors)
                ml, d = distances.shape[0], nb_features
                H_transient = self.get_entropy(ml, k, d, distances)
                pct_ent_error = abs(H_transient - H_full) / H_full * 100
                nb_transient_rows = X_transient.shape[0]
                if verbose:
                    log.info(
                        f"pct error: {pct_ent_error:0.3f}; Reduced matrix number of rows: {nb_transient_rows}; reduced sample entropy: {H_transient:0.3f}; full entropy: {H_full:0.3f}"
                    )
                nb_pts += H_search_inc

        if self.return_clusters:
            return X_transient, clusters
        return X_transient

    @staticmethod
    def vec_vec_distances_sq_fcn(X, metric="euclidean"):
        """
        Overall point-point distance square
        """
        return np.square(cdist(X, X, metric=metric))

    @staticmethod
    def cumulative_proximity_fcn(distances_sq, i):
        """
        i: index -> ith vector to get the proximity
        """
        return distances_sq[i].sum()

    def eccentricity_fcn(self, distances_sq, i, distances_sq_sum):
        """
        Calculates the eccentricity for a given index i
        """
        return 2 * self.cumulative_proximity_fcn(distances_sq, i) / distances_sq_sum

    def standardized_eccentricity_fcn(self, distances_sq, i, distances_sq_sum, K):
        """
        Calculates the standardized eccentricity for a given index i
        """
        return 2 * self.cumulative_proximity_fcn(distances_sq, i) / (distances_sq_sum / K)

    @staticmethod
    def data_density_fcn(standardized_eccentricity):
        """
        Calculates the data density for a given standardized eccentricity
        """
        return 1 / standardized_eccentricity

    @staticmethod
    def data_density_mat_fcn_all(distances_sq, distances_sq_sum, K):
        """
        Calculates the data density matrix for all data points
        """
        return (distances_sq_sum / K) / (2 * distances_sq.sum(axis=0))

    @staticmethod
    def typicality_fcn(data_density_mat_all, data_density_mat_all_sum, i):
        """
        Calculates the typicality forgiven index
        """
        return data_density_mat_all[i] / data_density_mat_all_sum

    @staticmethod
    def update_mu(mu, k, kp, x):
        """
        Recursively update a mean
        """
        return mu * k / (k + kp) + x.mean(axis=0) * kp / (k + kp)

    @staticmethod
    def update_cov(cov, mu, mu_old, k, kp, x):
        """
        Recursively update a covariance matrix
        """
        return (
            cov * (k - 1) / (k + kp - 1)
            + np.matmul(x.T, x) / (k + kp - 1)
            + np.matmul(mu_old.T, mu_old) * k / (k + kp - 1)
            - np.matmul(mu.T, mu) * (k + kp) / (k + kp - 1)
        )

    @staticmethod
    def k_nearest_neighbors(X, k=1):
        """
        Uses the NearestNeighbors class from SKlearn
        The first element is the point itself so we remove that and take the first point after, hence the k+1 and 1:
        We return the indices of the neighbours, and the distances
        """
        knn = NearestNeighbors(n_neighbors=k + 1)
        knn.fit(X)
        distance_mat, neighbours_mat = knn.kneighbors(X)
        # add +1 to avoid log(0) afterwards
        return neighbours_mat[:, 1:], distance_mat[:, 1:] + 1

    @staticmethod
    def get_indices(X, batch_size):
        """
        Iterator That returns indexes to use to create batches of size batch_size on the input matrix X. To be used in a for loop or next()
        Given a matrix and a batch size, it returns the indices to split the matrix into batches
        """
        nb_rows = X.shape[0]
        start_ind = 0
        end_ind = batch_size
        loop_flag = True
        while loop_flag:
            yield range(start_ind, min(nb_rows, end_ind))
            if nb_rows <= end_ind:
                loop_flag = False
            start_ind += batch_size
            end_ind += batch_size

    @staticmethod
    def get_top_bottom_entropy_pts(X, nb_pts):
        """
        We want to get nb_pts split between the top nb_pts/2 and bottom nb_pcts/2. So you want X to be sorted before hand
        """
        limit = nb_pts // 2
        nb_rows = X.shape[0]
        return np.concatenate((X[:limit, :], X[max(limit, (nb_rows - limit)) :, :]), axis=0)

    @staticmethod
    def get_entropy(ml, k, d, distances):
        """
        Entropy definition as per paper
        """
        d = min(d, 100) # to deal with gamma function reaching infinite values
        return (
            np.log(ml)
            - digamma(k)
            + np.log(np.pi ** (d / 2) / gamma((1 + d / 2)))
            + d / ml * (np.log(distances).sum())
        )

    @staticmethod
    def cluster_pts_func(cluster_sizes, nb_pts):
        """
        Get the number of points given a threshold.
        Say you have 3 clusters of size 20, 10, 5
        nb_pts references the number of point you want per cluster. Let's set that number to 7. That will yield 7 from 20, 7 from 10 and 5 from 5 so total 19.
        """
        return sum(
            [
                cluster_sizes[i] if (cluster_sizes[i] < nb_pts) else nb_pts
                for i in range(len(cluster_sizes))
            ]
        )

    @staticmethod
    def binary_search(
        f: Callable[[int, int], int], target, cluster_sizes, ub, lb, max_iterations=500
    ):
        """
        Simple binary search where f is a function
        """
        inc = 0
        res = 0
        while (abs(ub - lb) > 1) and (inc < max_iterations) and abs(res - target) > 1:
            inc += 1
            x = (ub + lb) / 2.0
            res = f(cluster_sizes, x)
            if res > target:
                ub = x
            else:
                lb = x
        return int(x)


def main():
    log.info("Testing the SampleReductionWithTypicality class")
    # Construct a dataset
    n = 1000
    Rho = [
        [1.35071688, 0.15321558, 0.84785951, 0.82255503, -0.33551541, 0.62205449, 0.42880575],
        [0.15321558, 1.35113273, -0.66183342, 0.74442862, 0.67287063, -0.28934146, 0.34474363],
        [0.84785951, -0.66183342, 1.16071755, 0.21553483, -0.54921448, 0.55342434, 0.42030557],
        [0.82255503, 0.74442862, 0.21553483, 1.27186731, 0.80719934, 0.81152044, 0.89989037],
        [-0.33551541, 0.67287063, -0.54921448, 0.80719934, 1.46557044, 0.58029024, 0.74410743],
        [0.62205449, -0.28934146, 0.55342434, 0.81152044, 0.58029024, 1.32526075, 0.60227779],
        [0.42880575, 0.34474363, 0.42030557, 0.89989037, 0.74410743, 0.60227779, 1.09473434],
    ]
    Z = np.random.multivariate_normal([0] * 7, Rho, n)
    U = sp.norm.cdf(Z, 0, 1)
    X_large_sample = [
        sp.gamma.ppf(U[:, 0], 2, scale=1),
        sp.beta.ppf(U[:, 1], 2, 2),
        sp.t.ppf(U[:, 2], 5),
        sp.gamma.ppf(U[:, 3], 2, scale=1),
        sp.t.ppf(U[:, 4], 2),
        sp.t.ppf(U[:, 5], 3),
        sp.t.ppf(U[:, 6], 15),
    ]

    beta = [0.1, 2, -0.5, 2, 0.3, 0.4, 10]
    sigma = np.random.normal(0, 10, len(X_large_sample[0]))
    y = (
        beta[0] * X_large_sample[0]
        + beta[1] * X_large_sample[1]
        + beta[2] * X_large_sample[2]
        + beta[3] * X_large_sample[3] ** 2
        + beta[4] * X_large_sample[4]
        + beta[5] * X_large_sample[5]
        + beta[6] * X_large_sample[6]
        + sigma
    )
    X_large = [
        X_large_sample[0],
        X_large_sample[1],
        X_large_sample[2],
        X_large_sample[3],
        X_large_sample[4],
        X_large_sample[5],
        X_large_sample[6],
        y,
    ]
    data = pd.DataFrame(
        {
            "x0": X_large[0],
            "x1": X_large[1],
            "x2": X_large[2],
            "x3": X_large[3],
            "x4": X_large[4],
            "x5": X_large[5],
            "x6": X_large[6],
            "y": X_large[7],
        }
    )
    # Reduce
    srwt = SampleReductionWithTypicality(batch_size=200, verbose=True)
    X_final_large = srwt.reduce(data.to_numpy())
    log.info(f"Start number of rows: {n}. End number of rows: {len(X_final_large)}")
    log.info("Testing SampleReductionWithTypicality class")


if __name__ == "__main__":
    main()
