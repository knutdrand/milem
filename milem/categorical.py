from dataclasses import dataclass
from scipy.special import logsumexp
# from pyclbr import Distribution, MixtureDistribution
import numpy as np

from npstructures import RaggedArray


@dataclass
class Categorical:
    log_ps: np.ndarray

    def logpmf(self, x):
        return self.log_ps[x]

    def sample(self, n):
        pass


class MultiCategorical(Categorical):
    def logpmf(self, X):
        logps = super().logpmf(X.ravel())
        assert np.all(logps <= 0), logps
        return RaggedArray(logps, X.shape).sum(axis=-1)

    def fit(self, X: RaggedArray, weights: np.ndarray):
        w = X.shape.broadcast_values(weights[:, np.newaxis])
        counts = np.bincount(X.ravel(), w, minlength=self.log_ps.size)
        self.log_ps = np.log(counts/counts.sum())

    def sample(self, n):
        pass


@dataclass
class MultiCategoricalRegularized(MultiCategorical):
    pseudocounts: int = 1

    def fit(self, X: RaggedArray, weights: np.ndarray):
        w = X.shape.broadcast_values(weights[:, np.newaxis])
        counts = np.bincount(X.ravel(), w, minlength=self.log_ps.size)+self.pseudocounts
        self.log_ps = np.log(counts/counts.sum())

    def sample(self, n):
        pass


@dataclass
class SparseEM:
    dists: list
    log_weights: list

    def calculate_responsibilities(self, X) -> tuple:
        """Calculate the posterior probability of class beloning

        Parameters
        ----------
        X : RaggedArray
            One row for each sample

        Returns
        -------
        tuple
            (list[np.ndarray] (n_dists x n_samples), float)

        """
        logpmfs = [log_weight + dist.logpmf(X)
                   for dist, log_weight in zip(self.dists, self.log_weights)]

        assert np.all(np.array(logpmfs) < 0), logpmfs
        tot_logpmf = logsumexp(logpmfs, axis=0)
        return [logpmf-tot_logpmf for logpmf in logpmfs], np.mean(tot_logpmf)

    def adjust_distribution_parameters(self, X, responsibilities, n_pos=None):
        for dist, dist_responsibilities in zip(self.dists, responsibilities):
            dist.fit(X, np.exp(dist_responsibilities))

        if n_pos is None:
            n_pos = len(X)
        self.log_weights = [logsumexp(dist_responsibilities[:n_pos])-np.log(n_pos)
                            for dist_responsibilities in responsibilities]
        assert np.allclose(logsumexp(self.log_weights), 0), (self.log_weights, responsibilities)

    def fit(self, X, logpmf_diff_threshold=10e-8, n_iter=500):
        cur_logpmf = -np.inf
        for i in range(n_iter):
            print(i)
            responsibilities, logpmf = self.calculate_responsibilities(X)
            self.adjust_distribution_parameters(X, responsibilities)
            print(i, logpmf, np.exp(self.log_weights))
            if logpmf-cur_logpmf < logpmf_diff_threshold:
                break
            cur_logpmf = logpmf


class SparseMIL(SparseEM):
    def fit(self, pos_X, neg_X, logpmf_diff_threshold=10e-8, n_iter=500):
        # y = y.ravel()
        # pos_X = X[y == 1]
        # neg_X = X[y == 0]
        X = np.concatenate((pos_X, neg_X))
        self.dists[0].fit(neg_X, np.ones(len(neg_X)))
        self.dists[1].fit(pos_X, np.ones(len(pos_X)))
        cur_logpmf = -np.inf
        responsibilities = [np.zeros(len(X)), np.full(len(X), -np.inf)]
        for i in range(n_iter):
            print(i)
            resp, logpmf = self.calculate_responsibilities(pos_X)
            for r, nr in zip(responsibilities, resp):
                r[:len(pos_X)] = nr
            self.adjust_distribution_parameters(X, responsibilities, n_pos=len(pos_X))
            print(i, logpmf, np.exp(self.log_weights))
            if logpmf-cur_logpmf < logpmf_diff_threshold:
                break
            cur_logpmf = logpmf
