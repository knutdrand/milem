from dataclasses import dataclass
from scipy.special import logsumexp
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
        return RaggedArray(logps, X.shape).sum(axis=-1)

    def fit(self, X: RaggedArray, weights: np.ndarray):
        w = X.shape.broadcast_values(weights[:, np.newaxis])
        counts = np.bincount(X.ravel(), w, minlength=self.log_ps.size)
        self.log_ps = counts/counts.sum()

    def sample(self, n):
        pass

@dataclass
class SparseEM:
    dists: list
    log_weights: list

    def calculate_responsibilities(self, X) -> list:
        logpmfs = [log_weight + dist.logpmf(X)
                   for dist, log_weight in zip(self.dists, self.log_weights)]
        tot_logpmf = logsumexp(logpmfs, axis=0)
        return [np.exp(logpmf-tot_logpmf) for logpmf in logpmfs]

    def adjust_distribution_parameters(self, X, responsibilities):
        for dist, dist_responsibilities in zip(self.dists, responsibilities):
            dist.fit(X, dist_responsibilities)

        self.log_weights = [logsumexp(responsibilities)-np.log(len(X))
                            for dist_responsibilities in responsibilities]

    def fit(self, X):
        for _ in range(self._n_iter):
            responsibilities = self.calculate_responsibilities(X)
            self.adjust_distribution_parameters(self, X, responsibilities)


class SparseMIL(SparseEM):
    def fit(self, X, y):
        pass
