import pytest
from milem import MultiCategorical, SparseEM
from npstructures import RaggedArray
from numbers import Number
import numpy as np

n_features = 3
n_samples = 4


@pytest.fixture
def dist():
    log_ps = np.full(n_features, np.log(1/n_features))
    return MultiCategorical(log_ps)


@pytest.fixture
def dist2():
    log_ps = np.full(n_features, np.log(1/n_features))
    return MultiCategorical(log_ps)


@pytest.fixture
def em():
    log_ps = np.full(n_features, np.log(1/n_features))
    weights = np.log([1/3, 2/3])
    dists = [MultiCategorical(log_ps.copy()) for _ in weights]
    return SparseEM(dists, weights)

@pytest.fixture
def log_resp():
    return np.log([np.full(n_samples, 0.5), np.full(n_samples, 0.5)])


@pytest.fixture
def X():
    return RaggedArray(np.arange(10) % n_features, [1, 2, 3, 4])


def test_logpmf_shape(dist, X):
    logpmf = dist.logpmf(X)
    assert logpmf.shape == (n_samples,)


def test_fit_shape(dist, X):
    dist.fit(X, np.ones(n_samples))
    assert dist.log_ps.shape == (n_features,)


def test_em_calculate_responsibilities(em, X):
    resp, logpmf = em.calculate_responsibilities(X)
    assert np.array(resp).shape == (2, n_samples)
    print(logpmf.shape)
    assert isinstance(logpmf, Number)
    # return MultiCategorical(log_ps)


def test_em_adjust(em, X, log_resp):
    em.adjust_distribution_parameters(X, log_resp)
    assert np.array(em.log_weights).shape == (2, )


def test_em_fit(em, X):
    em.fit(X, n_iter=20)
    assert np.array(em.log_weights).shape == (2, )
