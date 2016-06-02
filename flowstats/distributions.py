"""
Distributions used in FCS analysis
"""

import numpy as np
from numpy.random import random, multivariate_normal
from scipy.misc import logsumexp

try:
    from gpustats import mvnpdf_multi
    from dpmix_exp.utils import select_gpu
    has_gpu = True
except ImportError:
    has_gpu = False
    mvnpdf_multi = None
    select_gpu = None

from dpmix_exp.utils import mvn_weighted_logged


def _mvnpdf(x, mu, va, logged=False, use_gpu=True, **kwargs):
    if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
    if len(mu.shape) == 1:
        mu = mu.reshape((1, mu.shape[0]))
    if len(va.shape) == 2:
        va = va.reshape(1, va.shape[0], va.shape[1])

    if has_gpu and use_gpu:
        if 'device' in kwargs:
            dev = kwargs['device']
        else:
            dev = 0
        select_gpu(dev)
        return mvnpdf_multi(
            x,
            mu,
            va,
            weights=np.ones(mu.shape[0]),
            logged=logged,
            order='C').astype('float64')
    else:
        if logged:
            return mvn_weighted_logged(x, mu, va, np.ones(mu.shape[0]))
        else:
            return np.exp(mvn_weighted_logged(x, mu, va, np.ones(mu.shape[0])))


def _wmvnpdf(x, pi, mu, va, logged=False, use_gpu=True, **kwargs):
    if len(x.shape) == 1:
        x = x.reshape((1, x.shape))
    if len(mu.shape) == 1:
        mu = mu.reshape((1, mu.shape))
    if len(va.shape) == 2:
        va = va.reshape(1, va.shape[0], va.shape[1])

    if len(va.shape) == 1:
        va = va.reshape(va.shape[0], 1, 1)

    if isinstance(pi, float) or isinstance(pi, int):
        pi = np.array([pi])
    elif isinstance(pi, np.ndarray):
        if len(pi.shape) == 0:
            pi = pi.reshape(1)

    if has_gpu and use_gpu:
        if 'device' in kwargs:
            dev = kwargs['device']
        else:
            dev = 0
        select_gpu(dev)
        return mvnpdf_multi(
            x,
            mu,
            va,
            weights=pi,
            logged=logged,
            order='C').astype('float64')
    else:
        if logged:
            return mvn_weighted_logged(x, mu, va, pi)
        else:
            return np.exp(mvn_weighted_logged(x, mu, va, pi))


def mvnormpdf(x, mu, va, **kwargs):
    """
    calculate the multi-variate normal pdf
    D(x, mu, sigma) -> float
    """
    if len(mu.shape) > 1:  # single point in multi dim
        pass
    else:  # many points in single dim
        x = x.reshape(x.shape[0], 1)
    results = _mvnpdf(x, mu, va, **kwargs)

    return results.squeeze()


def compmixnormpdf(x, prop, mu, sigma, **kwargs):
    """Component mixture multivariate normal pdfs"""
    try:
        n, d = x.shape
    except ValueError:
        if len(mu.shape) == 1 or mu.shape[1] == 1:  # 1-D system, so many points
            n = x.shape[0]
            d = 1
        else:  # single point in a multi dimensional system
            n = 1
            d = x.shape[0]
        x = x.reshape((n, d))
    try:
        c = prop.shape[0]
    except AttributeError:
        c = 1
    except IndexError:
        c = 1

    if c == 1:
        tmp = _wmvnpdf(x, prop, mu, sigma, **kwargs)
        if n == 1:
            tmp = tmp[0]
    else:
        tmp = _wmvnpdf(x, prop, mu, sigma, **kwargs)
        tmp = np.reshape(tmp, (n, c))
        if n == 1:
            tmp = tmp[0]
    return tmp


def mixnormpdf(x, prop, mu, sigma, **kwargs):
    """Mixture of multivariate normal pdfs"""
    if 'logged' in kwargs:
        logged = kwargs['logged']
    else:
        logged = False
    tmp = compmixnormpdf(x, prop, mu, sigma, **kwargs)
    try:
        if logged:
            return logsumexp(tmp, 1)
        else:
            return np.sum(tmp, 1)
    except ValueError:
        if logged:
            return logsumexp(tmp, 0)
        else:
            return np.sum(tmp, 0)


def mixnormrnd(pi, mu, sigma, k):
    """Generate random variables from mixture of Guassians"""
    xs = []
    for unused in range(k):
        j = np.sum(random() > np.cumsum(pi))
        xs.append(multivariate_normal(mu[j], sigma[j]))
    return np.array(xs)
