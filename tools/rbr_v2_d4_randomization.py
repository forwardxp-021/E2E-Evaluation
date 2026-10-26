#!/usr/bin/env python3
"""D4 prospective candidate only: conditional whole-SESSION randomization, no SAP mutation."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def kernel(x, y, sigma):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all() or sigma <= 0:
        raise ValueError('Finite equal-width matrices and positive bandwidth required')
    return np.exp(-np.sum((x[:, None] - y[None, :])**2, axis=-1)/(2*sigma*sigma))


def direct_stat(x, y, sigma):
    """Independent implementation of the approved formula, including dependent cross diagonal."""
    n = len(x)
    if n < 2 or len(y) != n:
        raise ValueError('At least two complete pairs required')
    xx, yy, xy = kernel(x, x, sigma), kernel(y, y, sigma), kernel(x, y, sigma)
    return ((xx.sum()-np.trace(xx))+(yy.sum()-np.trace(yy)))/(n*(n-1))-2*xy.sum()/n**2


def orbit(x, y, sigma, signs):
    """Exact algebra of original statistic under pair flips; no changed optimization/kernel."""
    n = len(x)
    signs = np.asarray(signs, float)
    if x.shape != y.shape or signs.ndim != 2 or signs.shape[1] != n or not np.isin(signs, [-1., 1.]).all():
        raise ValueError('Complete paired arrays and +/-1 signs required')
    xx, yy, xy = kernel(x, x, sigma), kernel(y, y, sigma), kernel(x, y, sigma)
    q = xx+yy-xy-xy.T
    np.fill_diagonal(q, 0.)
    coefficient = (2*n-1)/(2*n*n*(n-1))
    base = direct_stat(x, y, sigma)
    constant = base-coefficient*q.sum()
    scores = constant+coefficient*np.einsum('bi,ij,bj->b', signs, q, signs, optimize=False)
    return scores, float(constant)


def all_signs(n):
    if not 0 <= n <= 20:
        raise ValueError('Exact enumeration bounded to20 informative pairs')
    return 1.-2.*((np.arange(2**n, dtype=np.uint32)[:, None] >> np.arange(n)) & 1)


def pvalue(x, y, sigma, signs, exact=False):
    scores, center = orbit(x, y, sigma, signs)
    observed = direct_stat(x, y, sigma)
    count = np.count_nonzero(scores >= observed-1e-12)
    p = count/len(scores) if exact else (1+count)/(1+len(scores))
    return float(p), float(observed), center
