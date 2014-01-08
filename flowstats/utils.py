from itertools import chain
from collections import deque
import numpy as np
from numpy.linalg import inv, solve
from scipy.spatial.distance import pdist as _pdist, squareform

from distributions import compmixnormpdf, mixnormpdf, mixnormrnd


def bfs(g, start):
    """BFS generator from start node"""
    queue, enqueued = deque([(None, start)]), {start}
    while queue:
        parent, n = queue.popleft()
        yield parent, n
        new = set(g[n]) - enqueued
        enqueued |= new
        queue.extend([(n, child) for child in new])


def flatten(list_of_lists):
    """Flatten one level of nesting"""
    return chain.from_iterable(list_of_lists)


def nodes(collection):
    """Return unique nodes"""
    return frozenset(node for node in flatten(collection) if node is not None)


def find_components(g):
    """Find connected components in graph g"""
    return np.unique(frozenset([nodes(bfs(g, i)) for i in range(len(g))]))


def matrix_to_graph(m):
    """Convert adjacency matrix to dictionary form of graph"""
    graph = {}
    for i, row in enumerate(m):
        graph[i] = np.nonzero(m[i, :])[0]
    return graph


def point_distribution(x, weight=None, scale=False):
    """
    Returns the distance matrix of points x. If scale is true,
    rescale distance by sqrt(number of dimensions).  If w is provided,
    it weights the original matrix of points *before* calculating the
    distance for efficiency rather than weighting of distances.
    """
    n, p = x.shape
    if weight is not None:
        weight = np.sqrt(weight)
        print weight
        print x
        x = x * weight
        print x
    if scale:
        return (1.0 / np.sqrt(p)) * squareform(_pdist(x, 'euclidean'))
    else:
        return squareform(_pdist(x, 'euclidean'))


def mode_search(
        pis,
        mus,
        sigmas,
        tol=1e-6,
        max_iterations=20,
        delta=0.1,
        w=None,
        scale=False):
    """
    Find the modes of a gaussian mixture
    """

    modes_dict, sm, unused_spm = _mode_search(
        pis,
        mus,
        sigmas,
        nk=0,
        tol=tol,
        max_iterations=max_iterations)

    m = np.array([i[0] for i in modes_dict.values()])
    xs = np.zeros((len(modes_dict.keys()), mus.shape[1]))

    # use stored index as dict items are not ordered
    for j, key in enumerate(modes_dict):
        cur_mode = tuple(m[j, :].tolist())
        xs[key[0], :] = cur_mode

    dm = point_distribution(xs, scale=scale, weight=w) < delta
    cs = find_components(matrix_to_graph(dm))
    cs = sorted(cs, key=len, reverse=True)

    result = {}
    modes = {}
    for i, j in enumerate(cs):
        modes[i] = np.mean(np.vstack([xs[k, :] for k in j]), 0)
        result[i] = tuple(j)

    return modes, result


def _mode_search(pi, mu, sigma, nk=0, tol=0.000001, max_iterations=20):
    """Search for modes in mixture of Gaussians"""
    k, unused_p = mu.shape
    omega = np.copy(sigma)
    a = np.copy(mu)

    for j in range(k):
        omega[j] = inv(sigma[j])
        a[j] = solve(sigma[j], mu[j])

    if nk > 0:
        all_x = np.concatenate([mu, mixnormrnd(pi, mu, sigma, nk)])
    else:
        all_x = np.copy(mu)
    all_px = mixnormpdf(all_x, pi, mu, sigma, use_gpu=False)
    nk += k

    modes_dict = {}  # modes
    sm = []  # starting point of mode search
    spm = []  # density at starting points

    etol = np.exp(tol)

    for js in range(nk):
        x = all_x[js]
        px = all_px[js]
        sm.append(x)
        spm.append(px)
        h = 0
        eps = 1 + etol

        while (h <= max_iterations) and (eps > etol):
            w = compmixnormpdf(x, pi, mu, sigma, use_gpu=False)
            orig_y = np.sum([w[j] * omega[j] for j in range(k)], 0)
            yy = np.dot(w, a)
            y = solve(orig_y, yy)
            py = mixnormpdf(y, pi, mu, sigma, use_gpu=False)
            eps = py / px
            x = y
            px = py
            h += 1

        modes_dict[(js, tuple(x))] = [x, px]  # eliminate duplicates

    return modes_dict, sm, spm