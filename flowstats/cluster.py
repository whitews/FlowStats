"""
Created on Oct 30, 2009

@author: jolly
"""

import numpy as np
from numpy.random import multivariate_normal
from numpy.random import seed as np_seed
from datetime import datetime

from dpmix_exp import DPNormalMixture, BEMNormalMixture, HDPNormalMixture

from .dp_cluster import DPCluster, DPMixture, HDPMixture


class DPMixtureModel(object):
    """
    Fits a Dirichlet process (DP) mixture model to a Numpy data set.
    """

    def __init__(
            self,
            n_clusters,
            n_iterations,
            burn_in,
            model='dp'):
        """
        n_clusters = number of clusters to fit
        n_iterations = number of MCMC iterations to sample
        burn_in = number of MCMC burn-in iterations
        """
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.model = model

        self.data = None
        self.cdp = None

        self.m_0 = None
        self.alpha_0 = 1
        self.nu_0 = None
        self.phi_0 = None

        self.prior_mu = None
        self.prior_sigma = None
        self.prior_pi = None

        self.e0 = 5
        self.f0 = 0.1

        self.parallel = False

        self._prior_mu = None
        self._prior_pi = None
        self._prior_sigma = None
        self._ref = None

        self._load_mu = None
        self._load_sigma = None
        self._load_pi = None

        self.mu_d = None

        self.m = None
        self.s = None
        self.d = None
        self.n = None

    def load_mu(self, mu):
        if len(mu.shape) > 2:
            raise ValueError('Shape of Mu is wrong')
        if len(mu.shape) == 2:
            (n, d) = mu.shape
        else:
            n = 1
            d = mu.shape[0]
        if n > self.n_clusters:
            raise ValueError(
                'Number of proposed Mus greater then number of clusters')

        self.prior_mu = mu
        self.mu_d = d
        self._load_mu = True

    def _load_mu_at_fit(self):
        (n, d) = self.prior_mu.shape
        if d != self.d:
            raise ValueError('Dimension mismatch between Mus and Data')

        elif n < self.n_clusters:
            self._prior_mu = np.zeros((self.n_clusters, self.d))
            self._prior_mu[0:n, :] = (self.prior_mu.copy() - self.m) / self.s
            self._prior_mu[n:, :] = multivariate_normal(
                np.zeros((self.d,)),
                np.eye(self.d),
                self.n_clusters - n)
        else:
            self._prior_mu = (self.prior_mu.copy() - self.m) / self.s

    def load_sigma(self, sigma):
        n, _ = sigma.shape[0:2]
        if len(sigma.shape) > 3:
            raise ValueError('Shape of Sigma is wrong')

        if len(sigma.shape) == 2:
            sigma = np.array(sigma)

        if sigma.shape[1] != sigma.shape[2]:
            raise ValueError("Sigmas must be square matrices")

        if n > self.n_clusters:
            raise ValueError(
                'Number of proposed Sigmas greater then number of clusters')

        self._load_sigma = True
        self.prior_sigma = sigma

    def _load_sigma_at_fit(self):
        n, d = self.prior_sigma.shape[0:2]

        if d != self.d:
            raise ValueError('Dimension mismatch between Sigmas and Data')

        elif n < self.n_clusters:
            self._prior_sigma = np.zeros((self.n_clusters, self.d, self.d))
            self._prior_sigma[0:n, :, :] = (self.prior_sigma.copy()) / np.outer(
                self.s, self.s)
            for i in range(n, self.n_clusters):
                self._prior_sigma[i, :, :] = np.eye(self.d)
        else:
            self._prior_sigma = (self.prior_sigma.copy()) / np.outer(
                self.s,
                self.s)

    def load_pi(self, pi):
        tmp = np.array(pi)
        if len(tmp.shape) != 1:
            raise ValueError("Shape of pi is wrong")
        n = tmp.shape[0]
        if n > self.n_clusters:
            raise ValueError(
                'Number of proposed Pis greater then number of clusters')

        if np.sum(tmp) > 1:
            raise ValueError('Proposed Pis sum to more than 1')
        if n < self.n_clusters:
            self._prior_pi = np.zeros(self.n_clusters)
            self._prior_pi[0:n] = tmp
            left = (1.0 - np.sum(tmp)) / (self.n_clusters - n)
            for i in range(n, self.n_clusters):
                self._prior_pi[i] = left
        else:
            self._prior_pi = tmp

        self._load_pi = True

    def load_ref(self, ref):
        self._ref = ref

    def _load_ref_at_fit(self, points):
        if isinstance(self._ref, DPMixture):
            self.prior_mu = self._ref.mus
            self.prior_sigma = self._ref.sigmas
            self.prior_pi = self._ref.pis
        else:
            self.prior_mu = np.zeros((self.n_clusters, points.shape[1]))
            self.prior_sigma = np.zeros(
                (self.n_clusters, points.shape[1], points.shape[1]))
            for i in range(self.n_clusters):
                try:
                    self.prior_mu[i] = np.mean(points[self._ref == i], 0)
                    self.prior_sigma[i] = np.cov(
                        points[self._ref == i],
                        rowvar=0)
                except Exception as e:
                    print(e)
                    self.prior_mu[i] = np.zeros(points.shape[1])
                    self.prior_sigma[i] = np.eye(points.shape[1])

            tot = float(points.shape[0])
            self.prior_pi = np.array(
                [
                    points[self._ref == i].shape[0] / tot for i in
                    range(self.n_clusters)
                ]
            )

    def fit(
            self,
            data,
            device,
            seed=None,
            verbose=False,
            normed=False,
            munkres_id=False,
            gamma=10,
            callback=None
    ):
        if isinstance(data, list) or isinstance(data, tuple):
            return [
                self._fit(
                    i,
                    device,
                    seed=seed,
                    verbose=verbose,
                    normed=normed,
                    munkres_id=munkres_id,
                    gamma=gamma) for i in data
            ]
        else:
            return self._fit(
                data,
                device,
                seed=seed,
                verbose=verbose,
                normed=normed,
                munkres_id=munkres_id,
                gamma=gamma,
                callback=callback
            )

    def _fit(
            self,
            data,
            device,
            seed=None,
            verbose=False,
            normed=False,
            munkres_id=False,
            gamma=10,
            callback=None
    ):
        """
        Fit the mixture model to the data
        use get_results() to get the fitted model
        """
        points = data.copy().astype('double')
        if normed:
            self.data = points
            self.m = np.zeros(self.data.shape[1])
            self.s = np.ones(self.data.shape[1])
        else:
            self.m = points.mean(0)
            self.s = points.std(0)
            # in case any of the std's are zero
            if type(self.s) == np.float64:
                if self.s == 0:
                    self.s = 1
            else:
                self.s[self.s == 0] = 1
            self.data = (points - self.m) / self.s

        if len(self.data.shape) == 1:
            self.data = self.data.reshape((self.data.shape[0], 1))

        if len(self.data.shape) != 2:
            raise ValueError("points array is the wrong shape")
        self.n, self.d = self.data.shape

        if self._ref is not None:
            munkres_id = True
            self._load_ref_at_fit(points)

        if self.prior_mu is not None:
            self._load_mu_at_fit()
        if self.prior_sigma is not None:
            self._load_sigma_at_fit()

        if seed:
            np_seed(seed)
        else:
            np_seed(datetime.now().microsecond)

        # TODO move hyper-parameter settings here
        if self.model.lower() == 'bem':
            self.cdp = BEMNormalMixture(
                self.data,
                ncomp=self.n_clusters,
                gamma0=gamma,
                m0=self.m_0,
                nu0=self.nu_0,
                Phi0=self.phi_0,
                e0=self.e0,
                f0=self.f0,
                mu0=self._prior_mu,
                Sigma0=self._prior_sigma,
                weights0=self._prior_pi,
                alpha0=self.alpha_0,
                parallel=self.parallel,
                verbose=verbose
            )
            self.cdp.optimize(self.n_iterations, device=device)
        else:
            self.cdp = DPNormalMixture(
                self.data,
                ncomp=self.n_clusters,
                gamma0=gamma,
                m0=self.m_0,
                nu0=self.nu_0,
                Phi0=self.phi_0,
                e0=self.e0,
                f0=self.f0,
                mu0=self._prior_mu,
                Sigma0=self._prior_sigma,
                weights0=self._prior_pi,
                alpha0=self.alpha_0,
                parallel=self.parallel,
                verbose=verbose)
            self.cdp.sample(
                niter=self.n_iterations,
                nburn=self.burn_in,
                thin=1,
                ident=munkres_id,
                device=device,
                callback=callback
            )

        if self.model.lower() == 'bem':
            results = []
            for j in range(self.n_clusters):
                tmp = DPCluster(
                    self.cdp.weights[j],
                    (self.cdp.mu[j] * self.s) + self.m,
                    self.cdp.Sigma[j] * np.outer(self.s, self.s),
                    self.cdp.mu[j],
                    self.cdp.Sigma[j]
                )
                results.append(tmp)
            tmp = DPMixture(results, niter=1, m=self.m, s=self.s)
        else:
            results = []
            for i in range(self.n_iterations):
                for j in range(self.n_clusters):
                    tmp = DPCluster(
                        self.cdp.weights[i, j],
                        (self.cdp.mu[i, j] * self.s) + self.m,
                        self.cdp.Sigma[i, j] * np.outer(
                            self.s, self.s),
                        self.cdp.mu[i, j],
                        self.cdp.Sigma[i, j]
                    )
                    results.append(tmp)
            tmp = DPMixture(
                results,
                self.n_iterations,
                self.m,
                self.s,
                munkres_id)
        return tmp


class HDPMixtureModel(DPMixtureModel):
    """
    n_clusters = number of clusters to fit
    n_iterations = number of MCMC iterations
    burn_in = number of MCMC burn-in iterations
    """

    def __init__(self, n_clusters, n_iterations, burn_in):
        super(HDPMixtureModel, self).__init__(
            n_clusters,
            n_iterations,
            burn_in,
            model='hdp'
        )

        self.e0 = 1.0
        self.f0 = 1.0
        self.g0 = 0.1
        self.h0 = 0.1

        self.n_data_sets = None
        self.hdp = None

    def load_pi(self, pi):
        """
        load_pi is not implemented in HDPMixtureModel since it's shape
        must be verified against the data sets used in the fit method.
        To specify initial weights use the keyword argument in the fit method.

        Note: load_mu and load_sigma can be called just as they would in a
              DPMixtureModel instance
        :param pi:
        :return: NotImplementedError
        """
        raise NotImplementedError("Initial weights should be set in fit()")

    def fit(
            self,
            data_sets,
            device,
            seed=None,
            verbose=False,
            munkres_id=False,
            tune_interval=100,
            initial_weights=None,
            gamma=10,
            callback=None
    ):
        self.d = data_sets[0].shape[1]

        data_sets = [i.copy().astype('double') for i in data_sets]
        self.n_data_sets = len(data_sets)
        total_data = np.vstack(data_sets)
        self.m = np.mean(total_data, 0)
        self.s = np.std(total_data, 0)
        standardized = []
        for i in data_sets:
            if i.shape[1] != self.d:
                raise RuntimeError("Shape of data sets do not match")
            standardized.append(((i - self.m) / self.s))

        if self.prior_mu is not None:
            self._load_mu_at_fit()
        if self.prior_sigma is not None:
            self._load_sigma_at_fit()
        if initial_weights is not None:
            if initial_weights.shape[0] != self.n_data_sets:
                raise ValueError(
                    "Initial weights do not match the number of data sets"
                )
            if initial_weights.shape[1] != self.n_clusters:
                raise ValueError(
                    "Initial weights do not match the number of components"
                )
            self._prior_pi = initial_weights

        if seed is not None:
            np_seed(seed)
        else:
            np_seed(datetime.now().microsecond)

        self.hdp = HDPNormalMixture(
            standardized,
            ncomp=self.n_clusters,
            gamma0=gamma,
            m0=self.m_0,
            nu0=self.nu_0,
            Phi0=self.phi_0,
            e0=self.e0,
            f0=self.f0,
            g0=self.g0,
            h0=self.h0,
            mu0=self._prior_mu,
            Sigma0=self._prior_sigma,
            weights0=self._prior_pi,
            alpha0=self.alpha_0,
            parallel=self.parallel,
            verbose=verbose)
        if not device:
            self.hdp.gpu = False
        self.hdp.sample(
            niter=self.n_iterations,
            nburn=self.burn_in,
            thin=1,
            ident=munkres_id,
            tune_interval=tune_interval,
            device=device,
            callback=callback
        )

        pis = np.array(
            [
                self.hdp.weights[-self.n_iterations:, k, :].flatten()
                for k in range(self.n_data_sets)
            ]
        )
        mus = (
            self.hdp.mu[-self.n_iterations:].reshape(
                self.n_clusters * self.n_iterations,
                self.d
            ) * self.s + self.m
        )
        sigmas = (
            self.hdp.Sigma[-self.n_iterations:].reshape(
                self.n_clusters * self.n_iterations,
                self.d,
                self.d
            ) * np.outer(self.s, self.s)
        )
        return HDPMixture(
            pis,
            mus,
            sigmas,
            self.n_iterations,
            self.m,
            self.s,
            munkres_id
        )
