import unittest
import numpy as np
from matplotlib import pyplot, gridspec
from matplotlib.patches import Ellipse
from flowstats import cluster
import datetime


class HDPMixtureModelTestCase(unittest.TestCase):
    @staticmethod
    def generate_data(n_data_sets, n_data_points=10000):
        data_sets = list()
        for i in range(0, n_data_sets):
            distribution_0 = np.random.multivariate_normal(
                [-10.0, -5.0],
                [[1, 0], [0, 5]],
                (n_data_points/2,))
            distribution_1 = np.random.multivariate_normal(
                [5.0, 0.5],
                [[3, 1.5], [1.5, 2]],
                (n_data_points/2,))

            data_sets.append(np.vstack([distribution_0, distribution_1]))

        return data_sets

    @staticmethod
    def calculate_ellipse(center_x, center_y, covariance_matrix, n_std_dev=3):
        values, vectors = np.linalg.eigh(covariance_matrix)
        # order = values.argsort()[::-1]
        # values = values[order]
        # vectors = vectors[:, order]

        theta = np.degrees(np.arctan2(*vectors[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * n_std_dev * np.sqrt(values)

        ellipse = Ellipse(
            xy=[center_x, center_y],
            width=width,
            height=height,
            angle=theta
        )

        return ellipse

    def test_hdp(self):
        n_data_sets = 10
        n_clusters = 4
        n_iterations = 50
        burn_in = 50

        data_sets = self.generate_data(n_data_sets)

        model = cluster.HDPMixtureModel(n_clusters, n_iterations, burn_in)

        time_0 = datetime.datetime.now()
        results = model.fit(data_sets, True, seed=123, munkres_id=True, verbose=True)
        time_1 = datetime.datetime.now()

        delta_time = time_1 - time_0
        print delta_time.total_seconds()

        results_averaged = results.average()
        results_modal = results_averaged.make_modal()

        # there should be n_clusters of rows in both mus and sigmas
        mus_averaged = results_averaged.mus
        sigmas_averaged = results_averaged.sigmas
        self.assertEqual(mus_averaged.shape[0], sigmas_averaged.shape[0])

        # calculate our ellipses from the averaged results
        ellipses = list()
        for i, mu in enumerate(mus_averaged):
            ellipse = self.calculate_ellipse(mu[0], mu[1], sigmas_averaged[i])
            ellipse.set_alpha(0.3)
            ellipse.set_facecolor((1, 0, 0))
            ellipse.set_edgecolor((0, 0, 0))
            ellipses.append(ellipse)

        figure = pyplot.figure(figsize=(6, 6))

        ax = figure.gca()
        pyplot.axis([-15, 15, -15, 15])
        pyplot.plot(
            data_sets[0][:, 0],
            data_sets[0][:, 1],
            ls="*",
            marker='.',
            alpha=0.1
        )
        for j, e in enumerate(ellipses):
            if results.pis[0][j] > 0.01:
                ax.add_artist(e)

        pyplot.tight_layout()
        pyplot.show()


if __name__ == '__main__':
    unittest.main()