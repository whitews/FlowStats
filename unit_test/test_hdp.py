import unittest
import numpy as np
from matplotlib import pyplot, gridspec
from flowstats import cluster


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

    def test_hdp(self):
        n_data_sets = 5
        n_clusters = 16
        n_iterations = 2
        burn_in = 100

        data_sets = self.generate_data(n_data_sets)
        figure = pyplot.figure(figsize=(6, 12))
        gs = gridspec.GridSpec(n_data_sets, 1, width_ratios=(2,3))

        for i, d in enumerate(data_sets):
            pyplot.subplot(gs[i])
            pyplot.axis([-15, 15, -15, 15])
            pyplot.plot(
                d[:, 0],
                d[:, 1],
                ls="*",
                marker='.'
            )

        model = cluster.HDPMixtureModel(n_clusters, n_iterations, burn_in)

        # results = model.fit(data_sets, None, seed=123, munkres_id=True)
        # results_averaged = results.average()
        # results_modal = results_averaged.make_modal()

        pyplot.tight_layout()
        pyplot.show()


if __name__ == '__main__':
    unittest.main()