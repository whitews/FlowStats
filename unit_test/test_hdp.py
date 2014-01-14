import unittest
import numpy as np
from matplotlib import pyplot
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
        order = values.argsort()[::-1]
        values = values[order]
        vectors = vectors[:, order]

        theta = np.degrees(np.arctan2(*vectors[:, 0][::-1]))

        # make all angles positive
        if theta < 0:
            theta += 360

        # Width and height are "full" widths, not radius
        width, height = 2 * n_std_dev * np.sqrt(values)

        ellipse = Ellipse(
            xy=[center_x, center_y],
            width=width,
            height=height,
            angle=theta
        )

        ellipse.set_alpha(0.3)
        ellipse.set_facecolor((1, 0, 0))

        return ellipse

    def test_hdp(self):
        n_data_sets = 3
        n_clusters = 16
        n_iterations = 4
        burn_in = 100

        figure_size = (16, 12)
        pis_threshold = 0.05

        data_sets = self.generate_data(n_data_sets)

        model = cluster.HDPMixtureModel(n_clusters, n_iterations, burn_in)

        time_0 = datetime.datetime.now()

        results = model.fit(
            data_sets,
            True,
            seed=123,
            munkres_id=True,
            verbose=True
        )

        time_1 = datetime.datetime.now()

        delta_time = time_1 - time_0
        print delta_time.total_seconds()

        # pis are split by data set, then iteration
        pis = np.array_split(results.pis, n_data_sets)
        for i, p in enumerate(pis):
            pis[i] = np.array_split(pis[i][0], n_iterations)

        # mus and sigmas are split by iteration
        mus = np.array_split(results.mus, n_iterations)
        sigmas = np.array_split(results.sigmas, n_iterations)

        # Get averaged results
        results_averaged = results.average()
        pis_averaged = np.array_split(results_averaged.pis, n_data_sets)
        mus_averaged = results_averaged.mus
        sigmas_averaged = results_averaged.sigmas

        # plot each iteration and the averaged result for each data set
        fig = pyplot.figure(figsize=figure_size)
        n_rows = n_data_sets
        n_columns = n_iterations + 1
        i_plot = 1  # plot iterator
        for i in range(n_data_sets):
            print "set: %d" % i

            for j in range(n_iterations):

                ax = fig.add_subplot(
                    n_rows,
                    n_columns,
                    i_plot,
                    aspect='equal'
                )
                i_plot += 1

                pyplot.plot(
                    data_sets[i][:, 0],
                    data_sets[i][:, 1],
                    ls="*",
                    marker=".",
                    alpha=0.1
                )
                pyplot.axis([-15, 15, -15, 15])

                print "\titeration: %d" % j

                for k in range(n_clusters):
                    if pis[i][j][k] > pis_threshold:
                        ellipse = self.calculate_ellipse(
                            mus[j][k][0],
                            mus[j][k][1],
                            sigmas[j][k]
                        )
                        ax.add_artist(ellipse)
                        print "\t\tclust: %02d, xy: (%.2f, %.2f), " \
                            "angle: %.2f, weight: %.3f" % (
                                k,
                                ellipse.center[0],
                                ellipse.center[1],
                                ellipse.angle,
                                pis[i][j][k]
                            )

            # now show the averaged results on this data set
            print "\taveraged:"
            ax = fig.add_subplot(
                n_rows,
                n_columns,
                i_plot,
                aspect='equal'
            )
            i_plot += 1

            pyplot.plot(
                data_sets[i][:, 0],
                data_sets[i][:, 1],
                ls="*",
                marker=".",
                alpha=0.1
            )
            pyplot.axis([-15, 15, -15, 15])

            for k in range(n_clusters):
                if pis_averaged[i][0][k] > pis_threshold:
                    ellipse = self.calculate_ellipse(
                        mus_averaged[k][0],
                        mus_averaged[k][1],
                        sigmas_averaged[k]
                    )
                    ax.add_artist(ellipse)
                    print "\t\tclust: %02d, xy: (%.2f, %.2f), " \
                        "angle: %.2f, weight: %.3f" % (
                            k,
                            ellipse.center[0],
                            ellipse.center[1],
                            ellipse.angle,
                            pis_averaged[i][0][k]
                        )

        pyplot.show()

        # there should be n_clusters of rows in both mus and sigmas
        self.assertEqual(mus_averaged.shape[0], sigmas_averaged.shape[0])


if __name__ == '__main__':
    unittest.main()