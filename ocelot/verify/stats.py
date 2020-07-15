"""Statistical tools (like tests and kth nn distribution tools) for use in verification of OCs."""

import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import norm, chi2, kstest

from ocelot.cluster.epsilon import kth_nn_distribution, constrained_a


class KthNNDistributionCDF:
    def __init__(self, a, d, k, grid_size=200, limits=(0, 1)):
        """Precalculates a grid of kth nn distribution CDF values for a given a, d and k."""
        # Precalculate cumulatively summed values!
        x_range = np.linspace(*limits, num=grid_size)
        y_range = np.cumsum(kth_nn_distribution(x_range, a, d, k))

        # Normalise these values so that the max val is just 1!
        y_range /= np.max(y_range)

        # Interpolate this and set it to __call__!
        self.cdf_function = interp1d(x_range, y_range, kind="linear", fill_value=(0., 1.), bounds_error=False)

    def __call__(self, x):
        return self.cdf_function(x)


class KthNNDistributionPDF:
    def __init__(self, a, d, k, grid_size=200, limits=(0, 1)):
        """Precalculates a grid of kth nn distribution PDF values for a given a, d and k."""
        # Precalculate cumulatively summed values!
        x_range = np.linspace(*limits, num=grid_size)
        y_range = kth_nn_distribution(x_range, a, d, k)

        # Normalise these values so that the max val is just 1!
        y_range /= np.trapz(y_range, x=x_range)

        # Interpolate this and set it to __call__!
        self.pdf_function = interp1d(x_range, y_range, kind="linear", fill_value=(0., 0.), bounds_error=False)

    def __call__(self, x):
        return self.pdf_function(x)


class _CurveToFit:
    def __init__(self, k):
        self.k = k

    def __call__(self, x_range, *params):
        """A version of the Chandrasekhar1943 curve for use with scipy's curve_fit function.

        Args:
            x_range: should *already* be cumulatively summed and sorted wrt point number!
            params: (a, dimension)

        Returns:
            y_range of points evaluated at x_range values!

        """
        y_func = np.cumsum(kth_nn_distribution(x_range, params[0], params[1], self.k))
        y_func /= np.trapz(y_func, x=x_range)

        return y_func


def _fit_kth_nn_distribution(nn_distances: dict, min_samples: int, resolution: int = 200):
    """Fits a kth nn distribution model to a dict of arrays of nn distances."""
    result = {}

    for a_cluster in nn_distances:

        # Sort nn distances and make an array of point numbers
        x_data = np.sort(nn_distances[a_cluster])
        y_data = np.arange(1, len(nn_distances[a_cluster]) + 1)

        # Interpolate this function into something more valid for fitting
        y_interpolator = interp1d(x_data, y_data, kind="linear")

        x_range = np.linspace(x_data.min(), x_data.max(), num=resolution)
        y_range = y_interpolator(x_range)

        # Normalise
        y_range /= np.trapz(y_range, x=x_range)

        # Fit a function!
        # Get starting guesses
        k = min_samples
        d_0 = min_samples / 2  # Might want a better estimate sometimes!
        a_0 = constrained_a(d_0, k, np.mean(nn_distances[a_cluster]))

        # Minimize - we suppress warnings because it's quite common for the curve_fit method to go into disallowed areas
        # sometimes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            minimisation_result, covariance = curve_fit(
                _CurveToFit(k), x_range, y_range, p0=np.asarray([a_0, d_0]), method="trf",
                bounds=([0, 0], [np.inf, np.inf]),
                verbose=False)

        result[a_cluster] = (minimisation_result[0], minimisation_result[1], k)

    # Return values
    return result


def _likelihood_of_object(x_data, a, d, k, cdf_resolution=200):
    # Get decent estimates of the range of the function we should work with
    limits = (0.0, x_data.max() * 2)

    # Make a cdf!
    func = KthNNDistributionCDF(a, d, k, limits=limits, grid_size=cdf_resolution)

    # Evaluate all the points at the cdf!
    return np.sum(np.log(func(x_data)))


def _convert_one_sided_p_value_to_z_score(p_value):
    # np.abs is to stop -0.0 from happening. Only 0.0 is allowed!
    return np.abs(-norm.ppf(p_value / 2))


def _likelihood_ratio_test(nn_distances, cluster_fit_params, field_fit_params,
                           cdf_resolution=200):
    """Finds the maximum likelihood of the cluser & field models and does a simple comparison between the two
    to see which one is a better fit.

    See: http://rnowling.github.io/machine/learning/2017/10/07/likelihood-ratio-test.html

    """
    significances, log_likelihoods = {}, {}

    for a_cluster in nn_distances:

        # Grab likelihoods
        likelihood_cluster = _likelihood_of_object(nn_distances[a_cluster], *cluster_fit_params[a_cluster],
                                                   cdf_resolution=cdf_resolution)
        likelihood_field = _likelihood_of_object(nn_distances[a_cluster], *field_fit_params[a_cluster],
                                                 cdf_resolution=cdf_resolution)

        # Log and square the ratio!
        log_likelihoods[a_cluster] = 2 * (likelihood_cluster - likelihood_field)

        # Convert to a pval, then to a sigma significance - using a weird formula because it gives me better floating pt.
        # precision, and dividing by 2 since our test is implicitly one-sided. I quite like this explanation of why:
        # https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-are-the-differences-between-one-tailed-and-two-tailed-tests/
        # Basically, in our one tailed test we're only interested in the likelihood of the cluster being better than the
        # field. We don't care about the field being better than the cluster: either way, that's a Z=0.0 not worth our
        # consideration as an OC candidate.
        p_val = chi2.sf(log_likelihoods[a_cluster], (len(nn_distances[a_cluster]) - 1) * (len(cluster_fit_params) - 1))
        significances[a_cluster] = _convert_one_sided_p_value_to_z_score(p_val)

    return significances, log_likelihoods


def _ks_test(cluster_nn_distances: dict, field_nn_distances: dict, one_sample: bool = True,
             cdf_resolution: int = 200):
    """Performs a KS test (one-sided) of cluster stars and field stars to see if they're compatible or not.

    Args:
        cluster_nn_distances (np.ndarray): dict of arrays of cluster nearest neighbour distances of shape (n_stars,)
        field_nn_distances (np.ndarray): dict of arrays of len(3) of field fit parameters if one_sample==True, or dict
            of arrays of of field nearest neighbour distances otherwise.
        one_sample (bool): whether or not to do the test in one sample mode. Can produce slightly more reliable results
            (since the number of field stars will stop being a factor) but requires that the fitting procedure was
            correct.
            Default: True
        cdf_resolution (int): resolution to sample the cdf at if one_sample=True.
            Default: 200

    Returns:
        significance value of the clusters and the KS test statistics in a dict.

    """
    significances, ks_test_statistics = {}, {}

    for a_cluster in cluster_nn_distances:
        # Turn the field_nn_distances into a callable function instead of a CDF if one_sample==True.
        if one_sample:
            limits = (0.0, np.max(cluster_nn_distances[a_cluster]) * 2)
            field_distribution = KthNNDistributionCDF(
                *field_nn_distances[a_cluster], grid_size=cdf_resolution, limits=limits)
        else:
            field_distribution = field_nn_distances[a_cluster]

        # Do the KS test, get a p-value, go home
        ks_test_statistics[a_cluster], p_value = kstest(
            cluster_nn_distances[a_cluster], field_distribution, alternative="greater")

        significances[a_cluster] = _convert_one_sided_p_value_to_z_score(p_value)

    return significances, ks_test_statistics