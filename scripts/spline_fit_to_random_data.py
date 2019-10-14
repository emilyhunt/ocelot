"""
2019-10-11
A test of whether or not a spline can reasonably/easily be fit to data from a typical isochrone. The major problem with
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

import ocelot

# Read in the isochrones
path_to_isochrones = Path('../../../data/isochrones/191008_isochrones_wide_logz_and_logt.dat')
isochrones = ocelot.isochrone.read_cmd_isochrone(path_to_isochrones, max_label=3)

# Define some constants for which one we want
logAge = 9.5
logZini = -1.7384
tolerance = 0.01
scatter = 0.3
point_multiplier = 5  # How much to tile the existing isochrones by
plot_resolution = 100

# ####################
# Input & perturbing of the data
# Grab the stars we want & remove everything else
wrong_age = np.where(isochrones['logAge'] != logAge, True, False)
wrong_z = np.logical_or(np.where(isochrones['logZini'] > logZini + tolerance, True, False),
                        np.where(isochrones['logZini'] < logZini - tolerance, True, False))
stars_to_drop = np.logical_or(wrong_age, wrong_z)
one_isochrone = isochrones.drop(labels=np.where(stars_to_drop)[0], axis='index')

# Increase the number of points we have to play with by making copies
isochrone_colour = one_isochrone['G_BP-RP'].to_numpy()
isochrone_magnitude = one_isochrone['Gmag'].to_numpy()
data_colour = np.tile(isochrone_colour, 5)
data_magnitude = np.tile(isochrone_magnitude, 5)

# Randomly perturb these points
np.random.seed(42)
data_colour += np.random.normal(scale=scatter, size=data_colour.size)
data_magnitude += np.random.normal(scale=scatter, size=data_magnitude.size)
sort_args = np.argsort(data_magnitude)
data_colour = data_colour[sort_args]
data_magnitude = data_magnitude[sort_args]

# ####################
# Find the middle point of the line
# First, grab all the stars within centre_finding_tolerance magnitudes of the median magnitude
centre_finding_tolerance = 0.1
central_magnitude = np.median(data_magnitude)
stars_to_find_centre_with = np.where(np.abs(data_magnitude - central_magnitude) < centre_finding_tolerance)[0]
data_magnitude_central = data_magnitude[stars_to_find_centre_with]
data_colour_central = data_colour[stars_to_find_centre_with]

# Then, find the star closest to the median colour in this window
central_colour = np.median(data_colour_central)
central_star = stars_to_find_centre_with[np.argmin(np.abs(data_colour_central - central_colour))]

# ####################
# Fit a... thing
# Calculate some appropriate knot points by finding a sequence of ten mean points
knots = np.linspace(0.01, 0.99, 20)
parameter = np.linspace(0, 1, data_magnitude.size)
magnitude_interpolator = LSQUnivariateSpline(parameter, data_magnitude, knots, k=3)
colour_interpolator = LSQUnivariateSpline(parameter, data_colour, knots, k=3)

# Interpolate some points
interpolated_parameter = np.linspace(0.01, 0.99, plot_resolution)
interpolated_magnitude = magnitude_interpolator(interpolated_parameter)
interpolated_colour = colour_interpolator(interpolated_parameter)

# ####################
# Plot this to test the results
plt.plot(isochrone_colour, isochrone_magnitude, 'r', label='original isochrone')
plt.plot(data_colour, data_magnitude, 'k.', alpha=0.2, label='random stars')
plt.plot(interpolated_colour, interpolated_magnitude, 'b-', label='interpolated stars')
plt.plot(data_colour[central_star], data_magnitude[central_star], 'cs', ms=4)
plt.gca().invert_yaxis()
plt.legend()
plt.show()
