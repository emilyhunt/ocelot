"""
2019-10-11
A test of whether or not a spline can reasonably/easily be fit to data from a typical isochrone. The major problem with
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

import ocelot

# Read in the isochrones
path_to_isochrones = Path('../../../data/isochrones/191015_isochrones_z-2_8_to_1_2')
isochrones = ocelot.isochrone.read_cmd_isochrone(path_to_isochrones, max_label=3)

# Define some constants for which one we want
logAge = 9.5
MH = 0.0
tolerance = 0.01
scatter = 0.15
point_multiplier = 2  # How much to tile the existing isochrones by
plot_resolution = 100

# ####################
# Input & perturbing of the data
# Grab the stars we want & remove everything else
wrong_age = np.where(isochrones['logAge'] != logAge, True, False)
wrong_z = np.logical_or(np.where(isochrones['MH'] > MH + tolerance, True, False),
                        np.where(isochrones['MH'] < MH - tolerance, True, False))
stars_to_drop = np.logical_or(wrong_age, wrong_z)
one_isochrone = isochrones.drop(labels=np.where(stars_to_drop)[0], axis='index')

# Increase the number of points we have to play with by making copies
isochrone_colour = one_isochrone['G_BP-RP'].to_numpy()
isochrone_magnitude = one_isochrone['Gmag'].to_numpy()
data_colour = np.tile(isochrone_colour, 5)
data_magnitude = np.tile(isochrone_magnitude, 5)

# Randomly perturb these points
np.random.seed(42)
noise_colour = np.random.normal(scale=scatter, size=data_colour.size)
noise_magnitude = np.random.normal(scale=scatter, size=data_magnitude.size)
data_colour += noise_colour
data_magnitude += noise_magnitude

# Make up some weights, even though we know the numbers here (we assume a clustering algorithm could do this very well)
combined_weight = (noise_colour**2 + noise_magnitude**2)**(-1/2)  # (1 / ...) makes low noise == high weight
normalised_combined_weight = combined_weight / combined_weight.max()

# Sort them so that the interpolator is happier
sort_args = np.argsort(data_magnitude)
data_colour = data_colour[sort_args]
data_magnitude = data_magnitude[sort_args]

# ####################
# Fit a... thing
# Calculate some appropriate knot points by finding a sequence of ten mean points
#knots = np.linspace(0.01, 0.99, 20)
#parameter = np.linspace(0, 1, data_magnitude.size)
#magnitude_interpolator = LSQUnivariateSpline(parameter, data_magnitude, knots, k=3)
#colour_interpolator = LSQUnivariateSpline(parameter, data_colour, knots, k=3)

# Calculate the difference between consecutive points
x_difference = np.diff(data_colour)
y_difference = np.diff(data_magnitude)

# Calculate the Euclidean distance between each point
distance_between_points = np.sqrt(x_difference**2 + y_difference**2)
cumulative_distance_between_points = np.cumsum(distance_between_points)

# Add a zero at the start because diff reduces the length by 1
cumulative_distance_between_points = np.insert(cumulative_distance_between_points, 0, 0)

# Normalise the distances to the range [0, 1] if desired (this makes later use a lot easier)
parameter = (cumulative_distance_between_points
             / np.max(cumulative_distance_between_points))

# parameter = np.linspace(0, 1, data_magnitude.size)

magnitude_interpolator = UnivariateSpline(parameter, data_magnitude,
                                          w=normalised_combined_weight,
                                          s=0.15, k=3)
colour_interpolator = UnivariateSpline(parameter, data_colour,
                                       w=normalised_combined_weight,
                                       s=0.15, k=3)

# Interpolate some points
interpolated_parameter = np.linspace(0.001, 0.999, plot_resolution)
interpolated_magnitude = magnitude_interpolator(interpolated_parameter)
interpolated_colour = colour_interpolator(interpolated_parameter)

# ####################
# Plot this to test the results
plt.plot(isochrone_colour, isochrone_magnitude, 'r', label='original isochrone')
plt.plot(data_colour, data_magnitude, 'k.', alpha=0.2, label='random stars')
plt.plot(interpolated_colour, interpolated_magnitude, 'b-', label='interpolated stars')
plt.gca().invert_yaxis()
plt.legend()
plt.show()


"""
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
"""
