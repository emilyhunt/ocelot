"""A little class for live exploration of a Gaia dataset."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union
from ..cluster import cut_dataset


class GaiaExplorer:
    def __init__(self, data_gaia: pd.DataFrame, cluster_location: Union[pd.Series, dict],
                 magnitude_range: Union[tuple, list, np.ndarray] = (8, 18),
                 initial_guess_factor: float = 2.,
                 allow_return: bool = False,
                 debug: bool = False):
        """A little class for live exploration of a Gaia dataset.

        Args:
            data_gaia (pd.DataFrame): your standard Gaia dataframe with astrometry & magnitude info.
            cluster_location (pd.Series, dict): information on the location of the cluster to look for.
                Mandatory keys: 'name', 'ra', 'dec', 'pmra', 'pmdec', 'parallax'
                Optional keys: 'ang_radius_t', 'pm_dispersion' or 'pmra_std' and 'pmdec_std', 'parallax_std'
            magnitude_range (list-like): magnitude range to plot.
                Default: (8, 18)
            initial_guess_factor (float): amount to multiply dispersions by when generating an initial guess zone. A bit
                like a zoom out factor.
                    Default: 2.0  (tends to be a pretty good value)
            allow_return (bool): whether or not to prompt the user to return a value upon quit.
                Default: False
            debug (bool): if True, print extra stuff to the console.
                Default: False
        """
        self.data = data_gaia
        self.data['bp-rp'] = self.data['phot_bp_mean_mag'] - self.data['phot_rp_mean_mag']

        # Check we have the minimum amount of information
        if not np.all(np.isin(("name", "ra", "dec", "pmra", "pmdec", "parallax"), cluster_location.keys())):
            raise ValueError("initial cluster location must contain all 5 astrometric dimensions and a name!")

        # Guess initial limit information if it isn't given
        # Firstly, guess the angular radius: assume 20pc if none given, or a minimum of 0.1 degrees
        if "ang_radius_t" not in cluster_location.keys():
            cluster_location["ang_radius_t"] = np.clip(
                np.arctan(20 / (1000 / cluster_location['parallax'])) * 180 / np.pi, 0.1, np.inf)

        # Simple pm dispersion guess of 2 mas/yr if none given
        if "pm_dispersion" not in cluster_location.keys():
            if np.all(np.isin(("pmra_std", "pmdec_std"), cluster_location.keys())):
                cluster_location["pm_dispersion"] = np.sqrt(
                    cluster_location["pmra_std"]**2 + cluster_location["pmdec_std"]**2)
            else:
                cluster_location["pm_dispersion"] = 2.

        # Simple 0.5 mas parallax guess if none given
        if "parallax_std" not in cluster_location.keys():
            cluster_location["parallax_std"] = 0.5

        # Finally, write this initial limit guess to the class
        guess_array = np.asarray([-initial_guess_factor, initial_guess_factor])
        self.cluster_location = cluster_location
        self.current_cuts = dict(
            ra=guess_array * cluster_location["ang_radius_t"] + cluster_location['ra'],
            dec=guess_array * cluster_location["ang_radius_t"] + cluster_location['dec'],
            pmra=guess_array * cluster_location["pm_dispersion"] + cluster_location['pmra'],
            pmdec=guess_array * cluster_location["pm_dispersion"] + cluster_location['pmdec'],
            parallax=guess_array * cluster_location["parallax_std"] + cluster_location['parallax'],
            phot_g_mean_mag=np.sort(magnitude_range)
        )

        # We'll also want to consider some move and zoom speeds. The latter is just set to two (we'll double or halve
        # the zoom every time) but the former uses the dispersions etc to make an educated first guess.
        self.move_speed = [
            np.asarray([cluster_location["ang_radius_t"], cluster_location["ang_radius_t"]]),  # ra / dec
            np.asarray([cluster_location["pm_dispersion"], cluster_location["pm_dispersion"]]),  # pmra / pmdec
            np.asarray([cluster_location["ang_radius_t"], cluster_location["parallax_std"]]),  # ra / parallax
        ]

        self.zoom_speed = [2, 2, 2]

        # Other info we might need
        self.fig, self.ax = None, None
        self.allow_return = allow_return
        self.cluster_name = cluster_location["name"]
        self.current_axis = 0
        self.axis_labels = (
            ("ra", "dec"),
            ("pmra", "pmdec"),
            ("ra", "parallax"),
            ("BP - RP", "G"),
        )

        self.debug = debug

    def _generate_figure(self):
        """Generates a figure for the class to use with four different panels."""
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

        # Assign labels to everyone
        for an_ax, labels in zip(ax, self.axis_labels):
            an_ax.set(xlabel=labels[0], ylabel=labels[1])

        # Magnitudes are a special case
        ax[3].invert_yaxis()

        # Plot initial objects
        for an_ax in ax:
            an_ax.plot([], [], 'ko', ms=3)

        # Save to the class
        self.fig, self.ax = fig, ax

    def _refresh_figure(self):
        """Refreshes the figure so that any change is displayed."""
        self.fig.canvas.flush_events()

    def _update_data(self):
        """Call for updating the figure with new cuts."""
        # Get the new data
        new_data = cut_dataset(self.data, parameter_cuts=self.current_cuts)

        # Update all of the lines with the new stuff
        self.ax[0].lines[0].set_data(new_data['ra'], new_data['dec'])
        self.ax[1].lines[0].set_data(new_data['pmra'], new_data['pmdec'])
        self.ax[2].lines[0].set_data(new_data['ra'], new_data['parallax'])
        self.ax[3].lines[0].set_data(new_data['bp-rp'], new_data['phot_g_mean_mag'])

        self._refresh_figure()

    def _update_active_axis(self, new_axis: int):
        """Updates which axis is the active one, both internally and with visual cues."""
        # Firstly, reset the colour of the old one
        an_ax = self.ax[self.current_axis]
        for a_spine in an_ax.spines:
            an_ax.spines[a_spine].set_color('k')

        # Now, change the active axis
        self.current_axis = new_axis

        # ... and change the colour of the new one
        an_ax = self.ax[self.current_axis]
        for a_spine in an_ax.spines:
            an_ax.spines[a_spine].set_color('r')

        self._refresh_figure()

    def _update_move_speed(self, go_faster: bool):
        """Changes the speed at which we move around the current axis."""
        if go_faster:
            self.move_speed[self.current_axis] *= 2
        else:
            self.move_speed[self.current_axis] /= 2

    def _change_zoom(self, go_in: bool):
        """Changes the zoom level on the current axis."""
        # Firstly, let's get the current difference between x and y values
        differences = np.asarray([
            np.diff(self.current_cuts[self.axis_labels[self.current_axis][0]])[0],
            np.diff(self.current_cuts[self.axis_labels[self.current_axis][1]])[0]
        ])

        # We can compute the new difference we need
        if go_in:
            new_differences = differences / self.zoom_speed[self.current_axis]
        else:
            new_differences = differences * self.zoom_speed[self.current_axis]

        # Finally, we can write this to the limits, with a hard coded special case that we won't change ra values if
        # we're editing the ra/parallax plot
        pass

    def _change_location(self, direction: str):
        pass

    def _close_figure(self):
        plt.close(self.fig)

    def _initialise_matplotlib_connections(self):
        """Starts matplotlib connections that mean we call self._handle_keypress when a key is pressed on the window"""
        self.fig.canvas.mpl_connect('key_press_event', self._handle_keypress)

    def _handle_keypress(self, event):
        """Core function for updates of the figure element"""
        if self.debug:
            print("Key pressed: ", event.key)

        # Change current axis being edited
        if event.key in ("1", "2", "3"):
            self._update_active_axis(int(event.key) - 1)

        # Change current axis move or zoom speed
        elif event.key in ("t", "g"):
            self._update_move_speed(event.key == "t")

        # Change current axis zoom level
        elif event.key in ("r", "f"):
            self._change_zoom(event.key == "r")

        # Change current axis center location
        elif event.key in ("w", "a", "s", "d"):
            self._change_location(event.key)

        # Quit the figure
        elif event.key == "q":
            self._close_figure()

        # Or alternatively, figure not supported
        else:
            if self.debug:
                print("  key not recognised")

    def __call__(self):
        """Plot a Gaia data chunk and let the user move around!"""
        self._generate_figure()
        self._update_data()
