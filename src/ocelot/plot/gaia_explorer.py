"""A little class for live exploration of a Gaia dataset."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle, Circle
import sys
from datetime import datetime

from typing import Union, Optional
from ..cluster.preprocess import cut_dataset
from .utilities import calculate_alpha
from pathlib import Path


DEFAULT_BACKEND = matplotlib.get_backend()


def ion(backend="QT4Agg"):
    """Turns on interactive programming. Essential to use the GaiaExplorer, but called separately to avoid issues!"""
    plt.ion()
    matplotlib.use(backend)


def ioff():
    """Opposite of ion()"""
    plt.ioff()
    matplotlib.use(DEFAULT_BACKEND)


class GaiaExplorer:
    def __init__(self, data_gaia: pd.DataFrame, cluster_location: Union[pd.Series, dict],
                 extra_catalogue_to_plot: Optional[pd.DataFrame] = None,
                 extra_catalogue_cmap: Optional[dict] = None,
                 extra_catalogue_cmap_key: str = "source",
                 magnitude_range: Union[tuple, list, np.ndarray] = ((8, 18), (-1, 5)),
                 initial_guess_factor: float = 2.,
                 adaptive_alpha: bool = True,
                 error_regions_multiplier: Union[int, float] = 0,
                 error_regions_systematic_tol: Union[int, float] = 0,
                 fig_dir: Union[str, Path] = "figs",
                 debug: bool = False):
        """A little class for live exploration of a Gaia dataset.

        Args:
            data_gaia (pd.DataFrame): your standard Gaia dataframe with astrometry & magnitude info.
            cluster_location (pd.Series, dict): information on the location of the cluster to look for.
                Mandatory keys: 'name', 'ra', 'dec', 'pmra', 'pmdec', 'parallax'
                Optional keys for initial guess: 'ang_radius_t', 'pm_dispersion' or 'pmra_std' and 'pmdec_std',
                    'parallax_std'
            magnitude_range (list-like): magnitude range to plot.
                Default: (8, 18)
            initial_guess_factor (float): amount to multiply dispersions by when generating an initial guess zone. A bit
                like a zoom out factor.
                    Default: 2.0  (tends to be a pretty good value)
            adaptive_alpha (bool): whether or not to adaptively change the alpha value while moving around the plot.
                Default: True
            error_regions_multiplier (int or float): size (in sigma) of error regions to add. Only activates when
                greater than 0. If greater than 0, then 'ang_radius_t', 'pmra_error', 'pmdec_error' and 'parallax_error'
                are required keys.
                Default: 0  (off)
            error_regions_systematic_tol (int or float): an extra amount to add on to the error regions for when your
                crossmatching also has a linear systematic tolerance. Only applies to pmra/pmdec/parallax.
                Default: 0
            fig_dir (str or pathlib.Path): directory to save figures to when the user asks to save a figure.
                Default: 'figs'
            debug (bool): if True, print extra stuff to the console.
                Default: False
        """
        # Initial setup of the data
        self.data = data_gaia
        self.data['bp_rp'] = self.data['phot_bp_mean_mag'] - self.data['phot_rp_mean_mag']
        self.fig_dir = Path(fig_dir)

        # Check we have the minimum amount of information
        required_keys = ("name", "ra", "dec", "pmra", "pmdec", "parallax")
        if not np.all(np.isin(required_keys, tuple(cluster_location.keys()))):
            raise ValueError("initial cluster location must contain all 5 astrometric dimensions and a name!")

        # Setup error region highlighting
        self.error_regions_multiplier = error_regions_multiplier
        self.error_regions_systematic_tol = error_regions_systematic_tol
        if self.error_regions_multiplier > 0:
            required_keys = ("ang_radius_t", "pmra_error", "pmdec_error", "parallax_error")
            if not np.all(np.isin(required_keys, tuple(cluster_location.keys()))):
                raise ValueError("initial cluster location must contain the following keys to add error regions: "
                                 "'ang_radius_t', 'pmra_error', 'pmdec_error' and 'parallax_error'.")

        # Initial setup of the catalogue too
        self.literature_data = extra_catalogue_to_plot
        if self.literature_data is not None:
            # Check that we have something to use for the cmap key
            if extra_catalogue_cmap_key not in self.literature_data.keys():
                raise ValueError("Column not found: to decide how to colour code literature objects, a column with name"
                                 " 'extra_catalogue_cmap_key' must be in extra_catalogue_to_plot!")

            # Generate a cmap if one wasn't provided
            self.literature_data_cmap_key = extra_catalogue_cmap_key
            if extra_catalogue_cmap is None:
                unique_cmap_keys = np.unique(self.literature_data[self.literature_data_cmap_key])
                cmap = plt.get_cmap("tab10")(np.arange(len(unique_cmap_keys)))
                self.literature_data_cmap = {k: v for (k, v) in zip(unique_cmap_keys, cmap)}
            else:
                self.literature_data_cmap = extra_catalogue_cmap

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
        )
        self.magnitude_cuts = dict(
            phot_g_mean_mag=np.sort(magnitude_range[0]),
            bp_rp=np.sort(magnitude_range[1])
        )

        # We'll also want to consider some move and zoom speeds. The latter is just set to two (we'll double or halve
        # the zoom every time) but the former uses the dispersions etc to make an educated first guess.
        self.move_speed = np.asarray([
            [cluster_location["ang_radius_t"], cluster_location["ang_radius_t"]],  # ra / dec
            [cluster_location["pm_dispersion"], cluster_location["pm_dispersion"]],  # pmra / pmdec
            [cluster_location["ang_radius_t"], cluster_location["parallax_std"]],  # ra / parallax
        ])

        self.zoom_speed = [2, 2, 2]

        # alpha-value related info
        self.adaptive_alpha = adaptive_alpha
        self.alpha_multiplier = 1.
        self.current_alpha = 1.
        self.first_update_data_call = True

        # Other info we might need
        self.fig, self.ax, self.main_lines = None, None, None
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
        ax = ax.flatten()

        # Assign labels to everyone
        for an_ax, labels in zip(ax, self.axis_labels):
            an_ax.set(xlabel=labels[0], ylabel=labels[1])

        # Magnitudes are a special case
        ax[3].set(xlim=self.magnitude_cuts['bp_rp'], ylim=self.magnitude_cuts['phot_g_mean_mag'])
        ax[3].invert_yaxis()

        # Plot initial objects
        self.main_lines = []
        for an_ax in ax:
            self.main_lines.append(an_ax.plot([], [], 'ko', ms=3)[0])

        # Save to the class and add the initial data
        self.fig, self.ax = fig, ax
        self._add_catalogue_clusters()
        self._add_error_regions()
        self._update_data()
        self._update_active_axis(0)
        self._initialise_matplotlib_connections()
        plt.show()

    def _refresh_figure(self):
        """Refreshes the figure so that any change is displayed."""
        self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

    def _add_catalogue_clusters(self):
        """Call for updating the figure with locations of clusters as marked by other catalogues."""
        if self.literature_data is not None:
            if len(self.literature_data) > 0:
                # Cycle over every literature object
                for index_literature, a_row in self.literature_data.iterrows():
                    color = self.literature_data_cmap[a_row[self.literature_data_cmap_key]]

                    # Add it to every axis!
                    for i_ax in range(3):
                        x_value, y_value = a_row[self.axis_labels[i_ax][0]], a_row[self.axis_labels[i_ax][1]]
                        name = " ".join(a_row['name'].rsplit("_"))
                        self.ax[i_ax].plot(x_value, y_value, 'x', ms=5, color=color)  # , mec='w', mew=0.5)
                        self.ax[i_ax].text(
                            x_value, y_value, name, va='bottom', ha='center', color=color, fontsize=8, clip_on=True)
                        # .set_path_effects([patheffects.withStroke(linewidth=0.5, foreground='w')])

            self._refresh_figure()

    def _add_error_regions(self):
        """Adds error regions to the figure if requested by the user."""
        if self.error_regions_multiplier > 0:
            # Firstly, let's try and grab a colour for our circle if it's in the literature_data list and hence has
            # a consistent catalogue colour we could use
            if self.literature_data is not None:
                if np.all(np.isin(self.cluster_location['name'], self.literature_data['name'])):
                    first_match = np.asarray(self.literature_data['name'].to_numpy() == self.cluster_location['name']
                                             ).nonzero()[0][0]
                    color = self.literature_data_cmap[
                        self.literature_data.loc[first_match, self.literature_data_cmap_key]]
                else:
                    color = 'r'

            # Otherwise, just use red!
            else:
                color = 'r'

            # Add them to the plots!
            style = dict(fc='None', ec=color)

            # ra/dec
            location = (self.cluster_location['ra'], self.cluster_location['dec'])
            self.ax[0].add_patch(Circle(location, radius=self.cluster_location['ang_radius_t'], **style))

            # pmra/pmdec
            width = self.cluster_location['pmra_error'] * 2 * self.error_regions_multiplier \
                + 2 * self.error_regions_systematic_tol
            height = self.cluster_location['pmdec_error'] * 2 * self.error_regions_multiplier \
                + 2 * self.error_regions_systematic_tol
            location = (self.cluster_location['pmra'] - width / 2, self.cluster_location['pmdec'] - height / 2)
            self.ax[1].add_patch(Rectangle(location, width, height, **style))

            # ra/parallax
            width = self.cluster_location['ang_radius_t']
            height = self.cluster_location['parallax_error'] * 2 * self.error_regions_multiplier \
                + 2 * self.error_regions_systematic_tol
            location = (self.cluster_location['ra'] - width / 2, self.cluster_location['parallax'] - height / 2)
            self.ax[2].add_patch(Rectangle(location, width, height, **style))

    def _update_data(self):
        """Call for updating the figure with new cuts."""
        # Get the new data
        new_data = cut_dataset(self.data, parameter_cuts=self.current_cuts)

        # Get an alpha estimate. We just do this for the first one cos it'll stay the same
        if self.adaptive_alpha or self.first_update_data_call:
            self.current_alpha = calculate_alpha(self.fig, self.ax[0], len(new_data), 3)
            self.first_update_data_call = False

        # Apply our current alpha value to the data
        self.alpha_to_use = np.clip(self.current_alpha * self.alpha_multiplier, 0.01, 1.0)
        for i in range(4):
            self.main_lines[i].set_alpha(self.alpha_to_use)

        # Update all of the lines with the new stuff
        self.main_lines[0].set_data(new_data['ra'], new_data['dec'])
        self.main_lines[1].set_data(new_data['pmra'], new_data['pmdec'])
        self.main_lines[2].set_data(new_data['ra'], new_data['parallax'])
        self.main_lines[3].set_data(new_data['bp_rp'], new_data['phot_g_mean_mag'])

        # Change the limits toooo
        for i in range(3):
            self.ax[i].set(xlim=self.current_cuts[self.axis_labels[i][0]],
                           ylim=self.current_cuts[self.axis_labels[i][1]])

        # Add a title cos we cool
        self.fig.suptitle(f"The region around {self.cluster_name}\nstars: {len(new_data)}", x=0.1, y=0.98, ha='left')

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
        x_cuts = self.current_cuts[self.axis_labels[self.current_axis][0]]
        y_cuts = self.current_cuts[self.axis_labels[self.current_axis][1]]

        differences = np.asarray([np.diff(x_cuts)[0], np.diff(y_cuts)[0]])
        midpoints = np.asarray([np.mean(x_cuts), np.mean(y_cuts)])

        # We can compute the new difference we need
        if go_in:
            new_differences = differences / self.zoom_speed[self.current_axis] / 2
            self._update_move_speed(False)
        else:
            new_differences = differences * self.zoom_speed[self.current_axis] / 2
            self._update_move_speed(True)

        # Hard coded special case that we won't change ra values if we're editing the ra/parallax plot
        if self.current_axis == 2:
            i_start = 1
        else:
            i_start = 0

        # Finally, we can write this to the limits!
        for i in range(i_start, 2):
            self.current_cuts[self.axis_labels[self.current_axis][i]] = (
                midpoints[i] + np.asarray([-new_differences[i], new_differences[i]]))

        # ... and run the data updater
        self._update_data()

    def _change_location(self, key: str):
        """Changes the midpoint of the current axis, i.e. the area we're looking at."""
        # Firstly, we need to process the direction
        # w or s are up/down, i.e. y axis
        if key in ("up", "down"):
            current_axis_name = self.axis_labels[self.current_axis][1]
            is_y_axis = 1
        else:
            current_axis_name = self.axis_labels[self.current_axis][0]
            is_y_axis = 0

        # w or d mean there's an increase
        if key in ("up", "right"):
            direction = 1
        else:
            direction = -1

        # Now, we can add this to the limits! Easy pweezy. And update the data too
        self.current_cuts[current_axis_name] += direction * self.move_speed[self.current_axis, is_y_axis]
        self._update_data()

    def _change_alpha(self, increase: bool):
        """Changes the alpha value that we plot our data points with."""
        if increase:
            self.alpha_multiplier *= 1.5
        else:
            self.alpha_multiplier /= 1.5

        self._update_data()

    def _save_figure(self):
        """Saves the current view of the figure!"""
        # Make sure the folder exists and save to it!
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        date_string = f"{now.year}-{now.month:0>2}-{now.day:0>2}--{now.hour:0>2}-{now.minute:0>2}-{now.second:0>2}"
        fig_name = f"{self.cluster_name}_{date_string}.png"
        self.fig.savefig(self.fig_dir / fig_name, bbox_inches="tight")
        print(f"  saved figure {fig_name}")
        sys.stdout.flush()

    def _close_figure(self):
        plt.close(self.fig)

    def _initialise_matplotlib_connections(self):
        """Starts matplotlib connections that mean we call self._handle_keypress when a key is pressed on the window"""
        self.fig.canvas.mpl_connect('key_press_event', self._handle_keypress)

    def _handle_keypress(self, event):
        """Core function for updates of the figure element"""
        if self.debug:
            print("Key pressed:", event.key)
            sys.stdout.flush()

        # Change current axis being edited
        if event.key in ("1", "2", "3"):
            self._update_active_axis(int(event.key) - 1)

        # Change current axis move or zoom speed
        elif event.key in ("*", "/"):
            self._update_move_speed(event.key == "*")

        # Change current axis zoom level
        elif event.key in ("+", "-"):
            self._change_zoom(event.key == "+")

        # Change current axis center location
        elif event.key in ("up", "right", "down", "left"):
            self._change_location(event.key)

        # Change current alpha multiplier
        elif event.key in ("pageup", "pagedown"):
            self._change_alpha(event.key == "pageup")

        # Save the current figure
        elif event.key == 'end':
            self._save_figure()

        # Quit the figure
        elif event.key == "q":
            self._close_figure()

        # Or alternatively, key not supported
        else:
            if self.debug:
                print("  key not recognised")
                sys.stdout.flush()

    def __call__(self):
        """Plot a Gaia data chunk and let the user move around!"""
        self._generate_figure()
