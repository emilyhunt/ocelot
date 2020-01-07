"""Some little friends to help with various odd jobs in OCELOT's plotting submod."""

import numpy as np


def calculate_alpha(fig, ax, n_points: int, marker_radius: int, dpi: int = 300, desired_max_density: float = 0.03):
    """Calculates the optimum alpha value for points to make the plot not be
    over-saturated. Relies on the area value being correctly calculated
    prior to input, which should be the value in inches for the figure. It also
    assumes that all points are uniformly distributed across the figure area
    (which they often are not), so the input area number may need tweaking.

    Args:
        fig (matplotlib figure): the figure element. Required to calculate alpha values precisely.
        ax (matplotlib axis): the figure element. Required to calculate alpha values precisely.
        n_points (int): the number of points to plot.
        marker_radius (px): the radius of the marker. Note that plt.scatter()
            specifies marker *area* but plt.plot() uses marker size, which is
            analogous to the marker radius.
        dpi (int): output resolution of the figure, which will scale the area
            parameter. In units of dots per inch. Default: 300.
        desired_max_density (float): tweakable parameter that was found to
            produce the best results. Simply mutliplying the area value is
            probably easier for normal function use, though. Default: 0.03.

    Returns:
        float: A recommended alpha value to use on these axes.

    Todo: could still consider if the data are clumped or not

    """
    # We calculate the area of the figure with some fancy magic
    # See https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    area = bbox.width * bbox.height

    # We aim to have a maximum density of uniform_max_density, assuming that
    # n_points are distributed evenly across the area.
    marker_area = n_points * np.pi * marker_radius ** 2  # In units of px
    total_area = area * dpi ** 2
    current_max_density = marker_area / total_area

    return np.clip(desired_max_density / current_max_density, 0., 1.)


def generate_figure_title():
    # Todo: a helper function to quickly grab information from... somewhere... and make a nicely formatted figure title, including all information of the run.
    pass


"""
# TODO: this function sucks. get rid of it
def add_text(my_axis, x: float, y: float, text: Union[str, List[str]], ha: str = 'center', va: str = 'center',
             mode: str = 'relative', separator: str = '\n', fontsize: str = 'small', ):
    ""Adds text to an axis at the specified location. Has lots of convenience use-cases, as adding text to an axis is
    often an absolute pain in the backside! Mirrors the axis.text method as much as possible, but with a few extra
    features:
    1. text can be a list of strings, separated by separator
    2. the text is automatically plotted in a mode relative to the axis
    3. the box is by default

    Args:
        my_axis (matplotlib.axis): axis to add text to.
        x (float): x location of the text to add.
        y (float): y location of the text to add.
        ha (string): the horizontal alignment of the text, ala matplotlib.
        va (string): the vertical alignment of the text, ala matplotlib.
        text (string, list of strings): text to add to the axis. If a list, it will join the list first for you.
            How kind? =)
        mode (string): the mode to using when choosing where to put the text. Allowed values: 'relative' (default) or
            'absolute'.

    Returns:
        The modified axis.
    ""
    # Join the list of text together if the input is a list
    if type(text) is list:
        text = separator.join(text)

    if style_box is None:
        dict(boxstyle='round', facecolor='w', alpha=0.8))

    # Put the text in a different spot depending on where the user specified
    if mode == 'relative':
        my_axis.text(x, y, text,
                     ha=ha, va=va, transform=my_axis.transAxes,
                     fontsize=fontsize,
                     bbox=dict(boxstyle='round', facecolor='w', alpha=0.8))
    elif mode == 'absolute':
        my_axis.text(x, y, text,
                     ha=ha, va=va,
                     fontsize=fontsize,
                     bbox=dict(boxstyle='round', facecolor='w', alpha=0.8))

    pass
"""
