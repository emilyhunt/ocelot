"""Various simple statistical utility functions."""

import numpy as np


def variable_bin_histogram(values, min, max, minimum_width, minimum_size=5):
    """Computes a variably binned histogram."""
    if minimum_size * 2 >= len(values):
        raise ValueError(
            "minimum_size must be no more than half the size of values to create a "
            "histogram with at least two bins. However, values is only "
            f"{len(values)} long, compared to minimum_size which is {minimum_size}."
            "Consider increasing the size of your input dataset or reducing "
            "minimum_size."
        )

    # First pass
    n_bins = int(np.round((max - min) / minimum_width)) + 1
    bins = np.linspace(min, max, num=n_bins)

    count, _ = np.histogram(values, bins=bins)

    # Early return condition if all bins are fine / minimum_size is zero
    if minimum_size == 0 or np.all(count >= minimum_size):
        return count, bins

    # Error check that should stop any bins from ever not being filled
    if np.sum(count) < minimum_size:
        raise ValueError(
            "Unable to fill bins due to fewer than minimum_size items in total"
        )

    # Otherwise, loop over all values, removing items until we get the desired bin
    # occupancies
    index = 0
    bins, count = list(bins), list(count)
    while index < len(count) - 2:
        if count[index] < minimum_size:
            count[index] += count[index + 1]
            count.pop(index + 1)
            bins.pop(index + 1)
        else:
            index += 1

    # Handle the final bin as a special case (since we have to go in reverse)
    while count[-1] < minimum_size:
        index = len(count) - 1
        count[index] += count[index - 1]
        count.pop(index - 1)
        bins.pop(index - 1)

    return np.asarray(count), np.asarray(bins)


def calculate_bin_centers(bin_edges):
    """Calculates the centers of a binned histogram."""
    return (bin_edges[:-1] + bin_edges[1:]) / 2
