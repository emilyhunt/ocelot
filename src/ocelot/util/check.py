"""Utilities for checking that things have the correct shape, etc."""


def _check_matching_lengths_of_non_nones(*args):
    """Checks that all non-None arguments have the same length."""
    if len(args) <= 1:
        return

    args_excluding_nones = [arg for arg in args if arg is not None]
    first_length = len(args_excluding_nones[0])
    for i, length in enumerate(args_excluding_nones[1:]):
        if length != first_length:
            raise ValueError(
                f"Length of argument {i+1} does not match length of first element"
            )
