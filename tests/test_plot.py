"""A set of tests for use with the pytest module, covering ocelot.plot"""

# FUCKING HATE PYTHON IMPORTS AAAA
# (the below fixes this though)
try:
    from .context import ocelot
except ModuleNotFoundError:
    print('Unable to find ocelot via .context! Trying to import from your python path instead...')
try:
    import ocelot
except ModuleNotFoundError:
    raise ModuleNotFoundError('Unable to find ocelot')

from pathlib import Path

# And now... everything else

# Path towards the test isochrones
max_label = 7
path_to_test_isochrones = Path('./test_data/isochrones.dat')




