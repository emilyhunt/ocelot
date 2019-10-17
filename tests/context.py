""" Allows the tests directory to be separate to the main directory."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ocelot

x = ocelot.version
