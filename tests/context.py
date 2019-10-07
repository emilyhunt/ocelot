# -*- coding: utf-8 -*-
# Allows the tests directory to be separate to the main directory.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ocelot
