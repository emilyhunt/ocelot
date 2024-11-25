import importlib.util
import os
from pathlib import Path

__version__ = "0.4.0"

# Set a library path, helping other modules to find the data directory etc.
MODULE_PATH = Path(importlib.util.find_spec("ocelot").origin).parent

# Set a data path where large files are downloaded to
DATA_PATH = MODULE_PATH.parent.parent / "data"
if environment_path := os.getenv("OCELOT_DATA"):
    DATA_PATH = Path(environment_path)
