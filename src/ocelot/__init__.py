import importlib.util
from pathlib import Path

__version__ = "0.4.0"

# Set a library path, helping other modules to find the data directory etc.
_module_path = Path(importlib.util.find_spec("ocelot").origin).parent
_data_path = _module_path / "data"
