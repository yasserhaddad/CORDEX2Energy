"""
CORDEX to Energy (cordex2energy)

A Python package for processing CORDEX climate data and energy conversion analysis for Switzerland.
Converts climate scenarios into actionable energy insights.
"""

__version__ = "0.1.0"

# Import main submodules
from . import conversion_energy
from . import data_processing
from . import results
from . import utils

__all__ = [
    "conversion_energy",
    "data_processing",
    "results",
    "utils",
]
