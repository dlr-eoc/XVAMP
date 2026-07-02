"""
Utility module containing helper functions and classes.
"""

# standard imports
from typing import Any
import numpy as np
from numpy.typing import NDArray
from astropy.units import Quantity

# type shorthands
float_or_array = float | NDArray[np.double]
""" Either a single float number or an array of float numbers """
complex_or_array = complex | NDArray[np.cdouble]
""" Either a single complex number or an array of complex numbers """


def cast_to_np(input: Any | Quantity, unit: str) -> Any | NDArray[np.floating]:
    """
    Convert a :class:`~astropy.units.Quantity` into a NumPy array of [unit],
    or simply return the input if it's not a :class:`~astropy.units.Quantity`.
    """
    try:
        return input.to_value(unit)
    except AttributeError:
        return input
