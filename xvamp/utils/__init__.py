"""
Utility module containing helper functions and classes.
"""

# standard imports
import numpy as np
from numpy.typing import NDArray

# type shorthands
float_or_array = float | NDArray[np.double]
""" Either a single float number or an array of float numbers """
complex_or_array = complex | NDArray[np.cdouble]
""" Either a single complex number or an array of complex numbers """
