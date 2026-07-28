"""
Utility module containing helper functions and classes.
"""

# standard imports
import numpy as np

# type shorthands
float_or_array = float | np.ndarray[np.floating]
""" Either a single float number or an array of float numbers """
complex_or_array = complex | np.ndarray[np.complexfloating]
""" Either a single complex number or an array of complex numbers """
