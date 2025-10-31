"""
Module that provides global constants.
"""

# standard imports
import re
from astropy.units import Quantity

# general
FREE_SPACE_PERM = Quantity(8.8541878188e-12, "F/m")
""" Free space permittivity [F/m] """
AVOGADRO = Quantity(6.02214076e23, "1/mol")
""" Avogadro constant [1/mol] """
BOLTZMANN = Quantity(1.380649e-23, "J/K")
""" Boltzmann constant [J/K] """
GAS_CONSTANT = (AVOGADRO * BOLTZMANN).decompose()
""" Gas constant [J / K mol]"""
SPEED_OF_LIGHT = Quantity(299792458, "m/s")
""" Speed of light in vacuum [m/s] """
PLANCK = Quantity(6.62607015e-34, "J s")
""" Planck constant [J s]"""
E_CHARGE = Quantity(-1.60217663e-19, "C")
""" (Negative) electron charge [C] """
E_MASS = Quantity(9.1093837e-31, "kg")
""" Electron mass [kg] """

# Venus
VENUS_GRAV_PARAM = Quantity(3.24859e14, "m3/s2")
""" Venus gravitational parameter [m^3/s^2] """
VENUS_RADIUS = Quantity(6051800, "m")
""" Venus radius [m] """

# mission
VISAR_FREQUENCY = Quantity(7.9e9, "Hz")
""" Nominal VISAR radar frequency [Hz] """
VISAR_WAVELENGTH = (SPEED_OF_LIGHT / VISAR_FREQUENCY).decompose()
""" Nominal VISAR radar wavelength [m] """

# non-physical
HEADERPATTERN = re.compile(r"(.+) \[(.*)\]")
""" File header regex pattern to extract column name and unit """
