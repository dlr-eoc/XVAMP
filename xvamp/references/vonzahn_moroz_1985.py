"""
Module that provides some data from :cite:t:`vonzahn1985`.
"""

# standard imports
from astropy.units import Quantity, dimensionless_unscaled, km

# package imports
from ..profile import Profile

# the frequently-used constant values of the Venusian atmosphere
CO2_MR = Quantity(0.965, dimensionless_unscaled)
""" CO2 mixing ratio [-] """
N2_MR = Quantity(0.035, dimensionless_unscaled)
""" N2 mixing ratio [-] """
AR_MR = Quantity(7e-5, dimensionless_unscaled)
""" Argon mixing ratio [-] """

# convenient access as Profiles
co2_molar_fraction = Profile(
    index=0.0,
    index_unit=km,
    data=CO2_MR.to("ppm"),
    lower=None,
    upper=None,
)
""" CO2 molar fraction profile from :cite:t:`vonzahn1985` """
n2_molar_fraction = Profile(
    index=0.0,
    index_unit=km,
    data=N2_MR.to("ppm"),
    lower=None,
    upper=None,
)
""" N2 molar fraction profile from :cite:t:`vonzahn1985` """
ar_molar_fraction = Profile(
    index=0.0,
    index_unit=km,
    data=AR_MR.to("ppm"),
    lower=None,
    upper=None,
)
""" Argon molar fraction profile from :cite:t:`vonzahn1985` """
