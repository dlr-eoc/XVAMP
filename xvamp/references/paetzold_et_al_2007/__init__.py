"""
Module that provides the reference profiles from :cite:t:`patzold2007`.
"""

# standard imports
import numpy as np
from importlib.resources import files as res_files
from astropy.units import Quantity, Unit

# package imports
from ...utils.interpolate import BoundedInterpolatingBasis
from ...utils.io import read_unit_csv
from ...profile import MultiProfile

# data content
DOYS = [196, 200, 202, 212, 218, 233, 234, 239]
""" Day-of-years of the electron density profiles """
TABLES = [
    f"fig{t}-doy{d}"
    for t, d in zip(["4a", "4b", "4b", "4b", "5a", "5b", "5c", "5d"], DOYS)
]
""" Table numbers to load """

# data format information
SZAS = Quantity([50, 56, 59, 80, 92.4, 113.0, 113.4, 113.5], "°")
""" Solar zenith angles [°] of the elctron density profiles """
UNIT_EL_DENSITY = Unit("1/m3")
""" Output unit of the electron density profile [1/m3]"""

# get local installation folder paths
datafolder = res_files()
""" Resolved current folder location """
# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}
""" Loaded tables """


# get function that returns the first 6 electron density profiles as a MultiProfile
def _get_el_densities() -> MultiProfile:
    # get unique altitude levels
    el_altitude = np.unique(
        np.concatenate([tables[t]["Altitude"].value for t in TABLES[:6]])
    )
    # interpolate all data onto common altitudes
    all_densities = np.zeros((el_altitude.size, 6))
    for i, t in enumerate(TABLES[:6]):
        all_densities[:, i] = np.clip(
            np.interp(
                el_altitude,
                tables[t]["Altitude"].value,
                tables[t]["Electron density"].to(UNIT_EL_DENSITY).value,
                left=0,
                right=None,
            ),
            a_min=0,
            a_max=None,
        )
    # return as MultiProfile
    return MultiProfile(
        index=el_altitude,
        index_unit=tables[t]["Altitude"].unit,
        data=all_densities,
        data_units=UNIT_EL_DENSITY,
    )


# create electron density MultiProfile
el_density = _get_el_densities()
""" Electron density profiles at different solar zenith angles """


# build inter- and extrapolator for first 6 angles
sza_interpolating_basis = BoundedInterpolatingBasis(
    0, 180, knots=SZAS[:6].to("°").value
)
"""
Interpolating basis for the well-resolved SZA angles between 50° and 113°,
returns a 2D array when called with each column representing the different
knots
"""


def interpolate_el_density(sza: Quantity["angle"]) -> Quantity:
    """
    Interpolate the electron density profiles from solar zenith angles
    between 50° and 113°.

    Parameters
    ----------
    sza
        Solar zenith angle [°]

    Returns
    -------
        Electron density profile at all altitudes for different SZA
    """
    sza_coefs = sza_interpolating_basis(sza.to("°").value)
    intp_density = Quantity(
        np.einsum(
            "ij,kj->ik",
            el_density.data,
            sza_coefs,
        ),
        el_density.unit,
    )
    return intp_density
