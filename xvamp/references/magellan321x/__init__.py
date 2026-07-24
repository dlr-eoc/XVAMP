"""
Reference class that provides the profiles from the Magellan orbits no. 3212,
3213, and 3214 from :cite:t:`jenkins1996` and :cite:t:`jenkins1996a` as
processed by :cite:t:`jenkins1994`.
"""

# standard imports
import numpy as np
from importlib.resources import files as res_files
from typing import Tuple
from astropy.table import QTable

# package imports
from ...utils.io import read_unit_fwf_desc
from ...profile import Profile

# data content
TABLES = ["mgn_abs", "mgn_rtpd"]
""" Table numbers to load """

# data format information
DESC_ABS = [
    ("WAVELENGTH", "U1", 3, ""),
    ("ORBIT_NUMBER", "i", 5, ""),
    ("ALTITUDE", "f8", 7, "km"),
    ("ABSORPTIVITY", "f8", 9, "dB/km"),
    ("ABSORP_DEV", "f8", 8, "dB/km"),
    ("H2SO4_VOLMIX", "f8", 6, "ppm"),
    ("H2SO4_VM_DEV", "f8", 6, "ppm"),
    ("LATITUDE", "f8", 7, "°"),
    ("LONGITUDE", "f8", 8, "°"),
    ("ZENITH_ANGLE", "f8", 8, "°"),
    ("LOCAL_TIME", "f8", 7, "h"),
    ("ERT", "f8", 10, "s"),
]
""" Data format description of the ``mgn_abs.dat`` file """
DESC_RTPD = [
    ("WAVELENGTH", "U1", 3, ""),
    ("ORBIT_NUMBER", "i", 5, ""),
    ("ALTITUDE", "f8", 7, "km"),
    ("REFRACTIVITY", "f8", 9, "Nunit"),
    ("REFRACT_DEV", "f8", 6, "Nunit"),
    ("TEMPERATURE", "f8", 7, "K"),
    ("TEMP_DEV", "f8", 6, "K"),
    ("PRESSURE", "f8", 9, "bar"),
    ("PRESS_DEV", "f8", 9, "bar"),
    ("DENSITY", "f8", 8, "kg/m3"),
    ("DENS_DEV", "f8", 8, "kg/m3"),
    ("LATITUDE", "f8", 7, "°"),
    ("LONGITUDE", "f8", 8, "°"),
    ("ZENITH_ANGLE", "f8", 8, "°"),
    ("LOCAL_TIME", "f8", 7, "h"),
    ("ERT", "f8", 10, "s"),
]
""" Data format description of the ``mgn_rtpd.dat`` file """
STR_CONVERTER = {0: lambda s: s.strip()}
""" Convenience converter to strip whitespace from the wavelength field """

# get local installation folder paths
datafolder = res_files()
""" Resolved current folder location """
# load raw data files
tables = {
    "mgn_abs": read_unit_fwf_desc(datafolder / "mgn_abs.dat", DESC_ABS, STR_CONVERTER),
    "mgn_rtpd": read_unit_fwf_desc(
        datafolder / "mgn_rtpd.dat", DESC_ABS, STR_CONVERTER
    ),
}
""" Loaded tables """


def get_wavelength_orbit(
    wavelength: str | None = None, orbit: int | None = None
) -> Tuple[QTable, QTable]:
    """
    Return the dataset for a specific wavelength and/or orbit.

    Parameters
    ----------
    wavelength
        Either ``"S"`` or ``"X"`` band. ``None`` returns all wavelengths.
    orbit
        One of ``3212``, ``3213`` and ``3214``. ``None`` returns all orbits.

    Returns
    -------
    mgn_abs
        Subset of the ``mgn_abs.dat`` file
    mgn_rtpd
        Subset of the ``mgn_rtpd.dat`` file
    """
    # default to all
    i_abs = np.ones(len(tables["mgn_abs"]), dtype=bool)
    i_rtpd = np.ones(len(tables["mgn_rtpd"]), dtype=bool)
    # restrict wavelength
    if wavelength is not None:
        i_abs = np.logical_and(i_abs, tables["mgn_abs"]["WAVELENGTH"] == wavelength)
        i_rtpd = np.logical_and(i_rtpd, tables["mgn_rtpd"]["WAVELENGTH"] == wavelength)
    # restrict orbit
    if orbit is not None:
        i_abs = np.logical_and(i_abs, tables["mgn_abs"]["ORBIT_NUMBER"] == orbit)
        i_rtpd = np.logical_and(i_rtpd, tables["mgn_rtpd"]["ORBIT_NUMBER"] == orbit)
    # subset data
    return tables["mgn_abs"][i_abs], tables["mgn_rtpd"][i_rtpd]


# quick function to get the H2SO4 mixing ratio for X-band
def _extract_xband_h2so4_profile(orbit: int) -> Tuple[Profile, Profile, Profile]:
    sub_abs = tables["mgn_abs"][
        np.logical_and(
            tables["mgn_abs"]["WAVELENGTH"] == "X",
            tables["mgn_abs"]["ORBIT_NUMBER"] == orbit,
        )
    ]
    return Profile(
        index=sub_abs["ALTITUDE"],
        data=np.clip(sub_abs["H2SO4_VOLMIX"].value, a_min=0, a_max=None),
        data_unit=sub_abs["H2SO4_VOLMIX"].unit,
    )


# create profiles
h2so4_mr_x_3212 = _extract_xband_h2so4_profile(3212)
""" H2SO4 mixing ratio from Magellan orbit 3212 and X-band """
h2so4_mr_x_3213 = _extract_xband_h2so4_profile(3213)
""" H2SO4 mixing ratio from Magellan orbit 3213 and X-band """
h2so4_mr_x_3214 = _extract_xband_h2so4_profile(3214)
""" H2SO4 mixing ratio from Magellan orbit 3214 and X-band """
