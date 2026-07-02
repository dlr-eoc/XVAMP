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
from .. import data

# data location
BASEFOLDER = "magellan321x"
""" Base folder for data """
TABLES = ["mgn_abs", "mgn_rtpd"]
""" Table numbers to load """

# data format information
DESC_ABS = [
    ("WAVELENGTH", "", "U1", 3),
    ("ORBIT_NUMBER", "", "i", 5),
    ("ALTITUDE", "km", "f8", 7),
    ("ABSORPTIVITY", "dB/km", "f8", 9),
    ("ABSORP_DEV", "dB/km", "f8", 8),
    ("H2SO4_VOLMIX", "ppm", "f8", 6),
    ("H2SO4_VM_DEV", "ppm", "f8", 6),
    ("LATITUDE", "°", "f8", 7),
    ("LONGITUDE", "°", "f8", 8),
    ("ZENITH_ANGLE", "°", "f8", 8),
    ("LOCAL_TIME", "h", "f8", 7),
    ("ERT", "s", "f8", 10),
]
""" Data format description of the ``mgn_abs.dat`` file """
DESC_RTPD = [
    ("WAVELENGTH", "", "U1", 3),
    ("ORBIT_NUMBER", "", "i", 5),
    ("ALTITUDE", "km", "f8", 7),
    ("REFRACTIVITY", "Nunit", "f8", 9),
    ("REFRACT_DEV", "Nunit", "f8", 6),
    ("TEMPERATURE", "K", "f8", 7),
    ("TEMP_DEV", "K", "f8", 6),
    ("PRESSURE", "bar", "f8", 9),
    ("PRESS_DEV", "bar", "f8", 9),
    ("DENSITY", "kg/m3", "f8", 8),
    ("DENS_DEV", "kg/m3", "f8", 8),
    ("LATITUDE", "°", "f8", 7),
    ("LONGITUDE", "°", "f8", 8),
    ("ZENITH_ANGLE", "°", "f8", 8),
    ("LOCAL_TIME", "h", "f8", 7),
    ("ERT", "s", "f8", 10),
]
""" Data format description of the ``mgn_rtpd.dat`` file """
STR_CONVERTER = {0: lambda s: s.strip()}
""" Convenience converter to strip whitespace from the wavelength field """


# get local installation folder paths
datafolder = res_files(data) / BASEFOLDER
""" Resolved data folder location """
# load raw data files
names_abs, units_abs, formats_abs, widths_abs = zip(*DESC_ABS)
names_rtpd, units_rtpd, formats_rtpd, widths_rtpd = zip(*DESC_RTPD)
tables = {
    "mgn_abs": QTable(
        np.genfromtxt(
            datafolder / "mgn_abs.dat",
            dtype=formats_abs,
            delimiter=widths_abs,
            encoding="utf8",
            converters=STR_CONVERTER,
        ),
        names=names_abs,
        units=units_abs,
    ),
    "mgn_rtpd": QTable(
        np.genfromtxt(
            datafolder / "mgn_rtpd.dat",
            dtype=formats_rtpd,
            delimiter=widths_rtpd,
            encoding="utf8",
            converters=STR_CONVERTER,
        ),
        names=names_rtpd,
        units=units_rtpd,
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
