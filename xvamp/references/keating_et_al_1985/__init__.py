"""
Module that provides the background, reference atmospheric properties from
:cite:t:`keating1985`.
"""

# standard imports
from typing import Tuple
import numpy as np
from importlib.resources import files as res_files
from astropy.units import Quantity
from astropy.table import hstack, vstack

# package imports
from ...utils.io import read_unit_csv
from ...profile import MultiProfile

# data content
SZA_TABLES = ["4-4", "4-9", "4-10", "4-11", "4-12", "4-13", "4-5"]
""" Table numbers that build the SZA-defined datacube between 150-250km """
TABLES = ["4-6", "4-7", "4-15", "4-16"] + SZA_TABLES
""" Table numbers to load """
SZA = Quantity([16, 34, 61, 90, 119, 146, 164], "°")
""" Solar zenith angles for tables 4-[4, 9-13, 5] [°] """

# get local installation folder paths
datafolder = res_files()
""" Resolved current folder location """
# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}
""" Loaded tables """


# function to combine the different tables to get joint profiles
# for day and night
def _combine_tables_day_night() -> (
    Tuple[MultiProfile, MultiProfile, MultiProfile, MultiProfile]
):
    # stack
    day = vstack(
        [
            hstack([tables["4-4"], tables["4-6"]], join_type="exact"),
            tables["4-16"][1:],
        ]
    )
    night = vstack(
        [
            hstack([tables["4-5"], tables["4-7"]], join_type="exact"),
            tables["4-15"][1:],
        ]
    )
    # sort
    day.sort("ALT")
    night.sort("ALT")
    # convert temperature, pressure, and density to MultiProfiles
    names_tpd = ["T", "P", "RHO"]
    mp_day_tpd = MultiProfile(index=day["ALT"], data=day[names_tpd])
    mp_night_tpd = MultiProfile(index=night["ALT"], data=night[names_tpd])
    # convert the atmospheric species to MultiProfiles
    names_species = ["CO2", "O", "CO", "HE", "N", "N2", "H", "NTOT"]
    mp_day_species = MultiProfile(index=day["ALT"], data=day[names_species])
    mp_night_species = MultiProfile(index=night["ALT"], data=night[names_species])
    # done
    return mp_day_tpd, mp_night_tpd, mp_day_species, mp_night_species


# run MultiProfile extractor
_combined_multiprofiles = _combine_tables_day_night()
tpd_day = _combined_multiprofiles[0]
""" Temperature, pressure, and mass density profiles at day """
tpd_night = _combined_multiprofiles[1]
""" Temperature, pressure, and mass density profiles at night """
species_day = _combined_multiprofiles[2]
""" Species number density profiles (incl. total number density) at day """
species_night = _combined_multiprofiles[3]
""" Species number density profiles (incl. total number density) at night """

# build the datacube for 150-250km for the seven samplings of
# solar zenith angle (which stands in for time)
dcube_150km_250km = np.stack(
    [tables[t].as_array().view((float, 11)) for t in SZA_TABLES],
    axis=2,
)
"""
Physical and species quantities as a function of altitude and
solar zenith angle between 150-250km

:meta hide-value:
"""
units_150km_250km = [
    tables["4-4"].columns[i].unit for i in range(dcube_150km_250km.shape[1])
]
""" Units of :attr:`~dcube_150km_250km` """
names_150km_250km = tables["4-4"].colnames
""" Column (axis 1) names of of :attr:`~dcube_150km_250km` """

# build the (smaller) datacube for 100-150km, which only contains
# noon and midnight
dcube_100km_150km = np.stack(
    [tables[t].as_array().view((float, 16)) for t in ["4-16", "4-15"]],
    axis=2,
)
"""
Physical and species quantities as a function of altitude and
midnight/noon between 100-150km

:meta hide-value:
"""
units_100km_150km = [
    tables["4-15"].columns[i].unit for i in range(dcube_100km_150km.shape[1])
]
""" Units of :attr:`~dcube_100km_150km` """
names_100km_150km = tables["4-15"].colnames
""" Column (axis 1) names of of :attr:`~dcube_100km_150km` """
