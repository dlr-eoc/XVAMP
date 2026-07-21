"""
Module that provides the reference profiles from :cite:t:`jenkins2002`.
"""

# standard imports
import numpy as np
from importlib.resources import files as res_files

# package imports
from .. import data
from ..utils.io import read_unit_csv
from ..profile import Profile

# data location
BASEFOLDER = "jenkins_et_al_2002"
""" Base folder for data """
TABLES = ["fig6raw"]
""" Table numbers to load """

# get local installation folder paths
datafolder = res_files(data) / BASEFOLDER
""" Resolved data folder location """
# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}
""" Loaded tables """


# define function to extract a profile tied to an assumed SO2 abundance
def _get_h2so4_assumed_so2(assumed_so2: str) -> Profile:
    # get assumed SO2 abundance
    col_alt = f"{assumed_so2} ppm SO2: altitude"
    col_mr = f"{assumed_so2} ppm SO2: mixing ratio of H2SO4"
    # get mask of valid rows
    try:
        mask = ~tables["fig6raw"][col_mr].mask
    except AttributeError:  # not a masked quantity, so all rows valid
        mask = np.s_[:]
    # create profile and return
    return Profile(tables["fig6raw"][col_alt][mask], tables["fig6raw"][col_mr][mask])


# create profiles
h2so4_molar_fraction_0ppm_so2 = _get_h2so4_assumed_so2("0")
""" H2SO4 molar fraction profile from :cite:t:`jenkins2002` assuming 0 ppm SO2 """
h2so4_molar_fraction_50ppm_so2 = _get_h2so4_assumed_so2("50")
""" H2SO4 molar fraction profile from :cite:t:`jenkins2002` assuming 50 ppm SO2 """
h2so4_molar_fraction_100ppm_so2 = _get_h2so4_assumed_so2("100")
""" H2SO4 molar fraction profile from :cite:t:`jenkins2002` assuming 100 ppm SO2 """
h2so4_molar_fraction_150ppm_so2 = _get_h2so4_assumed_so2("150")
""" H2SO4 molar fraction profile from :cite:t:`jenkins2002` assuming 150 ppm SO2 """
h2so4_molar_fraction_200ppm_so2 = _get_h2so4_assumed_so2("200")
""" H2SO4 molar fraction profile from :cite:t:`jenkins2002` assuming 200 ppm SO2 """
