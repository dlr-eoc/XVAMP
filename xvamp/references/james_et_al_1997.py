"""
Module that provides the reference profiles from :cite:t:`james1997`.
"""

# standard imports
import numpy as np
import astropy.units as u
from astropy.units import cds
from importlib.resources import files as res_files
from astropy.table import join

# package imports
from .. import data
from ..utils.io import read_unit_csv
from ..profile import Profile

# data location
BASEFOLDER = "james_et_al_1997"
""" Base folder for data """
TABLES = ["fig4bdroplets", "fig4bnuclei", "fig7"]
""" Table numbers to load """

# get local installation folder paths
datafolder = res_files(data) / BASEFOLDER
""" Resolved data folder location """
# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}
""" Loaded tables """


# function to derive mass mixing ratio
def _get_cloud_mass_mixing_ratio() -> Profile:
    # combine the two tables for Fig. 4 to get the mass mixing ratio
    # of the liquid part of the clouds
    clouds = join(
        tables["fig4bnuclei"],
        tables["fig4bdroplets"],
        keys="Altitude",
        join_type="outer",
    ).to_pandas(index="Altitude")
    # move to log space
    clouds = np.log10(clouds)
    # interpolate values in between bounds
    clouds = clouds.interpolate("index", limit_area="inside")
    # back to linear space
    clouds = 10**clouds
    # extrapolate nuclei: to zero above
    clouds.iloc[:, 0] = clouds.iloc[:, 0].fillna(0)
    # calculate liquid ratio
    mmr_name = "mass mixing ratio clouds"
    clouds[mmr_name] = np.fmax(clouds.iloc[:, 1] - clouds.iloc[:, 0], 0)
    # get non-zero start for mixing ratio for proper bounds
    i_start = np.argmax(np.diff(clouds[mmr_name]) > 0) + 1
    # subset data
    mmr_nonzero = clouds[mmr_name].to_numpy().squeeze()[i_start:]
    alt_nonzero = clouds.index[i_start:].to_numpy()
    # return as Profile
    return Profile(
        alt_nonzero,
        np.log10(mmr_nonzero),
        index_unit=u.km,
        data_unit=cds.ppm,
        log=True,
    )


# get cloud mixing ratio Profile
mass_mixing_ratio_clouds = _get_cloud_mass_mixing_ratio()
""" Mass mixing ratio of the clouds """
# concentration can be taken directly from the figure
cloud_concentration = Profile(
    tables["fig7"]["Altitude"].to("km"),
    tables["fig7"]["Weight Percent"].to("%"),
    lower=np.nan,
    upper=np.nan,
)
""" Cloud H2SO4 weight percent profile """
