"""
Module that provides the background, reference atmospheric properties.
"""

# standard imports
import numpy as np
import astropy.units as u
from importlib.resources import files as res_files
from astropy.table import QTable, join

# package imports
from .. import data
from ..utils import read_unit_csv

BASEFOLDER = "james_et_al_1997"
""" Base folder for data """

TABLES = ["fig4bdroplets", "fig4bnuclei", "fig7"]
""" Table numbers to load """

# get local installation folder paths
datafolder = res_files(data) / BASEFOLDER

# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}

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
# move index back to column for export
clouds.reset_index(names="altitude", inplace=True)
# save as QTable with ppm units
tables["clouds"] = QTable.from_pandas(
    clouds[["altitude", mmr_name]],
    units={"altitude": u.Unit("km"), mmr_name: u.Unit("1e-6")},
)
