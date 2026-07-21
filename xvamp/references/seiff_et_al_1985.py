"""
Module that provides the reference model from :cite:t:`seiff1985`.
"""

# standard imports
import numpy as np
from importlib.resources import files as res_files
from astropy.units import Quantity

# package imports
from .. import data
from ..utils.io import read_unit_csv

# data location
BASEFOLDER = "seiff_et_al_1985"
""" Base folder for data """
LAT_TABLES = ["1-2a", "1-2b", "1-2c", "1-2d", "1-2e"]
""" Table numbers of the 33-100km range for different latitudes"""
TABLES = ["1-1", "1-3"] + LAT_TABLES
""" Table numbers to load """

# data format information
LAT = Quantity([30, 45, 60, 75, 85], "°")
""" Latitudes of tables 1-2[a-e]"""

# get local installation folder paths
datafolder = res_files(data) / BASEFOLDER
""" Resolved data folder location """
# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}
""" Loaded tables """

# build the datacube for 33-100km for the five samplings of latitude
dcube_33km_100km = np.stack(
    [
        tables[t].as_array().view((float, 8 if t == "1-2a" else 4))[:, :4]
        for t in LAT_TABLES
    ],
    axis=2,
)
"""
Physical quantities as a function of altitude and
latitude between 33-100km
"""
units_33km_100km = [
    tables["1-2b"].columns[i].unit for i in range(dcube_33km_100km.shape[1])
]
""" Units of :attr:`~dcube_33km_100km` """
names_33km_100km = tables["1-2b"].colnames
""" Column (axis 1) names of of :attr:`~dcube_33km_100km` """
