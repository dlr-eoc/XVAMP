"""
Module that provides the reference profiles from :cite:t:`jenkins2002`.
"""

# standard imports
import numpy as np
from importlib.resources import files as res_files

# package imports
from .. import data
from ..utils import read_unit_csv

# data location
BASEFOLDER = "jenkins_et_al_2002"
""" Base folder for data """
TABLES = ["fig6raw"]
""" Table numbers to load """

# get local installation folder paths
datafolder = res_files(data) / BASEFOLDER
# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}

# extract different H2SO4 profiles as a function of assumed SO2 abundance
for cn in tables["fig6raw"].colnames[::2]:
    # get assumed SO2 abundance
    assum_so2 = cn.split(":")[0]
    # quick access to subtable
    src_columns = [f"{assum_so2}: altitude", cn]
    t = tables["fig6raw"][src_columns]
    # get mask of valid rows
    try:
        mask = ~t[cn].mask
    except AttributeError:  # not a masked quantity, so all rows valid
        mask = np.s_[:]
    # rename columns
    t.rename_columns(src_columns, ["altitude", "mixing ratio of H2SO4"])
    # save valid rows as new table
    tables[assum_so2] = t[mask]
