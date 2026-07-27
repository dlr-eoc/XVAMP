"""
Module that provides the reference profiles from :cite:t:`marcq2006`.
"""

# standard imports
from importlib.resources import files as res_files

# package imports
from ...utils.io import read_unit_csv
from ...profile import Profile

# data content
TABLES = ["fig8"]
""" Table numbers to load """

# get local installation folder paths
datafolder = res_files()
""" Resolved current folder location """
# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}
""" Loaded tables """

# define Profile for OCS
ocs_mr = Profile(
    index=tables["fig8"]["altitude"],
    data=tables["fig8"]["mixing ratio of OCS"].to("ppm"),
)
""" OCS mixing ratio """
