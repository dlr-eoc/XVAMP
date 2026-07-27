"""
Module that provides the reference model from :cite:t:`zasova2006`.
"""

# standard imports
from importlib.resources import files as res_files

# package imports
from ...utils.io import read_unit_csv

# data content
TABLES = ["2", "3", "4", "5", "6"]
""" Table numbers to load """

# get local installation folder paths
datafolder = res_files()
""" Resolved current folder location """
# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}
""" Loaded tables """
