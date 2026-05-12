"""
Module that provides the background, reference atmospheric properties.
"""

# standard imports
from importlib.resources import files as res_files

# package imports
from .. import data
from ..utils import read_unit_fwf

# data location
BASEFOLDER = "jpl_spectral_lines"
""" Base folder for data """
TABLES = {"OCS": "c060001.cat", "SO2": "c064002.cat"}
""" Tables to load """

# file format info
COLUMNS = [
    "FREQ",
    "ERR",
    "LGINT",
    "DR",
    "ELO",
    "GUP",
    "TAG",
    "QNFMT",
    "QN'",
    'QN"',
]
""" Column names in the catalog """
FORMATS = ["f8", "f8", "f8", "i", "f8", "i", "i", "i", "U12", "U12"]
""" Column formats """
WIDTHS = [13, 8, 8, 2, 10, 3, 7, 4, 12, 12]
""" Fixed widths of the catalog columns """
UNITS = ["MHz", "MHz", "dex(nm2 MHz)", "", "cm-1", "", "", "", "", ""]
""" Column units """


# get local installation folder paths
datafolder = res_files(data) / BASEFOLDER
# load raw data files
tables = {
    species: read_unit_fwf(
        datafolder / filename,
        names=COLUMNS,
        formats=FORMATS,
        widths=WIDTHS,
        units=UNITS,
    )
    for species, filename in TABLES.items()
}
