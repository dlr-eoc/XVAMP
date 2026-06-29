"""
Module that provides the reference profiles from :cite:t:`duan2010`,
including some profiles only present in the Matlab reference code.
"""

# standard imports
import numpy as np
from importlib.resources import files as res_files
import astropy.units as u
from astropy.units import cds

# package imports
from .. import data
from ..utils import read_unit_csv, Profile

# data location
BASEFOLDER = "duan_et_al_2010"
""" Base folder for data """
TABLES = [f"fig{n}" for n in ["6b", "7a", "7b", "7d", "8a", "8b", "9a", "9b"]] + ["4"]
""" Table numbers to load """

# get local installation folder paths
datafolder = res_files(data) / BASEFOLDER
""" Resolved data folder location """
# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}
""" Loaded tables """

# profiles defined manually

# ionosphere

ELECTRON_DENSITY_NODES = np.array(
    [
        [100, -2],
        [120, -2],
        [125, 0],
        [130, np.log10(1.3)],
        [140, np.log10(4)],
        [170, np.log10(0.7)],
        [250, np.log10(0.13)],
        [260, np.log10(0.02)],
        [375, np.log10(0.02)],
    ]
)
"""
Nodes that define the electron density profile as pairs of
altitude [km] and 1e11*log(electron density [1/m3])
"""
electron_density = Profile(
    ELECTRON_DENSITY_NODES[:, 0],
    ELECTRON_DENSITY_NODES[:, 1],
    index_unit=u.km,
    data_unit=u.m**-3,
    scale=1e11,
    log=True,
)
""" Electron density profile from Duan et al. (2010) """

# water vapor

H2O_FRACTION_NODES = np.array(
    [
        [4, 0],
        [5, 30],
        [60, 30],
        [60 + 1e-6, 10],
        [70 - 1e-6, 10],
        [70, 1.7],
        [75, 1.7],
        [85, 0.9],
        [88, 0.8],
        [92, 1.4],
        [98, 1.8],
        [99, 1.8],
        [110, 1.1],
        [112, 1.4],
        [113, 1.4],
        [116, 1.5],
        [117, 0],
    ]
)
"""
Nodes that define the H2O molar fraction profile as pairs of
altitude [km] and fraction [ppm]
"""
h2o_molar_fraction = Profile(
    H2O_FRACTION_NODES[:, 0],
    H2O_FRACTION_NODES[:, 1],
    index_unit=u.km,
    data_unit=cds.ppm,
)
""" Water vapor molar fraction profile from Duan et al. (2010) """

H2O_OLD_FRACTION_NODES = np.array([[5, 30], [60, 30]])
"""
Nodes that define the old H2O molar fraction profile as pairs of
altitude [km] and fraction [ppm]
"""
h2o_old_molar_fraction = Profile(
    H2O_OLD_FRACTION_NODES[:, 0],
    H2O_OLD_FRACTION_NODES[:, 1],
    index_unit=u.km,
    data_unit=cds.ppm,
)
""" Water vapor molar fraction profile from the reference code """

# sulfur dioxide

SO2_FRACTION_NODES = np.array(
    [
        [15, 25],
        [33, 60],
        [37, 90],
        [40, 110],
        [42, 150],
        [44, 220],
        [48, 150],
        [50, 90],
        [56, 40],
        [61, 40],
        [62, 0],
    ]
)
"""
Nodes that define the SO2 molar fraction profile as pairs of
altitude [km] and fraction [ppm]
"""
so2_molar_fraction = Profile(
    SO2_FRACTION_NODES[:, 0],
    SO2_FRACTION_NODES[:, 1],
    index_unit=u.km,
    data_unit=cds.ppm,
    lower=None,
)
""" Sulfur dioxide molar fraction profile from Duan et al. (2010) """

SO2_OLD_FRACTION_NODES = np.array(
    [
        [12, 22],
        [22, 38],
        [23, 38],
        [34, 130],
        [46, 130],
        [52, 110],
        [53, 110],
        [69, 0.6],
    ]
)
"""
Nodes that define the old SO2 molar fraction profile as pairs of
altitude [km] and fraction [ppm]
"""
so2_old_molar_fraction = Profile(
    SO2_OLD_FRACTION_NODES[:, 0],
    SO2_OLD_FRACTION_NODES[:, 1],
    index_unit=u.km,
    data_unit=cds.ppm,
)
""" Sulfur dioxide molar fraction profile from the reference code """

# gaseous sulfuric acid

H2SO4_FRACTION_NODES = np.array(
    [
        [29, 0],
        [30, 0],
        [40, 1],
        [44, 2.4],
        [46, 9],
        [50, 1.7],
        [52, 1.7],
        [56, 0],
        [57, 0],
    ]
)
"""
Nodes that define the H2SO4 molar fraction profile as pairs of
altitude [km] and fraction [ppm]
"""
h2so4_molar_fraction = Profile(
    H2SO4_FRACTION_NODES[:, 0],
    H2SO4_FRACTION_NODES[:, 1],
    index_unit=u.km,
    data_unit=cds.ppm,
)
""" Sulfuric acid molar fraction profile from Duan et al. (2010) """
H2SO4_3212_FRACTION_NODES = np.array(
    [
        [30, 0],
        [35 - 1e-6, 0],
        [35, 0.5 + 1.5],
        [39, 3.5 + 1.5],
        [42, 2.6 + 1.5],
        [45, 4.5 + 1.5],
        [48, 4.6 + 1.5],
        [50, 1 + 1.5],
        [51, 0.5 + 1.5],
        [55, 0.5 + 1.5],
        [55 + 1e-6, 0],
        [56, 0],
    ]
)
"""
Nodes that define the H2SO4 molar fraction profile from a modified Magellan
orbit 3212 as pairs of altitude [km] and fraction [ppm]
"""
h2so4_3212_molar_fraction = Profile(
    H2SO4_3212_FRACTION_NODES[:, 0],
    H2SO4_3212_FRACTION_NODES[:, 1],
    index_unit=u.km,
    data_unit=cds.ppm,
)
"""
Sulfuric acid molar fraction profile from the reference code
from Magellan orbit 3212
"""

# carbon monoxide
CO_FRACTION_NODES = np.array(
    [
        [42, np.log10(28)],
        [52, np.log10(42)],
        [64, np.log10(52)],
        [68, np.log10(20)],
        [70, 1],
        [71, 1],
        [73, np.log10(30)],
        [77, np.log10(30)],
        [87, 1],
        [90, 1],
        [126, 4],
        [127, -100],
    ]
)
"""
Nodes that define the CO molar fraction profile as pairs of
altitude [km] and fraction [ppm]
"""
co_molar_fraction = Profile(
    CO_FRACTION_NODES[:, 0],
    CO_FRACTION_NODES[:, 1],
    index_unit=u.km,
    data_unit=cds.ppm,
    log=True,
    lower=None,
)
""" Carbon monoxide molar fraction profile from Duan et al. (2010) """
CO_OLD_FRACTION_NODES = np.array([[30, 23], [40, 29]])
"""
Nodes that define the old CO molar fraction profile as pairs of
altitude [km] and fraction [ppm]
"""
co_old_molar_fraction = Profile(
    CO_OLD_FRACTION_NODES[:, 0],
    CO_OLD_FRACTION_NODES[:, 1],
    index_unit=u.km,
    data_unit=cds.ppm,
)
""" Carbon monoxide molar fraction profile from the reference code """

# carbonyl sulfide

OCS_FRACTION_NODES = np.array(
    [[29, 0], [30, 14], [33, 14], [34, 0], [37, 0], [38, 0.35], [40, 0.35], [41, 0]]
)
"""
Nodes that define the OCS molar fraction profile as pairs of
altitude [km] and fraction [ppm]
"""
ocs_molar_fraction = Profile(
    OCS_FRACTION_NODES[:, 0],
    OCS_FRACTION_NODES[:, 1],
    index_unit=u.km,
    data_unit=cds.ppm,
)
""" Carbonyl sulfide molar fraction profile from Duan et al. (2010) """
OCS_OLD_FRACTION_NODES = np.array([[30, 20], [38, 0.35]])
"""
Nodes that define the old OCS molar fraction profile as pairs of
altitude [km] and fraction [ppm]
"""
ocs_old_molar_fraction = Profile(
    OCS_OLD_FRACTION_NODES[:, 0],
    OCS_OLD_FRACTION_NODES[:, 1],
    index_unit=u.km,
    data_unit=cds.ppm,
)
""" Carbonyl sulfide molar fraction profile from the reference code """

# profile loaded from Table 4
h2so4_density_conc = Profile(
    tables["4"]["Weight Percentage"].to("%"),
    tables["4"]["Density"],
    lower=np.nan,
    upper=np.nan,
)
"""
Relationship profile between H2SO4 weight percentage of diluted sulfuric
acid and the resulting mass density
"""
