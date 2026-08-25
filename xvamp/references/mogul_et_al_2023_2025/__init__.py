"""
Module that provides the reference profiles from :cite:t:`mogul2023`
and :cite:t:`mogul2025`.
"""

# standard imports
from importlib.resources import files as res_files

# package imports
from ...utils.io import read_unit_csv
from ...profile import Profile
from ..vonzahn_moroz_1985 import CO2_MR, CO2_MR_ERR, N2_MR, N2_MR_ERR

# data content
TABLES = ["LNMS.MixingRatios.N2.CO2"]
""" Table names to load """

# get local installation folder paths
_datafolder = res_files()
""" Resolved current folder location """
# load raw data files
tables = {t: read_unit_csv(_datafolder / f"{t}.csv") for t in TABLES}
"""
Loaded tables

:meta hide-value:
"""

# define profiles
co2_molar_fraction = Profile(
    index=tables["LNMS.MixingRatios.N2.CO2"]["altitude"],
    data=tables["LNMS.MixingRatios.N2.CO2"]["CO2 Molar Fraction"].to("ppm"),
    lower=None,
    upper=CO2_MR.to_value("ppm"),
)
""" CO2 molar fraction [ppm] """
co2_molar_fraction_error = Profile(
    index=tables["LNMS.MixingRatios.N2.CO2"]["altitude"],
    data=tables["LNMS.MixingRatios.N2.CO2"]["CO2 Molar Fraction Error"].to("ppm"),
    lower=None,
    upper=CO2_MR_ERR.to_value("ppm"),
)
""" CO2 molar fraction error [ppm] """
n2_molar_fraction = Profile(
    index=tables["LNMS.MixingRatios.N2.CO2"]["altitude"],
    data=tables["LNMS.MixingRatios.N2.CO2"]["N2 Molar Fraction"].to("ppm"),
    lower=None,
    upper=N2_MR.to_value("ppm"),
)
""" N2 molar fraction [ppm] """
n2_molar_fraction_error = Profile(
    index=tables["LNMS.MixingRatios.N2.CO2"]["altitude"],
    data=tables["LNMS.MixingRatios.N2.CO2"]["N2 Molar Fraction Error"].to("ppm"),
    lower=None,
    upper=N2_MR_ERR.to_value("ppm"),
)
""" N2 molar fraction error [ppm] """
