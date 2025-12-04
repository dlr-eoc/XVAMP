"""
Package initialization
"""

# standard imports
import numpy as np
import astropy.units as u
from astropy.units import cds

# define the "N-unit" for the refractivity, given as
# N = (n-1) * 10^6
N_UNIT = u.def_unit("Nunit")
N_UNIT_EQ = [
    (
        N_UNIT,
        u.dimensionless_unscaled,
        lambda x: (x / 1e6) + 1,
        lambda x: (x - 1) * 1e6,
    ),
    (
        N_UNIT * u.Unit("cm3/mol"),
        u.Unit("cm3/mol"),
        lambda x: (x / 1e6) + 1,
        lambda x: (x - 1) * 1e6,
    ),
]
# add the absorption coefficient equivalence between dB/km and 1/cm
DBKM_CM_FACTOR = np.log(10) / 1e6
DB_CM_EQ = [
    (u.dB / u.km, u.cm**-1, lambda x: x * DBKM_CM_FACTOR, lambda x: x / DBKM_CM_FACTOR)
]
# define the Neper unit
NP_UNIT = u.def_unit("Np")
# add the absorption coefficient equivalences between Np and dB and 1/cm
DB_NP_FACTOR = np.log10(np.e) / 20
CM_DB_FACTOR = DB_NP_FACTOR / DBKM_CM_FACTOR
DBKM_CM_NPKM_EQ = [
    (
        u.dB / u.km,
        NP_UNIT / u.km,
        lambda x: x * DB_NP_FACTOR,
        lambda x: x / DB_NP_FACTOR,
    ),
    (
        u.cm**-1,
        NP_UNIT / u.km,
        lambda x: x * CM_DB_FACTOR,
        lambda x: x / CM_DB_FACTOR,
    ),
]
# add units and equivalences to global scope
# also add "ppm" and "atm" as units necessary for files or equations
try:
    u.add_enabled_units([N_UNIT, NP_UNIT, cds.ppm, cds.atm])
except ValueError as e:  # already enabled
    if "Object with name 'Nunit' already exists in namespace" not in str(e):
        raise
else:
    u.add_enabled_equivalencies(N_UNIT_EQ + DB_CM_EQ + DBKM_CM_NPKM_EQ)
