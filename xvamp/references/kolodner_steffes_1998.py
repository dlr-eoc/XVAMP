"""
Module that provides the reference profiles and some H2SO4 experiment formulas
from :cite:t:`kolodner1998`.
"""

# standard imports
import astropy.units as u
from importlib.resources import files as res_files
from typing import Tuple
from astropy.units import Quantity

# package imports
from .. import data
from ..constants import ESU_CM, SPEC_MOL_M
from ..utils.io import read_unit_csv
from ..profile import Profile

# data location
BASEFOLDER = "kolodner_steffes_1998"
""" Base folder for data """
TABLES = ["fig789"]
""" Table numbers to load """

# get local installation folder paths
datafolder = res_files(data) / BASEFOLDER
""" Resolved data folder location """
# load raw data files
tables = {t: read_unit_csv(datafolder / f"table{t}.csv") for t in TABLES}
""" Loaded tables """

# parameters of the experiment
MU_H2SO4 = Quantity(2.72e-18, ESU_CM)
""" Molecular dipole moment for gaseous sulfuric acid """
T_H2SO4 = Quantity(553, "K")
""" Temperature of the experiment of :cite:t:`kolodner1998` """
# others only defined in _get_eps_prime_r_and_molar_density


# define function to compute the real part of the relative permittivity and the
# molar density of H2SO4 in the experiment as described in Section 3.2.
def _get_eps_prime_r_and_molar_density() -> (
    Tuple[Quantity["dimensionless"], Quantity["molar concentration"]]
):
    # refractivity of the gaseous sulfuric acid
    n_h2so4 = Quantity((340.64 + 245.36) / 2, "Nunit")
    # mass density of the sulfuric acid solution before it evaporates
    d_h2so4_l = Quantity(1.8305, "g/ml")
    # dissociation constant of vaporized H2SO4
    diss_h2so4 = 0.461
    # volume of the H2SO4 solution which vaporizes
    v_h2so4 = Quantity((4.12 + 3.18) / 2, "cm3")
    # volume of the pressure vessel
    v_vessel = Quantity(31, "l")
    # number of moles of pure H2SO4 liquid which vaporizes
    nvap = (v_h2so4 * d_h2so4_l / SPEC_MOL_M["H2SO4"]).decompose()
    # number of moles of H2SO4 vapor
    nmol = nvap * (1 - diss_h2so4)
    # real part of the relative permittivity
    eps_prime_r = (n_h2so4.to(u.dimensionless_unscaled)) ** 2
    # molar density of gaseous sulfuric acid
    rho = (nmol / v_vessel).to("mol/cm3")
    # done
    return eps_prime_r, rho


# evaluate the experiment
_eps_mdens = _get_eps_prime_r_and_molar_density()
eps_prime_r_h2so4 = _eps_mdens[0]
""" Real part of the relative permittivity of H2SO4 in the experiment """
rho_h2so4 = _eps_mdens[1]
""" Molar density of H2SO4 in the experiment """

# define quick access to each of the three X-band orbital data as Profiles,
# clipped to non-negative values
h2so4_mr_3212 = Profile(
    index=tables["fig789"]["altitude"],
    data=tables["fig789"]["3212"].value.clip(0),
    data_unit=tables["fig789"]["3212"].unit,
)
""" H2SO4 mixing ratio from Magellan orbit 3212 """
h2so4_mr_3213 = Profile(
    index=tables["fig789"]["altitude"],
    data=tables["fig789"]["3213"].value.clip(0),
    data_unit=tables["fig789"]["3213"].unit,
)
""" H2SO4 mixing ratio from Magellan orbit 3213 """
h2so4_mr_3214 = Profile(
    index=tables["fig789"]["altitude"],
    data=tables["fig789"]["3214"].value.clip(0),
    data_unit=tables["fig789"]["3214"].unit,
)
""" H2SO4 mixing ratio from Magellan orbit 3214 """

# average the three different orbits to have a single H2SO4 abundance
# profile from X-band
h2so4_mr_mean = Profile(
    index=tables["fig789"]["altitude"],
    data=tables["fig789"]
    .to_pandas(index="altitude")
    .mean(axis=1)
    .clip(lower=0)
    .to_numpy(),
    data_unit=tables["fig789"]["3212"].unit,
)
""" Mean H2SO4 mixing ratio from Magellan orbits 3212, 3213 and 3214 """
