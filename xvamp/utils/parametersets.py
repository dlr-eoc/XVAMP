"""
Utility module for the atmospheric model.
"""

# standard imports
import numpy as np
from dataclasses import dataclass
from astropy.units import Quantity

# package imports
from . import float_or_array
from ..constants import AVOGADRO, BOLTZMANN, ESU_CM


@dataclass
class HarveyLemmon2005Parameters:
    """
    Parameters for mixture components from :cite:t:`harvey2005`,
    as represented in :cite:t:`duan2010`, Table 1 for eq. (8).
    Parameters are NOT converted to astropy :class:`~astropy.units.Quantity`
    because of the unknown exponent.
    """

    a0: float = 0
    """ [cm^3/mol] """
    a1: float = 0
    """ [cm^3/mol] """
    b0: float = 0
    """ [cm^6/mol^2] """
    b1: float = 0
    """ [cm^6/mol^2] """
    c0: float = 0
    """ [cm^(3(D+1))/mol^-(D+1)] """
    c1: float = 0
    """ [cm^(3(D+1))/mol^-(D+1)] """
    D: float = 0
    """ [-] """
    T0: float = 273.16
    """ Temperature [K] """
    A_mu: float = 0
    """ Dipolar term in the virial expansion [cm^3 K/mol] """

    @staticmethod
    def get_A_mu(mu: Quantity) -> float_or_array:
        """
        Compute the dipolar term in the dielectric virial expansion,
        assuming CGS units in the input, but SI in the output.

        Parameters
        ----------
        mu
            Permanent dipole moment [esu cm]

        Returns
        -------
            Dipolar term in the virial expansion [cm^3 K/mol]
        """
        return ((4 * np.pi * AVOGADRO * mu**2) / (9 * BOLTZMANN)).to("cm3 K/mol").value


@dataclass
class Pitzer1983Parameters:
    """
    Parameters for the :cite:t:`pitzer1983` model to calculate the polarization
    per molar volume as given by :cite:t:`duan2010` on p. 5, eq. (14).
    """

    mu: Quantity[ESU_CM]
    """ Molecular dipole moment [esu cm = 1e18 D] """
    alpha_T: Quantity["cm3"]
    """ Molecular polarizability [cm^3] """


@dataclass
class LineShapeParameters:
    """
    Line shape parameters compatible with the Ben-Reuven line shape function, following
    the notation from :cite:t:`duan2010`, eqs. (27-32) on p. 10f.
    Can be used for Lorentzian line shapes if only specifying
    :attr:`~LineShapeParameters.gamma_min_min`.
    """

    T_0: Quantity["K"]
    """ Reference temperature of broadening coefficients [K] """
    gamma_min_min: Quantity["MHz/torr"]
    """ Self-broadened linewidth parameter [MHz/torr] """
    gamma_min_maj: Quantity["MHz/torr"] = Quantity(0, "MHz/torr")
    """ Foreign-broadened linewidth parameter [MHz/torr] """
    zeta_min_min: Quantity["MHz/torr"] = Quantity(0, "MHz/torr")
    """ Self-coupling linewidth parameter [MHz/torr] """
    zeta_min_maj: Quantity["MHz/torr"] = Quantity(0, "MHz/torr")
    """ Foreign-coupling parameter [MHz/torr] """
    delta_min: Quantity["MHz/torr"] = Quantity(0, "MHz/torr")
    """ Frequency shift parameter [MHz/torr] """
    m: float = 0.0
    """ Temperature dependence of the coupling [-] """
    n: float = 0.0
    """ Temperature dependence of the linewidth [-] """
