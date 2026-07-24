"""
Module containing classes which enable a convenient and well-documented
way to store and compare parameter sets.
"""

# standard imports
from __future__ import annotations
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

    def __eq__(self, other: HarveyLemmon2005Parameters | Pitzer1983Parameters):
        """
        Check whether two parameter sets are the same.

        Parameters
        ----------
        other
            The other parameter set to check equality with.
        """
        if isinstance(other, HarveyLemmon2005Parameters):
            return np.allclose(
                [
                    self.a0,
                    self.a1,
                    self.b0,
                    self.b1,
                    self.c0,
                    self.c1,
                    self.D,
                    self.T0,
                    self.A_mu,
                ],
                [
                    other.a0,
                    other.a1,
                    other.b0,
                    other.b1,
                    other.c0,
                    other.c1,
                    other.D,
                    other.T0,
                    other.A_mu,
                ],
                rtol=1e-14,
                atol=1e-14,
            )
        elif isinstance(other, Pitzer1983Parameters):
            if not (self.a1 == self.b0 == self.b1 == self.c0 == self.c1 == 0.0):
                return False
            other_A_eps = (
                ((4 * np.pi * AVOGADRO * other.alpha_T) / 3).to("cm3/mol").value
            )
            if not np.allclose(self.a0, other_A_eps, rtol=1e-14, atol=1e-14):
                return False
            other_A_mu = self.get_A_mu(other.mu)
            if not np.allclose(self.A_mu, other_A_mu, rtol=1e-14, atol=1e-14):
                return False
            return True
        else:
            raise NotImplementedError


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

    def __eq__(self, other: HarveyLemmon2005Parameters | Pitzer1983Parameters):
        """
        Check whether two parameter sets are the same.

        Parameters
        ----------
        other
            The other parameter set to check equality with.
        """
        if isinstance(other, HarveyLemmon2005Parameters):
            if not (other.a1 == other.b0 == other.b1 == other.c0 == other.c1 == 0.0):
                return False
            self_A_eps = ((4 * np.pi * AVOGADRO * self.alpha_T) / 3).to("cm3/mol").value
            if not np.allclose(self_A_eps, other.a0, rtol=1e-14, atol=1e-14):
                return False
            self_A_mu = self.get_A_mu(other.mu)
            if not np.allclose(self_A_mu, other.A_mu, rtol=1e-14, atol=1e-14):
                return False
            return True
        elif isinstance(other, Pitzer1983Parameters):
            return np.allclose(
                [self.mu.to_value() * 1e18, self.alpha_T.to_value() * 1e24],
                [
                    other.mu.to_value(self.mu.unit) * 1e18,
                    other.alpha_T.to_value(self.alpha_T.unit) * 1e24,
                ],
                rtol=1e-14,
                atol=1e-14,
            )
        else:
            raise NotImplementedError


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
