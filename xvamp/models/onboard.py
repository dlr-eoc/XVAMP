"""
Module with the onboard atmospheric model.
"""

# standard imports
import numpy as np

# package imports
from ..constants import VENUS_RADIUS
from ..utils import float_or_array


class OnboardPolynomial:
    """
    Onboard method to compute the atmoshperically-derived
    range error and attenuation.
    """

    # static parameters
    platfhref = 220e3
    """
    Altitude reference value [m] to be used for the
    range radiometric correction
    """
    platfrref = VENUS_RADIUS.to("m").value + platfhref
    """
    Radius reference value [m] to be used for the
    range radiometric correction
    """
    # ppAtmAtt = np.polynomial.Polynomial(
    #     np.array(
    #         [
    #             3.294820214284067e-13,
    #             2.102876898752960e-08,
    #             -6.194056295420677e-04,
    #             3.867025738546215,
    #         ]
    #     )[::-1]
    # )
    # """
    # Polynomial for fitting the atmospheric attenuation in dB relative
    # to a refence value
    # """
    ppRngGeoAppInpt = np.polynomial.Polynomial(
        np.array(
            [
                9.386174831179326e-14,
                5.375234162035260e-09,
                -4.948962971501351e-04,
                9.800611433117087,
            ]
        )[::-1]
    )
    """
    Polynomial for fitting the intercept of geometric range vs. apparent range
    as a function of terrain height
    """
    ppRngGeoAppSlope = np.polynomial.Polynomial(
        np.array(
            [
                1.450737895980288e-17,
                -1.602031859787405e-12,
                7.138626911289445e-08,
                0.998767935981350,
            ]
        )[::-1]
    )
    """
    Polynomial for fitting the slope of geometric range vs. apparent range
    as a function of terrain height
    """

    def get_geometric_range(
        self, h_t: float_or_array, r_o: float_or_array, rho_tilde: float_or_array
    ) -> float_or_array:
        """
        Compute the geomtric range from the apparent range.

        Parameters
        ----------
        h_t
            Terrain height [m]
        r_o
            Observer radius [m]
        rho_tilde
            Apparent range [m]

        Returns
        -------
            Geometric range [m]
        """
        # approximate cosine of look angle
        cos_theta_app = (
            r_o**2 + rho_tilde**2 - (VENUS_RADIUS.to("m").value + h_t) ** 2
        ) / (2 * rho_tilde * r_o)
        # range to reference altitude
        rho_ref = r_o * cos_theta_app - np.sqrt(
            (r_o * cos_theta_app) ** 2 + self.platfrref**2 - r_o**2
        )
        # slope and intercept of geometric range vs. apparent range
        a_rho = self.ppRngGeoAppSlope(h_t)
        b_rho = self.ppRngGeoAppInpt(h_t)
        # geometric range
        rho = a_rho * (rho_tilde - rho_ref) + b_rho + rho_ref
        # done
        return rho

    def get_apparent_range(
        self,
        h_t: float_or_array,
        r_o: float_or_array,
        rho: float_or_array,
        iter: int = 2,
    ) -> float_or_array:
        """
        Compute the geomtric range from the apparent range.

        Parameters
        ----------
        h_t
            Terrain height [m]
        r_o
            Observer radius [m]
        rho
            Geometric range [m]
        iter
            Number of iterations [-]

        Returns
        -------
            Apparent range [m]
        """
        # initial guess
        rho_tilde = rho
        # slope and intercept of geometric range vs. apparent range
        a_rho = self.ppRngGeoAppSlope(h_t)
        b_rho = self.ppRngGeoAppInpt(h_t)
        # fixed number of iterations
        for _ in range(iter):
            # approximate cosine of look angle
            cos_theta_app = (
                r_o**2 + rho_tilde**2 - (VENUS_RADIUS.to("m").value + h_t) ** 2
            ) / (2 * rho_tilde * r_o)
            # range to reference altitude
            rho_ref = r_o * cos_theta_app - np.sqrt(
                (r_o * cos_theta_app) ** 2 + self.platfrref**2 - r_o**2
            )
            # update apparent range
            rho_tilde = ((rho - rho_ref) - b_rho) / a_rho + rho_ref
        # done
        return rho_tilde
