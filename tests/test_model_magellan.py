"""
Test whether the model yields absorption and refraction profiles with its
"Verification" settings that approximately match the Magellan data for orbit 3212.
"""

# standard imports
import unittest
import numpy as np
from astropy.units import Quantity, dimensionless_unscaled

# package imports
from xvamp.models.duan_et_al_2010 import Duan2010Verification
from xvamp.references import magellan321x


class TestDuan2010Magellan3212(unittest.TestCase):

    # define bounds to check the real part of the relative permittivity
    EPS_PRIME_R_DIFF_MIN = -5e-6
    EPS_PRIME_R_DIFF_MAX = 1.3e-4

    # define maximum number of s.d. the absorptivity should be away from Magellan
    NSD_ABSORPTIVITY_MAX = 3.4

    def test_model_magellan_data(self):
        # instantiate model and reference dataset
        model = Duan2010Verification()
        mgn_abs, mgn_rtpd = magellan321x.get_wavelength_orbit("X", 3212)
        mgn_epsprimer = mgn_rtpd["REFRACTIVITY"].to(dimensionless_unscaled) ** 2
        # check eps_prime_r difference at the Magellan altitudes
        with self.subTest("eps_prime_r difference"):
            eps_prime_r_diff = np.interp(
                mgn_rtpd["ALTITUDE"].to_value("km"),
                model.altitude.to_value("km"),
                model.relative_permittivity.real.to_value(dimensionless_unscaled),
            ) - mgn_epsprimer.to_value(dimensionless_unscaled)
            np.testing.assert_array_equal(
                np.logical_and(
                    eps_prime_r_diff > self.EPS_PRIME_R_DIFF_MIN,
                    eps_prime_r_diff < self.EPS_PRIME_R_DIFF_MAX,
                ),
                True,
            )
        # check absorption at the Magellan altitudes
        with self.subTest("absorptivity z-score"):
            absorptivity_zscore = (
                np.interp(
                    mgn_abs["ALTITUDE"].to_value("km"),
                    model.altitude.to_value("km"),
                    model.absorption.to_value("dB/km"),
                )
                - mgn_abs["ABSORPTIVITY"].to_value("dB/km")
            ) / mgn_abs["ABSORP_DEV"].to_value("dB/km")
            np.testing.assert_array_less(absorptivity_zscore, self.NSD_ABSORPTIVITY_MAX)
