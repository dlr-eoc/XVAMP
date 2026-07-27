"""
Test whether the model yields brightness temperatures consistent with the literature.
"""

# standard imports
import unittest
import numpy as np
from astropy.units import Quantity

# package imports
from xvamp.models.duan_et_al_2010 import Duan2010
from xvamp.geometry import get_brightness_temperature


class TestDuan2010Brightness(unittest.TestCase):

    # define surface emissivity [-] to test
    SURFACE_EMISSIVITY = 0.82

    def test_model_brightness_temperature(self):
        # instantiate model
        model = Duan2010()
        # get indices of -4, 0, 4 km altitude
        _list_alts = model.altitude.to_value("km").tolist()
        ix_alt = [_list_alts.index(h) for h in [-4.0, 0.0, 4.0]]
        # get surface brightness temperature
        T_B_surface = self.SURFACE_EMISSIVITY * model.temperature[ix_alt]
        # for each assumed element in T_B_surface, get the total brightness temperature
        T_B = Quantity(
            [
                get_brightness_temperature(
                    model.altitude[i:],
                    model.temperature[i:],
                    model.refraction[i:],
                    model.absorption[i:],
                    Quantity(30, "°"),
                    T_B_surface[ii],
                )
                for ii, i in enumerate(ix_alt)
            ]
        ).to_value("K")
        # check that it's inside the plausible range of 578-657 K
        np.testing.assert_array_equal(np.logical_and(T_B >= 578, T_B <= 657), True)
