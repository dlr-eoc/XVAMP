"""
Test whether the values saved previously by ``create_reference_output.py`` can be
recovered by the current version of the package.
"""

# standard imports
import unittest
import numpy as np
from astropy.units import Quantity
from astropy.table import QTable
from warnings import warn

# package imports
from xvamp.models.duan_et_al_2010 import Duan2010
from xvamp.geometry import geometry_from_central_angle, get_cross_track_displacement
from xvamp.utils.io import read_polarization_parameters


# helper function to compare quantities
def compare_quantities(actual: Quantity, desired: Quantity, name: str):
    """
    Helper function to compare two :class:`astropy.unit.Quantity`s, comparing
    the values in the same units to numerical precision but only warning if the
    units are different (as long as they're compatible).

    Parameters
    ----------
    actual
        Actual Quantity
    desired
        Desired Quantity
    name
        Name to be used in the unit warning

    Raises
    ------
    AssertionError
        If the two quantities do not match numerically (using the desired's units)
    """
    np.testing.assert_allclose(
        actual.to_value(desired.unit),
        desired.to_value(),
        rtol=1e-14,
        atol=1e-14,
    )
    if actual.unit != desired.unit:
        warn(f"{name}: actual unit {actual.unit} != {desired.unit}")


class TestDuan2010(unittest.TestCase):

    # tables to load
    # (quantities.fits and profile_values.fits are considered separately)
    QTABLES = [
        "absorptions",
        "mass_densities",
        "molar_densities",
        "molar_fractions",
        "polarizations",
    ]

    # quantities to check
    QUANTITIES = [
        "altitude",
        "temperature",
        "pressure",
        "mass_density",
        "electron_density",
        "cloud_mass_density",
        "cloud_concentration",
        "number_density",
        "molar_density",
        "polarization",
        "absorption",
        "eps_prime_r_atmo",
        "eps_prime_r_iono",
        "relative_permittivity",
        "refraction",
    ]

    # predefined viewing geometries
    TEST_LOOK_ANGLE = Quantity(np.linspace(28, 32, num=5), "deg")
    TEST_HEIGHT_TERRAIN = Quantity(np.linspace(-6, 16, num=23), "km")
    TEST_HEIGHT_PLATFORM = Quantity(220, "km")
    TEST_GRID_TERRAIN, TEST_GRID_LOOK = np.meshgrid(
        TEST_HEIGHT_TERRAIN.to_value("km"), TEST_LOOK_ANGLE.to_value("rad")
    )

    @classmethod
    def setUpClass(cls):

        # load all previously-saved outputs
        for ff in cls.QTABLES + ["quantities", "profile_values"]:
            setattr(cls, f"_{ff}", QTable.read(f"reference_output/{ff}.fits"))
        cls._polarization_parameters = read_polarization_parameters(
            "reference_output/polarization_parameters.toml"
        )

        # load default packaged polarization parameters
        cls._default_polarization_parameters = read_polarization_parameters()

        # create our own model
        cls._model = Duan2010(load_polarization_parameters=False)

    def test_polarization_parameters(self):
        # all parameters in the model should match the packaged ones,
        # and enforce that they are the same
        self.assertEqual(
            set(list(self._model.polarization_parameters)),
            set(list(self._polarization_parameters)),
        )
        for species, pp in self._model.polarization_parameters.items():
            with self.subTest(f"species {species}"):
                self.assertEqual(pp, self._polarization_parameters[species])

    def test_packaged_polarization_parameters(self):
        # all parameters in the model should match the packaged ones,
        # but ignore that the packaged one may contain more
        for species, pp in self._model.polarization_parameters.items():
            with self.subTest(f"species {species}"):
                self.assertEqual(pp, self._default_polarization_parameters[species])

    def test_quantities(self):
        # all saved Quantities should match the ones in the model
        for qname in self.QUANTITIES:
            with self.subTest(qname):
                q_actual: Quantity = getattr(self._model, qname)
                q_desired: Quantity = self._quantities[qname]
                compare_quantities(q_actual, q_desired, qname)

    def test_qtables(self):
        # all saved QTables should match the ones in the model
        for qtname in self.QTABLES:
            with self.subTest(qtname):
                qt_actual: QTable = getattr(self._model, qtname)
                qt_desired: QTable = getattr(self, f"_{qtname}")
                self.assertEqual(set(qt_desired.colnames), set(qt_actual.colnames))
                for col in qt_desired.colnames:
                    compare_quantities(
                        qt_actual[col], qt_desired[col], f"{qtname} -> {col}"
                    )

    def test_integrated_values(self):
        # the computed profile-integrated values should all match
        compare_quantities(
            Quantity(self.TEST_GRID_LOOK.ravel(), "rad"),
            self._profile_values["col0"],
            "grid_look",
        )
        compare_quantities(
            Quantity(self.TEST_GRID_TERRAIN.ravel(), "km"),
            self._profile_values["col1"],
            "grid_terrain",
        )

        # get profile-integrated values
        apparent_range, attenuation, central_angle, apparent_incidence_angle = (
            self._model.get_range_attenuation_angles(
                self.TEST_GRID_LOOK.ravel(),
                self.TEST_GRID_TERRAIN.ravel(),
                self.TEST_HEIGHT_PLATFORM,
            )
        )
        with self.subTest("get_range_attenuation_angles"):
            compare_quantities(
                apparent_range, self._profile_values["col2"], "apparent_range"
            )
            compare_quantities(attenuation, self._profile_values["col3"], "attenuation")
            compare_quantities(
                central_angle, self._profile_values["col4"], "central_angle"
            )
            compare_quantities(
                apparent_incidence_angle,
                self._profile_values["col5"],
                "apparent_incidence_angle",
            )

        # use law of cosines to get geometric quantities
        geometric_range, geometric_look_angle, geometric_incidence_angle = (
            geometry_from_central_angle(
                central_angle, self.TEST_GRID_TERRAIN.ravel(), self.TEST_HEIGHT_PLATFORM
            )
        )
        with self.subTest("geometry_from_central_angle"):
            compare_quantities(
                geometric_range, self._profile_values["col6"], "geometric_range"
            )
            compare_quantities(
                geometric_look_angle,
                self._profile_values["col7"],
                "geometric_look_angle",
            )
            compare_quantities(
                geometric_incidence_angle,
                self._profile_values["col8"],
                "geometric_incidence_angle",
            )

        # use it again to get the cross-track displacement
        cross_track_displacement = get_cross_track_displacement(
            np.repeat(
                self.TEST_LOOK_ANGLE[:, None], self.TEST_HEIGHT_TERRAIN.size, axis=1
            ).ravel(),
            central_angle,
            self.TEST_GRID_TERRAIN.ravel(),
            self.TEST_HEIGHT_PLATFORM,
        )
        with self.subTest("get_cross_track_displacement"):
            compare_quantities(
                cross_track_displacement,
                self._profile_values["col9"],
                "cross_track_displacement",
            )
