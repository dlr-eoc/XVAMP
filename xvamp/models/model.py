"""
Module containing the abstract base class for atmospheric models.
"""

# standard imports
import numpy as np
import astropy.units as u
import astropy.table as astrotable
from typing import Tuple
from astropy.units import Quantity, Unit
from numpy.polynomial import Polynomial
from scipy.integrate import cumulative_trapezoid

# package imports
from ..constants import VENUS_GRAV_PARAM, VENUS_RADIUS
from ..utils import float_or_array
from ..geometry import geometry_from_central_angle
from ..utils.parametersets import (
    HarveyLemmon2005Parameters,
    Pitzer1983Parameters,
)
from ..reference import seiff1985


class Model:
    """
    Abstract base class for final models.
    """

    # general atmospheric properties
    altitude: Quantity["length"]
    """ Altitude levels of the model """
    temperature: Quantity["temperature"]
    """ Temperature levels of the model """
    pressure: Quantity["pressure"]
    """ Pressure levels of the model """

    # mixture quantities
    mass_density: Quantity["mass density"]
    """ Mass density of the model """
    molar_density: Quantity["molar concentration"]
    """ Molar density of the model """
    number_density: Quantity["number density"]
    """ Number density of the model """
    electron_density: Quantity["number density"]
    """ Electron density of the model """
    polarization: Quantity["dimensionless"]
    """ Polarization of the model """
    absorption: Quantity["wavenumber"]
    """ Absorption of the model """
    eps_prime_r_atmo: Quantity["dimensionless"]
    """ Real part of the relative permittivity of the atmosphere """
    eps_prime_r_iono: Quantity["dimensionless"]
    """ Real part of the relative permittivity of the ionosphere """
    relative_permittivity: Quantity["dimensionless"]
    """ Complex relative model permittivity """
    refraction: Quantity["dimensionless"]
    """ Index of refraction of the model """

    # cloud quantities
    cloud_mass_density: Quantity["mass density"]
    """ Mass density of the clouds """
    cloud_concentration: Quantity["%"]
    """ H2SO4 concentration by weight of the cloud droplets """

    # component quantities
    polarization_parameters: dict[
        str, HarveyLemmon2005Parameters | Pitzer1983Parameters
    ]
    """ Dictionary that containes the parameter objects for each species """
    molar_fractions: astrotable.QTable
    """ Table with species molar fractions """
    mass_densities: astrotable.QTable
    """ Table with species mass densities """
    molar_densities: astrotable.QTable
    """ Table with species molar densities """
    polarizations: astrotable.QTable
    """ Table with species polarizations """
    absorptions: astrotable.QTable
    """ Table with species absorptions """

    def get_interpolated_attribute(
        self,
        attribute: str,
        unit: Unit,
        altitude: Quantity | float_or_array,
        left: float,
        right: float,
    ) -> Quantity:
        """
        Retrieve an attribute profile and interpolate it.

        Parameters
        ----------
        attribute
            Name of the attribute
        unit
            Unit to be used for the interpolation
        altitude
            Height in [km], if not a :class:`~astropy.units.Quantity`
        left
            Value to use below the available data of the profile
        right
            Value to use above the available data of the profile

        Returns
        -------
            Interpolated attribute profile
        """
        # get existing profile of the attribute
        attr = getattr(self, attribute)
        # format the altitude input levels
        if isinstance(altitude, Quantity):
            altitude = altitude.to("km").value
        alt = np.atleast_1d(altitude)
        assert np.all(np.diff(alt) >= 0)
        # interpolate
        prof = np.interp(
            alt,
            self.altitude.to("km").value,
            attr.to(unit).value,
            left=left,
            right=right,
        )
        # return with unit
        return Quantity(prof, unit)

    def get_refraction(self, altitude: Quantity | float_or_array) -> Quantity:
        """
        Return the index of refraction at specific altitudes.
        At altitudes below the defined dataset, this function will return ``NaN``,
        and ``1`` above.

        Parameters
        ----------
        altitude
            Height in [km], if not a :class:`~astropy.units.Quantity`

        Returns
        -------
            Refractive index [-]
        """
        return self.get_interpolated_attribute(
            "refraction", u.dimensionless_unscaled, altitude, np.nan, 1
        )

    def get_absorption(self, altitude: Quantity | float_or_array) -> Quantity:
        """
        Return the absorption at specific altitudes.
        At altitudes below the defined dataset, this function will return ``NaN``,
        and ``0`` above.

        Parameters
        ----------
        altitude
            Height in [km], if not a :class:`~astropy.units.Quantity`.

        Returns
        -------
            Absorption [dB/km]
        """
        return self.get_interpolated_attribute(
            "absorption", Unit("dB/km"), altitude, np.nan, 0
        )

    def get_temperature(self, altitude: Quantity | float_or_array) -> Quantity:
        """
        Return the temperature at specific altitudes.
        At altitudes below the defined dataset, this function will return ``NaN``,
        and ``0`` above.

        Parameters
        ----------
        altitude
            Height in [km], if not a :class:`~astropy.units.Quantity`.

        Returns
        -------
            Temperature [K]
        """
        return self.get_interpolated_attribute("temperature", u.K, altitude, np.nan, 0)

    def get_range_attenuation_angles(
        self,
        look_angle: Quantity["angle"] | float_or_array,
        height_terrain: Quantity["length"] | float_or_array,
        height_platform: Quantity["length"] | float,
    ) -> Tuple[
        Quantity["length"], Quantity["dB"], Quantity["angle"], Quantity["angle"]
    ]:
        """
        Calculate the apparent range, two-way attenuation through the atmosphere,
        the central angle, and the apparent incidence angle for a range of look angles,
        terrain heights, and platform heights.

        Parameters
        ----------
        look_angle
            Look angle(s) of the instrument in [rad], if not a
            :class:`~astropy.units.Quantity`
        height_terrain
            Height(s) of the terrain relative to the mean planet radius in [km],
            if not a :class:`~astropy.units.Quantity`
        height_platform
            Height(s) of the platform relative to the mean planet radius in [km],
            if not a :class:`~astropy.units.Quantity`

        Returns
        -------
        apparent_range
            Apparent range from the platform to the surface [km]
        attenuation
            Two-way signal attenuation [dB] (note that the *power absorption* is twice
            this value)
        central_angle
            Central angle [rad]
        apparent_incidence_angle
            Apparent incidence angle [rad]
        """
        # input format
        if isinstance(look_angle, Quantity):
            look_angle = look_angle.to("rad").value
        if isinstance(height_terrain, Quantity):
            height_terrain = height_terrain.to("km").value
        if isinstance(height_platform, Quantity):
            height_platform = height_platform.to("km").value
        look_angle = np.atleast_1d(look_angle)
        assert look_angle.ndim == 1
        height_terrain = np.atleast_1d(height_terrain)
        assert height_terrain.ndim == 1
        height_platform = np.atleast_1d(height_platform)
        assert height_platform.ndim == 1
        venus_radius = VENUS_RADIUS.to("km").value
        height_model = self.altitude.to("km").value
        # get output size
        nout = np.max([look_angle.size, height_terrain.size, height_platform.size])
        assert look_angle.size in [
            1,
            nout,
        ], f"{look_angle.size=}, expected 1 or {nout}."
        assert height_terrain.size in [
            1,
            nout,
        ], f"{height_terrain.size=}, expected 1 or {nout}."
        assert height_platform.size in [
            1,
            nout,
        ], f"{height_platform.size=}, expected 1 or {nout}."
        nh_model = height_model.size
        nh_terrain = height_terrain.size
        nh_platform = height_platform.size
        # get a joint altitude array that contains all model altitudes, as well
        # as all terrain and platform altitudes
        altitudes, altitude_indices = np.unique(
            np.r_[height_model, height_terrain, height_platform], return_inverse=True
        )
        # interpolate refraction and absorption to this new altitude array
        refractions = self.get_refraction(altitudes)
        absorptions = self.get_absorption(altitudes)
        # get the indices that for each terrain and platform height return the altitude
        # range over which to integrate
        index_pairs = np.array(
            [
                altitude_indices[
                    [int(iterrain) + nh_model, int(iplatform) + nh_model + nh_terrain]
                ]
                for iterrain, iplatform in zip(
                    np.arange(nout) if nh_terrain > 1 else np.zeros(nout),
                    np.arange(nout) if nh_platform > 1 else np.zeros(nout),
                )
            ]
        )
        # make a mask that can be used to set the output to NaN
        # where the input is invalid
        invalid = ~np.logical_and(np.isfinite(refractions), np.isfinite(absorptions))
        mask = np.array(
            [
                np.any(invalid[index_pairs[i, 0] : index_pairs[i, 1] + 1])
                for i in range(nout)
            ]
        )
        # compute index of refraction at platform altitudes
        refraction_0 = refractions[altitude_indices[-nh_platform:]]
        # compute cosine of look angle for all platform and evaluation altitudes
        sine_look_angle = (
            (venus_radius + height_platform[:, None])
            / (venus_radius + altitudes[None, :])
            * (refraction_0[:, None] / refractions[None, :])
            * np.sin(look_angle[:, None])
        )
        cosine_look_angle = np.sqrt(1 - sine_look_angle**2)
        tangent_look_angle = sine_look_angle / cosine_look_angle
        # the integrand for the apparent range
        d_rho_a_dz = refractions[None, :] / cosine_look_angle
        # the integrand for the attenuation
        d_alpha_L_dz = 2 * absorptions[None, :] / cosine_look_angle
        # the integrand for the central angle
        d_beta_dz = tangent_look_angle / Quantity(
            venus_radius + altitudes[None, :], "km"
        )
        # the integrand for the apparent incidence angle
        d_theta_dz = tangent_look_angle
        # cumulatively integrate to get solutions for all starting altitudes
        cumu_rho_a = cumulative_trapezoid(
            np.nan_to_num(d_rho_a_dz), x=altitudes, axis=1, initial=0
        )
        cumu_alpha_L = cumulative_trapezoid(
            np.nan_to_num(d_alpha_L_dz), x=altitudes, axis=1, initial=0
        )
        cumu_beta = cumulative_trapezoid(
            np.nan_to_num(d_beta_dz), x=altitudes, axis=1, initial=0
        )
        cumu_theta = cumulative_trapezoid(
            np.nan_to_num(d_theta_dz), x=altitudes, axis=1, initial=0
        )
        # extract the values at the start at end point of the cumulative integration
        rho_a_from = np.take_along_axis(
            cumu_rho_a, index_pairs[:, 0][:, None], axis=1
        ).ravel()
        rho_a_to = np.take_along_axis(
            cumu_rho_a, index_pairs[:, 1][:, None], axis=1
        ).ravel()
        alpha_L_from = np.take_along_axis(
            cumu_alpha_L, index_pairs[:, 0][:, None], axis=1
        ).ravel()
        alpha_L_to = np.take_along_axis(
            cumu_alpha_L, index_pairs[:, 1][:, None], axis=1
        ).ravel()
        beta_from = np.take_along_axis(
            cumu_beta, index_pairs[:, 0][:, None], axis=1
        ).ravel()
        beta_to = np.take_along_axis(
            cumu_beta, index_pairs[:, 1][:, None], axis=1
        ).ravel()
        theta_from = np.take_along_axis(
            cumu_theta, index_pairs[:, 0][:, None], axis=1
        ).ravel()
        theta_to = np.take_along_axis(
            cumu_theta, index_pairs[:, 1][:, None], axis=1
        ).ravel()
        # set values to NaN if their interval contains any invalid inputs
        # (one side is enough)
        rho_a_from[mask] = np.nan
        alpha_L_from[mask] = np.nan
        beta_from[mask] = np.nan
        theta_from[mask] = np.nan
        # difference the two to get final integration value
        apparent_range = Quantity(rho_a_to - rho_a_from, "km")
        attenuation = Quantity(alpha_L_to - alpha_L_from, "dB")
        central_angle = Quantity(beta_to - beta_from, "rad")
        apparent_incidence_angle = Quantity(
            np.arctan((theta_to - theta_from) / (height_platform - height_terrain)),
            "rad",
        )
        # done
        return apparent_range, attenuation, central_angle, apparent_incidence_angle

    def get_delay_attenuation(
        self,
        height_terrain: Quantity | float_or_array,
        height_platform: Quantity | float_or_array,
        look_angle: Quantity | float_or_array,
    ) -> Tuple[Quantity, Quantity]:
        """
        Calculate the range delay (defined as the difference between the apparent and
        geometric range) and two-way attenuation through the atmosphere.
        Convenience wrapper around :meth:`~Model.get_range_attenuation_angles`
        and :func:`~xvamp.utils.geometry_from_central_angle`.

        Parameters
        ----------
        height_terrain
            Height of the terrain relative to the mean planet radius in [km],
            if not a :class:`~astropy.units.Quantity`
        height_platform
            Height of the platform relative to the mean planet radius in [km],
            if not a :class:`~astropy.units.Quantity`
        look_angle
            Look angle of the instrument in [rad], if not a
            :class:`~astropy.units.Quantity`

        Returns
        -------
        delay
            Range delay [m]
        attenuation
            Two-way signal attenuation [dB]
        """
        # get profile-integrated values
        apparent_range, attenuation, central_angle = self.get_range_attenuation_angles(
            look_angle, height_terrain, height_platform
        )[:3]
        # use law of cosines to get geometric range
        geometric_range = geometry_from_central_angle(
            central_angle, height_terrain, height_platform
        )[0]
        # get delay
        delay = (apparent_range - geometric_range).to("m")
        # done
        return delay, attenuation

    @staticmethod
    def rel_permittivity_to_refraction(
        relative_permittivity: Quantity | float_or_array,
    ) -> Quantity | float_or_array:
        """
        Compute the index of refraction from the complex relative
        permittivity.

        Parameters
        ----------
        relative_permittivity
            Complex relative permittivity [-]

        Returns
        -------
            Index of refraction [-]
        """
        return np.sqrt(
            (
                np.sqrt(relative_permittivity.real**2 + relative_permittivity.imag**2)
                + relative_permittivity.real
            )
            / 2
        )

    @staticmethod
    def tpd_below_0km(venus_gas_constant: Quantity, add_3K: bool = False) -> Tuple[
        Quantity["length"],
        Quantity["temperature"],
        Quantity["pressure"],
        Quantity["mass density"],
    ]:
        """
        Use the barometric formula to extend the near-surface temperature,
        pressure, and density profiles from :cite:t:`seiff1985` to negative
        altitudes.

        Parameters
        ----------
        venus_gas_constant
            Assumed Venus standard atmospheric gas constant (= R/M) [J/kg K]
        add_3K
            This refers to the 3 K addition done in the :cite:t:`duan2010` model
            when combining the :cite:t:`seiff1985` and :cite:t:`zasova2006` profiles.

        Returns
        -------
        alt_neg
            Altitudes of the profile
        temp_neg
            Temperature
        press_neg
            Pressure
        dens_neg
            Mass density
        """
        # since we have negative altitudes, we need to extrapolate:
        # we can use the lapse rate and compressibility together with the barometric
        # formula to extend the temperature and pressure profiles
        # first: extend temperature by assuming a linearly-continuing lapse rate
        # from 5 km and below
        alt_help_w0 = Quantity(np.arange(-7000, 1, dtype=int) / 1e3, "km")
        lapse_rate_neg = Quantity(
            Polynomial.fit(
                seiff1985.tables["1-1"]["z"][:3].value,
                seiff1985.tables["1-1"]["Γ"][:3].value,
                deg=1,
            )(alt_help_w0.value),
            seiff1985.tables["1-1"]["Γ"][:3].unit,
        )
        temp_help_w0 = Quantity(
            cumulative_trapezoid(-lapse_rate_neg, x=alt_help_w0, initial=0),
            lapse_rate_neg.unit * alt_help_w0.unit,
        )
        temp_help_w0 -= temp_help_w0[-1] - seiff1985.tables["1-1"]["T"][0]
        if add_3K:
            temp_help_w0 -= 3 * u.K
        temp_neg = temp_help_w0[:-1:1000]
        # then, extend the imperfect gas compressibiltiy factor
        zeta_neg = Quantity(
            Polynomial.fit(
                seiff1985.tables["1-1"]["z"][:3].value,
                seiff1985.tables["1-1"]["ς"][:3].value,
                deg=1,
            )(alt_help_w0.value),
            seiff1985.tables["1-1"]["ς"][:3].unit,
        )
        # then, use the barometric formula to get pressure
        g_help_w0 = VENUS_GRAV_PARAM / (VENUS_RADIUS + alt_help_w0) ** 2
        baro_int_values = g_help_w0[::-1] / (
            venus_gas_constant * zeta_neg[::-1] * temp_help_w0[::-1]
        )
        baro_int = Quantity(
            cumulative_trapezoid(baro_int_values, x=alt_help_w0[::-1], initial=0)[::-1],
            baro_int_values.unit * alt_help_w0.unit,
        )
        press_neg = seiff1985.tables["1-1"]["p"][0] * np.exp(-baro_int[:-1:1000])
        # use the compressible ideal gas law to get density
        dens_neg = (
            press_neg / (zeta_neg[:-1:1000] * venus_gas_constant * temp_neg)
        ).decompose()
        # done
        alt_neg = alt_help_w0[:-1:1000]
        return alt_neg, temp_neg, press_neg, dens_neg
