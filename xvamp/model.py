"""
Model class that loads all the reference data, maybe adds its own,
and returns permittivity.
"""

# standard imports
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.table as astrotable
from pathlib import Path
from typing import Tuple
from collections.abc import Callable
from astropy.units import Quantity, Unit
from numpy.typing import NDArray
from numpy.polynomial import Polynomial
from scipy.integrate import cumulative_trapezoid

# package imports
from .constants import *
from .utils import (
    float_or_array,
    fill_df,
    read_polarization_parameters,
    HarveyLemmon2005Parameters,
    Pitzer1983Parameters,
    BenReuvenParameters,
)
from .reference import (
    cimino1982,
    duan2010figures,
    james1997,
    jplspectrallines,
    keating1985,
    kolodnersteffes1998,
    marcq2006,
    paetzold2007,
    seiff1985,
    seiffkeating,
    zasova2006,
)


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
        # input format
        if isinstance(altitude, Quantity):
            altitude = altitude.to("km").value
        alt = np.atleast_1d(altitude)
        assert np.all(np.diff(alt) >= 0)
        # interpolate where we can, fill according to direction elsewhere
        n = np.interp(
            alt,
            self.altitude.to("km").value,
            self.refraction.to(u.dimensionless_unscaled).value,
            left=np.nan,
            right=1,
        )
        return Quantity(n, u.dimensionless_unscaled)

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
        # input format
        if isinstance(altitude, Quantity):
            altitude = altitude.to("km").value
        alt = np.atleast_1d(altitude)
        assert np.all(np.diff(alt) >= 0)
        # interpolate where we can, fill according to direction elsewhere
        alpha = np.interp(
            alt,
            self.altitude.to("km").value,
            self.absorption.to("dB/km").value,
            left=np.nan,
            right=0,
        )
        return Quantity(alpha, "dB/km")

    def get_range_attenuation_angle(
        self,
        height_terrain: Quantity | float_or_array,
        height_platform: Quantity | float_or_array,
        look_angle: Quantity | float,
    ) -> Tuple[Quantity, Quantity, Quantity]:
        """
        Calculate the apparent range, two-way attenuation through the atmosphere,
        and the central angle.

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
        apparent_range
            Apparent range from the platform to the surface [km]
        attenuation
            Two-way signal attenuation [dB]
        central_angle
            Central angle [rad]
        """
        # input format
        if isinstance(height_terrain, Quantity):
            height_terrain = height_terrain.to("km").value
        if isinstance(height_platform, Quantity):
            height_platform = height_platform.to("km").value
        if isinstance(look_angle, Quantity):
            look_angle = look_angle.to("rad").value
        assert look_angle.size == 1
        venus_radius = VENUS_RADIUS.to("km").value
        height_model = self.altitude.to("km").value
        # get sizes to loop over
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
                [
                    altitude_indices[
                        [iterrain + nh_model, iplatform + nh_model + nh_terrain]
                    ]
                    for iterrain in range(nh_terrain)
                ]
                for iplatform in range(nh_platform)
            ]
        )
        # make a mask that can be used to set the output to NaN
        # where the input is invalid
        invalid = ~np.logical_and(np.isfinite(refractions), np.isfinite(absorptions))
        mask = np.array(
            [
                [
                    np.any(
                        invalid[
                            index_pairs[iplatform, iterrain, 0] : index_pairs[
                                iplatform, iterrain, 1
                            ]
                            + 1
                        ]
                    )
                    for iterrain in range(nh_terrain)
                ]
                for iplatform in range(nh_platform)
            ]
        )
        # compute index of refraction at platform altitude
        refraction_0 = refractions[altitude_indices[-nh_platform:]]
        # compute cosine of look angle for all platform and evaluation altitudes
        sine_look_angle = (
            (venus_radius + height_platform[:, None])
            / (venus_radius + altitudes[None, :])
            * (refraction_0[:, None] / refractions[None, :])
            * np.sin(look_angle)
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
        # extract the values at the start at end point of the cumulative integration
        rho_a_from = np.take_along_axis(cumu_rho_a, index_pairs[:, :, 0], axis=1)
        rho_a_to = np.take_along_axis(cumu_rho_a, index_pairs[:, :, 1], axis=1)
        alpha_L_from = np.take_along_axis(cumu_alpha_L, index_pairs[:, :, 0], axis=1)
        alpha_L_to = np.take_along_axis(cumu_alpha_L, index_pairs[:, :, 1], axis=1)
        beta_from = np.take_along_axis(cumu_beta, index_pairs[:, :, 0], axis=1)
        beta_to = np.take_along_axis(cumu_beta, index_pairs[:, :, 1], axis=1)
        # set values to NaN if their interval contains any invalid inputs
        # (one side is enough)
        rho_a_from[mask] = np.nan
        alpha_L_from[mask] = np.nan
        beta_from[mask] = np.nan
        # difference the two to get final integration value
        apparent_range = Quantity(rho_a_to - rho_a_from, "km")
        attenuation = Quantity(alpha_L_to - alpha_L_from, "dB")
        central_angle = Quantity(beta_to - beta_from, "rad")
        # done
        return apparent_range, attenuation, central_angle

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
    def tpd_below_0km(venus_gas_constant: Quantity) -> Tuple[
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
        temp_help_w0 -= temp_help_w0[-1] - (seiff1985.tables["1-1"]["T"][0] + 3 * u.K)
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


# Model class that implements the Duan et al. (2010) paper
class Duan2010(Model):

    # general constants
    VENUS_GAS_CONSTANT = Quantity(191.4, "J/kg K")
    """ Venus standard atmospheric gas constant (= R/M) [J/kg K] """
    VENUS_STANDARD_CO2 = Quantity(0.965, u.dimensionless_unscaled)
    """ Venus standard CO2 molar fraction [-] """
    VENUS_STANDARD_N2 = Quantity(0.035, u.dimensionless_unscaled)
    """ Venus standard N2 molar fraction [-] """
    VENUS_MOLAR_MASS = (
        VENUS_STANDARD_CO2 * SPEC_MOL_M["CO2"] + VENUS_STANDARD_N2 * SPEC_MOL_M["N2"]
    ).to("kg/mol")
    """ Venus standard atmospheric molar mass [kg/mol] """
    TRANSITION_ATMO_IONO = Quantity(100, "km")
    """
    Altitude at which the computation of the real part of the relative
    permittivity switches from the individual components in the
    atmosphere to the overall effect of the ionosphere
    """

    # constants relating to Argon (Ar)
    MR_AR = Quantity(7e-5, u.dimensionless_unscaled)
    """ Mixing ratio of Argon below 100 km from von :cite:t:`vonzahn1985` """
    HLP_AR = HarveyLemmon2005Parameters(4.1414, 0.0, 1.597, 0.262, -117.9, 0.0, 2.1)
    """ Mixture parameters for Ar in cgs units """

    # constants relating to carbon monoxide (CO)
    EPS_PRIME_R_CO = Quantity(1.000634, u.dimensionless_unscaled)
    """ X-band estimated dielectric constant of CO at 1 atm and 0 °C"""
    P_CO = Quantity(101325, "Pa")
    """ Pressure at which the dielectric constant for CO was calculated """
    T_CO = Quantity(298, "K")
    """ Temperature at which the dielectric constant for CO was calculated """
    RHO_CO = ((P_CO) / (GAS_CONSTANT * T_CO)).decompose()
    """ Molar density from ``P_CO`` and ``T_CO`` """
    MU_CO = Quantity(0.112e-18, ESU_CM)
    """ Permanent dipole moment of CO [esu cm] """

    # constants relating to carbon dioxide (CO2)
    HLP_CO2 = HarveyLemmon2005Parameters(
        7.3455, 0.00335, 83.93, 145.1, -578.8, -1012.0, 1.55
    )
    """ Mixture parameters for CO2 in cgs units """

    # constants relating to water vapor (H2O)
    PP_water_vapor = Pitzer1983Parameters(
        Quantity(1.84e-18, ESU_CM),
        alpha_T=Quantity(1.444e-24, "cm3"),
    )
    """ Polarization parameters for water vapor """

    # constants relating to gaseous sulfuric acid (H2SO4)
    # are all defined in the KolodnerSteffes1998 class

    # constants relating to nitrogen (N2)
    HLP_N2 = HarveyLemmon2005Parameters(
        4.3872, 0.00226, 2.206, 1.135, -169.0, -35.83, 2.1
    )
    """ Mixture parameters for N2 in cgs units """

    # constants relating to carbonyl sulfide (OCS or COS)
    EPS_PRIME_R_INF_OCS = Quantity(1.0031248, u.dimensionless_unscaled)
    """ Estimated dielectric constant of SO2 at infinite frequency """
    P_OCS = Quantity(101325, "Pa")
    """ Pressure at which the dielectric constant for OCS was calculated """
    T_OCS = Quantity(273.18, "K")
    """ Temperature at which the dielectric constant for OCS was calculated """
    RHO_OCS = ((P_OCS) / (GAS_CONSTANT * T_OCS)).decompose()
    """ Molar density from ``P_OCS`` and ``T_OCS`` """
    MU_OCS = Quantity(0.71521e-18, ESU_CM)
    """ Permanent dipole moment of OCS [esu cm] """
    BR_OCS_CO2 = BenReuvenParameters(
        T_0=Quantity(300, "K"),
        gamma_min_maj=Quantity(7.2, "MHz/torr"),
        gamma_min_min=Quantity(16, "MHz/torr"),
        zeta_min_maj=Quantity(0, "MHz/torr"),
        zeta_min_min=Quantity(0, "MHz/torr"),
        delta_min=Quantity(0, "MHz/torr"),
        m=0.85,
        n=0.85,
    )
    """ Ben-Reuven line parameters for OCS in CO2 """

    # constants relating to sulfur dioxide (SO2)
    EPS_PRIME_R_INF_SO2 = Quantity(1.00587032, u.dimensionless_unscaled)
    """ Estimated dielectric constant of SO2 at infinite frequency """
    P_SO2 = Quantity(101325, "Pa")
    """ Pressure at which the dielectric constant for SO2 was calculated """
    T_SO2 = Quantity(273.15, "K")
    """ Temperature at which the dielectric constant for SO2 was calculated """
    RHO_SO2 = ((P_SO2) / (GAS_CONSTANT * T_SO2)).decompose()
    """ Molar density from ``P_SO2`` and ``T_SO2`` """
    MU_SO2 = Quantity(1.633e-18, ESU_CM)
    """ Permanent dipole moment of SO2 [esu cm] """
    BR_SO2_CO2 = BenReuvenParameters(
        T_0=Quantity(300, "K"),
        gamma_min_maj=Quantity(7.2, "MHz/torr"),
        gamma_min_min=Quantity(16, "MHz/torr"),
        zeta_min_maj=Quantity(1.3, "MHz/torr"),
        zeta_min_min=Quantity(1.6, "MHz/torr"),
        delta_min=Quantity(2.9, "MHz/torr"),
        m=0.85,
        n=0.85,
    )
    """ Ben-Reuven line parameters for SO2 in CO2 """

    # other constants

    # pressure extrapolation coefficients
    EXT_PRESSURE_COEFFS = [
        11.201473859081256,
        0.006260686643162,
        -9.240397971368619,
        0.010200118486472,
    ]
    """
    Coefficients fit to a douple exponential function to extrapolate
    pressure [log10(atm)] from altitude [km], taken from the reference code
    """

    # Cimino (1982) Figs. 7-9 as lookup tables
    EPS_PRIME_R_H2SO4 = np.r_[
        np.linspace(80, 59, 11),
        np.linspace(57.5, 45, 11),
        np.linspace(43.5, 38, 11),
        np.linspace(37.5, 35.5, 11),
        np.linspace(35.0, 34.5, 10),
        np.linspace(34.5, 34.5, 10),
        np.linspace(34.5, 34, 10),
        np.linspace(34.0, 31.5, 8),
        np.linspace(31.0, 24.5, 9),
        np.linspace(24.0, 17.5, 9),
        17,
    ]
    """
    Real part of the relative permittivity for H2SO4 at 2650 MHz
    for concentrations between 0% and 100% [-]
    """
    EPS_DPRIME_R_H2SO4 = np.r_[
        np.linspace(0, 235, 10),
        np.linspace(248.5, 305, 5),
        np.linspace(320, 370, 5),
        np.linspace(375, 420, 11),
        np.linspace(420, 385, 11),
        np.linspace(377, 310, 9),
        np.linspace(302, 230, 10),
        np.linspace(222, 140, 10),
        np.linspace(134, 90, 10),
        np.linspace(90, 75, 5),
        np.linspace(75, 75, 5),
        np.linspace(75, 65, 5),
        np.linspace(61, 50, 5),
    ]
    """
    Imaginary part of the relative permittivity for H2SO4 at 2650 MHz
    for concentrations between 0% and 100% [-]
    """

    def __init__(
        self,
        use_compressible_gas: bool = True,
        use_keating_temp_press_above100km: bool = False,
        use_keating_co_co2_n2_above_100km: bool = False,
        use_kolste_h2so4: bool = False,
        use_marcq_ocs: bool = False,
        add_ar: bool = False,
        cutoff_so2_frequency: Quantity["frequency"] | None = None,
        use_kolbe_ocs: bool = False,
        use_virial_approximation: bool = True,
        use_cimino_clouds: bool = True,
        use_cimino_fitted_lookup: bool = False,
        load_polarization_parameters: bool | str | Path = True,
        min_altitude_spacing: Quantity = Quantity(1, "km"),
    ) -> None:
        """
        Initialize the :cite:t:`duan2010` model. All parameters are set such that
        they correspond to the Matlab implementation of the model.

        Parameters
        ----------
        use_compressible_gas
            Whether to use the gas compressibility factor when deriving the mass
            density for the 0-100 km altitude range, or assume the ideal gas law.
            This has no effect on the model, since all species quantities are
            derived from the pressure profile, which is directly loaded from
            :cite:t:`seiff1985` and :cit:t:`zasova2006`.
        use_keating_temp_press_above100km
            Whether to use the temperature profile from :cite:t:`keating1985`
            above 100 km, and get its matching pressure profile from the
            ideal gas law.
            This option has no effect on the model, since the transition between
            atmosphere- and ionosphere-dominated permittivity profiles is at 100 km,
            and the ionosphere is modeled differently. It is only useful if one
            wants to load these quantities for later plotting.
        use_keating_co_co2_n2_above_100km
            Whether to use the :cite:t:`keating1985` mixing ratios for
            CO, CO2, and N2 as continuation above the :cite:t:`duan2010`
            profiles (instead of continuing CO2 and N2 upwards as a constant,
            and setting CO to zero upwards of the highest value).
            This option has no effect on the model, since the transition between
            atmosphere- and ionosphere-dominated permittivity profiles is at 100 km,
            and the ionosphere is modeled differently. It is only useful if one
            wants to load these quantities for later plotting.
        use_kolste_h2so4
            Whether to use the H2SO4 profile from :cite:t:`kolodner1998`
            instead of the :cite:t:`duan2010` profile.
            This has a range delay effect on the millimeter scale, and an effect
            on the two-way attenuation on the centidecibel scale.
        use_marcq_ocs
            Whether to use the OCS profile from :cite:t:`marcq2006`
            instead of the :cite:t:`duan2010` profile.
            This has a range delay effect on the sub-millimeter scale, and an effect
            on the two-way attenuation on the millidecibel scale.
        add_ar
            Whether to add a constant value for Argon into the mixture.
            This has a range delay effect on the sub-micrometer scale, and an effect
            on the two-way attenuation on the tens of microdecibel scale.
        cutoff_so2_frequency
            When computing the absorption coefficient of SO2, include all spectral
            lines up to this frequency. If ``None``, use all available ones.
            This option is only kept for development purposes.
        use_kolbe_ocs
            Whether to use the :cite:t:`kolbe1977`, Lorentzian-based approach to
            compute the absorption coefficient from OCS, or not.
        use_virial_approximation
            Whether to use the leading terms of the virial approximation to calculate
            the total polarization of the polar species (from Harvey & Lemmon, 2005),
            or to use the polarization relationship by :cite:t:`pitzer1983`.
            These two approaches are numerically fully equivalent.
        use_cimino_clouds
            Whether to use the polarization and absorption equations for the clouds in
            :cite:t:`cimino1982`, eq. (10) and (16), or sections 2.1.5 and 2.2.5
            in :cite:t:`duan2010`.
            See the notes on the importance of this parameter at
            :ref:`implementation:Cloud polarization and absorption`.
        use_cimino_fitted_lookup
            Whether to estimate the complex permittivity of gaseous H2SO4 from
            lookup tables and then pre-fitted analytical extrapolation functions,
            or to numerically inter- and extrapolate.
            This option is only kept for development purposes, since the pre-fitted
            model is flawed. Regardless, this options only has a range delay effect
            on the sub-micrometer scale, and an effect on the two-way attenuation
            on the millidecibel scale.
        load_polarization_parameters
            By default, the polarization parameters are loaded from a prepackaged
            configuration file (in ``"data/default_polarization_parameters.toml"``).
            If set to ``False``, they are recomputed with the current settings.
            If set to a filename, the parameters are loaded from there.
        min_altitude_spacing
            Minimum height spacing between altitude nodes.
        """

        # keep track of interpolators and units
        interpolators = {}
        units = {}

        # part 1: physical quantities

        # temperature and pressure
        altitude, temperature, pressure, mass_density = Duan2010.get_tpd(
            use_compressible_gas=use_compressible_gas,
            use_keating_temp_press_above100km=use_keating_temp_press_above100km,
        )
        atpd = [altitude, temperature, pressure] + (
            [mass_density] if use_compressible_gas else []
        )
        # join the tables as a pandas DataFrame so we can interpolate easier
        physquant_df = astrotable.hstack(atpd).to_pandas(index="altitude")
        # save units
        units |= {qt.info.name: qt.unit for qt in atpd}

        # part 2: compositional profiles

        # get the mixing ratios of the chemical species
        mixratios, comp_interpolators, comp_unit = Duan2010.get_composition(
            use_keating_co_co2_n2_above_100km=use_keating_co_co2_n2_above_100km,
            use_kolste_h2so4=use_kolste_h2so4,
            use_marcq_ocs=use_marcq_ocs,
            add_ar=add_ar,
        )
        # keep track of the chemical species we added
        all_species = list(mixratios.keys())
        # save interpolators and units
        interpolators |= comp_interpolators
        units |= {comp: comp_unit for comp in mixratios.keys()}

        # get the electron density
        el_altitude, el_density, el_interpolator = Duan2010.get_electrons()
        # make electron dataframe for later merging
        el_df = pd.DataFrame(
            index=el_altitude.to("km").value,
            data={el_density.info.name: el_density.value},
        )
        # save interpolators and units
        interpolators |= {el_density.info.name: el_interpolator}
        units |= {el_density.info.name: el_density.unit}

        # get cloud values
        cloud_altitude, cloud_mass_density, cloud_concentration = Duan2010.get_clouds(
            altitude=altitude, temperature=temperature, pressure=pressure
        )
        # make cloud dataframe for later merging
        cloud_df = pd.DataFrame(
            index=cloud_altitude.to("km").value,
            data={
                cloud_mass_density.info.name: cloud_mass_density.value,
                cloud_concentration.info.name: cloud_concentration.value,
            },
        )
        # save units
        units |= {
            cloud_mass_density.info.name: cloud_mass_density.unit,
            cloud_concentration.info.name: cloud_concentration.unit,
        }

        # part 3: combine all physical quantities, mixing ratios, electron density,
        # and cloud profile, ensuring a minimum spacing of altitude values

        # initialize empty DataFrame
        min_alt_spacing_km = min_altitude_spacing.to("km").value
        bigdf = pd.DataFrame(
            index=np.arange(
                -7, 375 + min_alt_spacing_km / 2, min_alt_spacing_km, dtype=float
            )
        )
        bigdf.index.rename("altitude", inplace=True)
        # merge with physical quantities
        bigdf = physquant_df.merge(
            bigdf, how="outer", left_index=True, right_index=True
        )
        # merge with mixing ratios
        for temp in mixratios.values():
            bigdf = bigdf.merge(temp, how="outer", left_index=True, right_index=True)
        # merge with electron density
        bigdf = bigdf.merge(el_df, how="outer", left_index=True, right_index=True)
        # merge with cloud quantities
        bigdf = bigdf.merge(cloud_df, how="outer", left_index=True, right_index=True)

        # part 4: inter- and extrapolating

        # define logarithmic quantities
        intp_quants = list(interpolators.keys())
        log_list = [
            s
            for s in all_species + ["pressure", cloud_mass_density.info.name]
            if s not in intp_quants
        ]
        if use_compressible_gas:
            log_list.append("mass density")
        # define extrapolating behavior
        ffill_list = ["temperature", "CO2", "N2"]
        bfill_list = ["CO2", "N2", "SO2", "CO"]
        if add_ar:
            ffill_list.append("AR")
            bfill_list.append("AR")
        zero_list = all_species + ["electron density"]
        # run interpolator
        bigdf = fill_df(
            bigdf,
            interpolators=interpolators,
            log_list=log_list,
            ffill_list=ffill_list,
            bfill_list=bfill_list,
            zero_list=zero_list,
        )
        # convert back to a QTable
        bigdf.reset_index(names="altitude", inplace=True)
        bigqt = astrotable.QTable.from_pandas(bigdf, units=units)

        # part 5: reformatting for readability

        # split up the table into the different types of values
        # general atmospheric properties
        self.altitude = bigqt[altitude.info.name]
        self.temperature = bigqt[temperature.info.name]
        self.pressure = bigqt[pressure.info.name]
        if use_compressible_gas:
            self.mass_density = bigqt[mass_density.info.name]
        # mixture quantities
        self.electron_density = bigqt[el_density.info.name]
        # component quantities
        self.molar_fractions = bigqt[all_species]
        # cloud quantities
        self.cloud_mass_density = bigqt[cloud_mass_density.info.name]
        self.cloud_concentration = bigqt[cloud_concentration.info.name]

        # part 6: computation of total and per-species densities

        self.update_densities()
        # sets self.mass_density (if not already present), self.number_density,
        # self.molar_density, and self.[molar_densities,mass_densities]

        # part 7: get individual contributions to polarization and absorption
        # for each species and the clouds in the atmosphere, as well as the
        # resulting real part of the relative permittivity

        # the computation of the polarization parameters is independent of
        # the loaded atmospheric profiles

        # check if we should recompute them
        if load_polarization_parameters == False:
            self.polarization_parameters = Duan2010.get_polarization_parameters(
                add_ar=add_ar, use_virial_approximation=use_virial_approximation
            )
        # or load them (either the defaults or from a file)
        else:
            self.polarization_parameters = read_polarization_parameters(
                None
                if load_polarization_parameters == True
                else load_polarization_parameters
            )

        # everything else depends on the current state
        self.update_pol_absorp_atmosphere(
            cutoff_so2_frequency=cutoff_so2_frequency,
            use_kolbe_ocs=use_kolbe_ocs,
            use_cimino_clouds=use_cimino_clouds,
            use_cimino_fitted_lookup=use_cimino_fitted_lookup,
        )
        # sets self.polarization[s], self.absorption[s], and self.eps_prime_r_atmo

        # part 8: ionosphere

        self.update_ionosphere()
        # sets self.eps_prime_r_iono

        # part 9: combine all contributions

        self.update_rel_perm_refraction()
        # sets self.relative_permittivity and self.refraction

        # done

    def update_densities(self):
        """
        Compute the total and specific mass, number, and molar densities
        from the total pressure and temperature, and the molar fractions.
        If the mass density has not been set yet, it is derived from the
        ideal gas law.

        Notes
        -----
        Reads: :attr:`~Model.pressure`, :attr:`~Model.temperature`,
        :attr:`~Model.molar_fractions`

        Writes: :attr:`~Model.number_density`, :attr:`~Model.mass_densities`,
        :attr:`~Model.molar_density`, :attr:`~Model.molar_densities`, and
        (if not already present) :attr:`~Model.mass_density`
        """

        # total densities
        try:
            # number density
            self.number_density = (
                self.mass_density * AVOGADRO / Duan2010.VENUS_MOLAR_MASS
            ).decompose()
        except AttributeError as e:  # try again if mass_density was not found
            # mass density
            self.mass_density = (
                self.pressure / (Duan2010.VENUS_GAS_CONSTANT * self.temperature)
            ).decompose()
            # number density
            self.number_density = (
                self.mass_density * AVOGADRO / Duan2010.VENUS_MOLAR_MASS
            ).decompose()
        # molar density
        self.molar_density = (
            self.pressure / (GAS_CONSTANT * self.temperature)
        ).decompose()

        # per-species densities
        try:
            # initialize
            molar_densities = astrotable.QTable()
            mass_densities = astrotable.QTable()
            # loop over all species
            for c in self.molar_fractions.colnames:
                # molar densities
                molar_densities[c] = (
                    self.molar_fractions[c] * self.molar_density
                ).decompose()
                # mass densities
                mass_densities[c] = (molar_densities[c] * SPEC_MOL_M[c]).decompose()
        except NameError as e:
            # if molar_fractions are missing, then we simply skip
            # the computation, but otherwise something's wrong
            if "object has no attribute 'molar_fractions'" not in str(e):
                raise
        else:
            # save results
            self.molar_densities = molar_densities
            self.mass_densities = mass_densities

        # done

    def update_pol_absorp_atmosphere(
        self,
        cutoff_so2_frequency: Quantity["frequency"] | None = None,
        use_kolbe_ocs: bool = False,
        use_cimino_clouds: bool = True,
        use_cimino_fitted_lookup: bool = False,
    ):
        """
        Update the individual and total polarization and absorption
        of the atmosphere's species and clouds given the polarization
        and absorption parameters. Then, sum up the contributions
        and compute the resulting real part of the relative permittivity.

        Parameters
        ----------
        cutoff_so2_frequency
            When computing the absorption coefficient of SO2, include all spectral
            lines up to this frequency. If ``None``, use all available ones.
            This option is only kept for development purposes.
        use_kolbe_ocs
            Whether to use the :cite:t:`kolbe1977`, Lorentzian-based approach to
            compute the absorption coefficient from OCS, or not.
        use_cimino_clouds
            Whether to use the polarization and absorption equations for the clouds in
            :cite:t:`cimino1982`, eq. (10) and (16), or sections 2.1.5 and 2.2.5
            in :cite:t:`duan2010`.
        use_cimino_fitted_lookup
            Whether to estimate the complex permittivity of gaseous H2SO4 from
            lookup tables and then pre-fitted analytical extrapolation functions,
            or to numerically inter- and extrapolate.
            This option is only kept for development purposes, since the pre-fitted
            model is flawed. Regardless, this options only has a range delay effect
            on the sub-micrometer scale, and an effect on the two-way attenuation
            on the millidecibel scale.

        Notes
        -----
        Reads: :attr:`~Model.polarization_parameters`, :attr:`~Model.temperature`,
        :attr:`~Model.pressure`, :attr:`~Model.molar_fractions`,
        :attr:`~Model.molar_densities`, :attr:`~Model.mass_densities`,
        :attr:`~Model.cloud_concentration`, and :attr:`~Model.cloud_mass_density`.

        Writes: :attr:`~Model.polarizations`, :attr:`~Model.polarization`,
        :attr:`~Model.absorptions`, :attr:`~Model.absorption`, and
        :attr:`~Model.eps_prime_r_atmo`.
        """

        # sections 2.1.3-2.1.4: non-polar and polar components
        # convert polarization parameters to actual polarizations
        self.polarizations = self.evaluate_polarization_parameters()

        # sections 2.2.1-2.2.4: absorptions from species
        self.absorptions = self.evaluate_absorptions(
            cutoff_so2_frequency=cutoff_so2_frequency,
            use_kolbe_ocs=use_kolbe_ocs,
        )

        # sections 2.1.5 and 2.2.5: clouds
        # add quantities to existing QTable
        self.polarizations["cloud"], self.absorptions["cloud"] = (
            self.evaluate_cloud_permittivity(
                use_cimino_clouds=use_cimino_clouds,
                use_cimino_fitted_lookup=use_cimino_fitted_lookup,
            )
        )

        # section 2.1.2: sum of polarizations
        self.polarization = self.sum_polarizations()
        # section 2.2: sum of absorptions
        self.absorption = self.sum_absorptions()

        # section 2.1.1: convert total polarization
        # to real part of relative permittivity
        self.eps_prime_r_atmo = Duan2010.eps_prime_r_from_eq3(self.polarization)

        # done

    def update_ionosphere(self):
        """
        Converts the model's electron density to the corresponding
        real part of the relative permittivity.

        Notes
        -----
        Reads: :attr:`~Model.electron_density`.

        Writes: :attr:`~Model.eps_prime_r_iono`.
        """
        # section 2.1.6: real part of the relative permittivity of the ionosphere
        self.eps_prime_r_iono = Duan2010.eq22_mod(
            self.electron_density, VISAR_FREQUENCY
        )

    def update_rel_perm_refraction(self):
        """
        Update the complex relative permittivity from the real parts of the
        atmos- and ionosphere, as well as the total absorption profile.

        Notes
        -----
        Reads: :attr:`~Model.altitude`, :attr:`~Model.eps_prime_r_atmo`,
        :attr:`~Model.eps_prime_r_iono`, and :attr:`~Model.absorption`.

        Writes: :attr:`~Model.relative_permittivity` and :attr:`~Model.refraction`.
        """

        # combine the computed real parts of the relative permittivity from
        # the atmosphere and the ionosphere, simply switching from one to the other
        i_transition = np.argmax(self.altitude > self.TRANSITION_ATMO_IONO)
        eps_prime_r = np.r_[
            self.eps_prime_r_atmo[:i_transition],
            self.eps_prime_r_iono[i_transition:],
        ]
        assert np.all(np.isfinite(eps_prime_r))

        # convert absorption and real part of the permittivity to imaginary part
        eps_dprime_r = Duan2010.eps_dprime_r_from_eq25(eps_prime_r, self.absorption)

        # get the total complex relative permittivity
        self.relative_permittivity = Quantity(
            eps_prime_r + eps_dprime_r * 1j, u.dimensionless_unscaled
        )

        # convert total complex relative permittivity to index of refraction
        self.refraction = Model.rel_permittivity_to_refraction(
            self.relative_permittivity
        )

        # done

    @staticmethod
    def get_tpd(
        use_compressible_gas: bool = True,
        use_keating_temp_press_above100km: bool = False,
    ) -> Tuple[
        Quantity["length"],
        Quantity["temperature"],
        Quantity["pressure"],
        Quantity["mass density"] | None,
    ]:
        """
        Follow Section  3.1 to build the temperature, pressure, and mass density
        profiles.

        Parameters
        ----------
        use_compressible_gas
            Whether to use the gas compressibility factor when deriving the mass
            density for the 0-100 km altitude range, or assume the ideal gas law.
            Gas compressibility is always assumed below 0 km, and never above 100 km.
        use_keating_temp_press_above100km
            Whether to use the temperature profile from :cite:t:`keating1985`
            above 100 km, and get its matching pressure profile from the
            ideal gas law.

        Returns
        -------
        altitude
            Altitude levels
        temperature
            Temperature profile
        pressure
            Pressure profile
        mass_density
            Mass density profile (only if ``use_compressible_gas=True``)
        """

        # p. 13: "for the lower atmosphere, [...] the temperature curve at
        # latitude of 75° in the work of Seiff et al. (1985) is used after being
        # increased by 3 K"
        ix_seiff_below_zasova = (
            seiff1985.tables["1-2d"]["z"] < zasova2006.tables["5"]["H"][-1]
        )
        alt = [
            seiff1985.tables["1-1"]["z"],
            seiff1985.tables["1-2d"]["z"][ix_seiff_below_zasova],
        ]
        temp = [
            seiff1985.tables["1-1"]["T"] + 3 * u.K,
            seiff1985.tables["1-2d"]["T"][ix_seiff_below_zasova] + 3 * u.K,
        ]
        press = [
            seiff1985.tables["1-1"]["p"],
            seiff1985.tables["1-2d"]["p"][ix_seiff_below_zasova],
        ]
        if use_compressible_gas:
            dens = [
                seiff1985.tables["1-1"]["ρ"],
                seiff1985.tables["1-2d"]["ρ"][ix_seiff_below_zasova],
            ]

        # p. 14: "In the simulation, the middle atmosphere temperature and
        # pressure profiles are using the column of Ls = 200°–270° in
        # Table 5 of Zasova et al. (2006)"
        alt.append(zasova2006.tables["5"]["H"][::-1])
        temp.append(zasova2006.tables["5"]["Ls = 200°-270°, T"][::-1])
        press.append(zasova2006.tables["5"]["Ls = 200°-270°, P"][::-1])
        # interpolate the pressure levels of Zasova et al. (2006) onto compressible
        # density profile from Seiff et al. (1985)
        if use_compressible_gas:
            dens.append(
                Quantity(
                    np.interp(
                        press[-1].to("bar").value,
                        seiff1985.tables["1-2d"]["p"].to("bar").value[::-1],
                        seiff1985.tables["1-2d"]["ρ"].value[::-1],
                    ),
                    seiff1985.tables["1-2d"]["ρ"].unit,
                )
            )
        # extrapolate to negative altitudes
        alt_neg, temp_neg, press_neg, dens_neg = Model.tpd_below_0km(
            Duan2010.VENUS_GAS_CONSTANT
        )
        # insert into list
        alt.insert(0, alt_neg)
        temp.insert(0, temp_neg)
        press.insert(0, press_neg)
        if use_compressible_gas:
            dens.insert(0, dens_neg)

        # for altitudes higher than 100 km, we can use the VIRA model from
        # Keating et al. (1985) directly from 105 km upwards
        # using the night side to be consistent with the Zasova data below 100 km
        # for 150 km and above
        if use_keating_temp_press_above100km:
            alt.append(keating1985.tables["night"]["ALT"][1:])
            temp.append(keating1985.tables["night"]["T"][1:])
            press.append(keating1985.tables["night"]["P"][1:])
            if use_compressible_gas:
                dens.append(keating1985.tables["night"]["RHO"][1:])
        # otherwise, we use a previously-fitted extrapolating function for pressure,
        # continue the temperature as a constant, and use the ideal gas law to get
        # mass density
        else:
            extrap_alt_km = np.arange(101, 376)
            alt.append(
                Quantity(extrap_alt_km, "km").to(
                    keating1985.tables["night"]["ALT"].unit
                )
            )
            temp.append(
                Quantity(
                    np.full(
                        extrap_alt_km.size,
                        zasova2006.tables["5"]["Ls = 200°-270°, T"][0].value,
                    ),
                    zasova2006.tables["5"]["Ls = 200°-270°, T"].unit,
                )
            )
            press.append(
                Quantity(
                    10
                    ** (
                        Duan2010.EXT_PRESSURE_COEFFS[0]
                        * np.exp(Duan2010.EXT_PRESSURE_COEFFS[1] * extrap_alt_km)
                        + Duan2010.EXT_PRESSURE_COEFFS[2]
                        * np.exp(Duan2010.EXT_PRESSURE_COEFFS[3] * extrap_alt_km)
                    ),
                    "atm",
                ).to(keating1985.tables["night"]["P"].unit)
            )
            if use_compressible_gas:
                dens.append(press[-1] / (Duan2010.VENUS_GAS_CONSTANT * temp[-1]))

        # combine
        altitude = np.concatenate(alt)
        temperature = np.concatenate(temp)
        pressure = np.concatenate(press)

        # set table names for easier joining
        altitude.info.name = "altitude"
        temperature.info.name = "temperature"
        pressure.info.name = "pressure"

        # repeat for mass density if we extracted it
        if use_compressible_gas:
            mass_density = np.concatenate(dens)
            mass_density.info.name = "mass density"
        else:
            mass_density = None

        # done
        return altitude, temperature, pressure, mass_density

    @staticmethod
    def get_composition(
        use_keating_co_co2_n2_above_100km: bool = False,
        use_kolste_h2so4: bool = False,
        use_marcq_ocs: bool = False,
        add_ar: bool = False,
    ) -> Tuple[dict[str, pd.DataFrame], dict[str, Callable], Unit]:
        """
        Load the compositions for the different chemical species.

        Parameters
        ----------
        use_keating_co_co2_n2_above_100km
            Whether to use the :cite:t:`keating1985` mixing ratios for
            CO, CO2, and N2 as continuation above the :cite:t:`duan2010`
            profiles (instead of continuing CO2 and N2 upwards as a constant,
            and setting CO to zero upwards of the highest value).
        use_kolste_h2so4
            Whether to use the H2SO4 profile from :cite:t:`kolodner1998`
            instead of the :cite:t:`duan2010` profile.
        use_marcq_ocs
            Whether to use the OCS profile from :cite:t:`marcq2006`
            instead of the :cite:t:`duan2010` profile.
        add_ar
            Whether to add a constant value for Argon into the mixture.

        Returns
        -------
        mixratios
            Dictionary of all mixing ratios as :class:`~pandas.DataFrame`
        interpolators
            Dictionary of matching generating functions for all species
        comp_unit
            Unit of all mixing ratios
        """

        # initialize
        mixratios = {}
        interpolators = {}
        comp_unit = Unit("ppm")

        # optional, load data from above 100km
        if use_keating_co_co2_n2_above_100km:
            # the components from Keating et al. (1985),
            # subset to what is actually used
            highcomps_list = ["CO", "CO2", "N2"]
            highcomps_df = keating1985.tables["night"][
                ["ALT"] + highcomps_list + ["NTOT"]
            ].to_pandas()
            # now we can convert from number density to ppm
            highcomps_df.loc[:, highcomps_list] *= (
                1e6 / highcomps_df["NTOT"].to_numpy()[:, None]
            )
            # set altitude as the index
            highcomps_df.set_index("ALT", inplace=True)

        # CO2
        if use_keating_co_co2_n2_above_100km:
            mixratios["CO2"] = highcomps_df["CO2"]
            # set constant value at 100 km to the standard one
            mixratios["CO2"].iloc[0] = Duan2010.VENUS_STANDARD_CO2.to(comp_unit).value
        else:
            mixratios["CO2"] = pd.DataFrame(
                index=[100.0],
                data={"CO2": Duan2010.VENUS_STANDARD_CO2.to(comp_unit).value},
            )

        # N2
        if use_keating_co_co2_n2_above_100km:
            mixratios["N2"] = highcomps_df["N2"]
            # set constant value at 100 km to the standard one
            mixratios["N2"].iloc[0] = Duan2010.VENUS_STANDARD_N2.to(comp_unit).value
        else:
            mixratios["N2"] = pd.DataFrame(
                index=[100.0],
                data={"N2": Duan2010.VENUS_STANDARD_N2.to(comp_unit).value},
            )

        # AR
        if add_ar:
            mixratios["AR"] = pd.DataFrame(
                index=[100.0],
                data={"AR": Duan2010.MR_AR.to(comp_unit).value},
            )

        # H2O
        # no options to check here
        mixratios["H2O"] = pd.DataFrame(
            index=duan2010figures.H2O_FRACTION_NODES[:, 0],
            data={
                "H2O": duan2010figures.get_h2o_density(
                    duan2010figures.H2O_FRACTION_NODES[:, 0]
                )
                .to(comp_unit)
                .value
            },
        )
        interpolators["H2O"] = duan2010figures.get_h2o_density

        # SO2
        # no options to check here
        mixratios["SO2"] = pd.DataFrame(
            index=duan2010figures.SO2_FRACTION_NODES[:, 0],
            data={
                "SO2": duan2010figures.get_so2_density(
                    duan2010figures.SO2_FRACTION_NODES[:, 0]
                )
                .to(comp_unit)
                .value
            },
        )
        interpolators["SO2"] = duan2010figures.get_so2_density

        # H2SO4
        if use_kolste_h2so4:
            mixratios["H2SO4"] = kolodnersteffes1998.tables["H2SO4 X-band"].to_pandas(
                index="altitude"
            )
            mixratios["H2SO4"].rename(
                columns={mixratios["H2SO4"].columns[0]: "H2SO4"}, inplace=True
            )
        else:
            mixratios["H2SO4"] = pd.DataFrame(
                index=duan2010figures.H2SO4_FRACTION_NODES[:, 0],
                data={
                    "H2SO4": duan2010figures.get_h2so4_density(
                        duan2010figures.H2SO4_FRACTION_NODES[:, 0]
                    )
                    .to(comp_unit)
                    .value
                },
            )
            interpolators["H2SO4"] = duan2010figures.get_h2so4_density

        # CO
        co_alt = duan2010figures.CO_FRACTION_NODES[:, 0]
        if use_keating_co_co2_n2_above_100km:
            co_alt = co_alt[co_alt < 100]
            co_alt = pd.DataFrame(
                index=co_alt,
                data={"CO": duan2010figures.get_co_density(co_alt).to(comp_unit).value},
            )
            temp = pd.concat([co_alt, highcomps_df["CO"]], axis=0)
            mixratios["CO"] = temp
        else:
            mixratios["CO"] = pd.DataFrame(
                index=co_alt,
                data={"CO": duan2010figures.get_co_density(co_alt).to(comp_unit).value},
            )
            interpolators["CO"] = duan2010figures.get_co_density

        # OCS
        # NOTE: only the imaginary contribution is considered currently
        if use_marcq_ocs:
            mixratios["OCS"] = marcq2006.tables["fig8"].to_pandas(index="altitude")
            mixratios["OCS"].rename(
                columns={mixratios["OCS"].columns[0]: "OCS"}, inplace=True
            )
            mixratios["OCS"] *= 1e6
        else:
            mixratios["OCS"] = pd.DataFrame(
                index=duan2010figures.OCS_FRACTION_NODES[:, 0],
                data={
                    "OCS": duan2010figures.get_ocs_density(
                        duan2010figures.OCS_FRACTION_NODES[:, 0]
                    )
                    .to(comp_unit)
                    .value
                },
            )
            interpolators["OCS"] = duan2010figures.get_ocs_density

        # done
        return mixratios, interpolators, comp_unit

    @staticmethod
    def get_electrons() -> (
        Tuple[Quantity["length"], Quantity["number density"], Callable]
    ):
        """
        Return the default electron density profile as given by Fig. 6a (blue line).

        Returns
        -------
        el_altitude
            Altitude levels at which the electron density is defined
        el_density
            Electron number density
        el_interpolator
            Interpolating function for the electron number density
        """
        # default altitudes
        el_altitude = Quantity(duan2010figures.ELECTRON_DENSITY_NODES[:, 0], "km")
        # generating (interpolating) function
        el_interpolator = duan2010figures.get_electron_density
        # call function
        el_density = el_interpolator(duan2010figures.ELECTRON_DENSITY_NODES[:, 0])
        # set table names for easier joining
        el_density.info.name = "altitude"
        el_density.info.name = "electron density"
        # done
        return el_altitude, el_density, el_interpolator

    @staticmethod
    def get_clouds(
        altitude: Quantity["length"],
        temperature: Quantity["temperature"],
        pressure: Quantity["pressure"],
    ) -> Tuple[Quantity["length"], Quantity["mass density"], Quantity["%"]]:
        """
        Compute the cloud mass density and concentration following Section 2.1.5.

        Parameters
        ----------
        altitude
            Altitude levels
        temperature
            Temperature profile
        pressure
            Pressure profile

        Returns
        -------
        cloud_altitude
            Altitude levels at which the cloud profile is defined
        cloud_mass_density
            Cloud mass density profile
        cloud_concentration
            Cloud concentration profile
        """

        # get cloud altitudes from reference
        cloud_altitude = james1997.tables["clouds"]["altitude"]
        # extract pressure and temperature from the main profile
        # onto these altitudes
        cloud_pressure = Quantity(
            np.interp(
                cloud_altitude.to("km").value,
                altitude.to("km").value,
                pressure.to("bar").value,
                left=np.nan,
                right=np.nan,
            ),
            "bar",
        )
        cloud_temperature = Quantity(
            np.interp(
                cloud_altitude.to("km").value,
                altitude.to("km").value,
                temperature.to("K").value,
                left=np.nan,
                right=np.nan,
            ),
            "K",
        )
        # calculate mass density of the distributed solution
        total_cloud_mass_density = cloud_pressure / (
            Duan2010.VENUS_GAS_CONSTANT * cloud_temperature
        )
        cloud_mass_density = (
            james1997.tables["clouds"]["mass mixing ratio clouds"]
            * total_cloud_mass_density
        ).decompose()
        # set to NaN where it's zero to avoid some division-by-zero later
        cloud_mass_density[cloud_mass_density == 0] = np.nan
        # next, we translate the concentration profile from James et al. (1997)
        # to densities using Duan et al. (2010), Table 4
        cloud_concentration = Quantity(
            np.interp(
                cloud_altitude.to("km").value,
                james1997.tables["fig7"]["Altitude"].to("km").value,
                james1997.tables["fig7"]["Weight Percent"].to("%").value,
                left=np.nan,
                right=np.nan,
            ),
            "%",
        )

        # set table names for easier joining
        cloud_altitude.info.name = "altitude"
        cloud_mass_density.info.name = "cloud mass density"
        cloud_concentration.info.name = "cloud concentration"

        # done
        return cloud_altitude, cloud_mass_density, cloud_concentration

    @staticmethod
    def get_polarization_parameters(
        add_ar: bool = False,
        use_virial_approximation: bool = True,
    ) -> dict[str, HarveyLemmon2005Parameters | Pitzer1983Parameters]:
        """
        Get the polarization parameters of the different species.
        Follows Section 2.1.

        Parameters
        ----------
        add_ar
            Whether to add a constant value for Argon into the mixture.
        use_virial_approximation
            Whether to use the leading terms of the virial approximation to calculate
            the total polarization of the polar species (from Harvey & Lemmon, 2005),
            or to use the polarization relationship by :cite:t:`pitzer1983`.

        Returns
        -------
            Dictionary that containes the parameter objects for each species
        """

        # initialize
        polarization_parameters = {}

        # section 2.1.3: non-polar components
        # here, we have the Harvey Lemmon parameters already

        # CO2
        polarization_parameters["CO2"] = Duan2010.HLP_CO2
        # N2
        polarization_parameters["N2"] = Duan2010.HLP_N2
        # AR, if we have it
        if add_ar:
            polarization_parameters["AR"] = Duan2010.HLP_AR

        # section 2.1.4: polar components

        # section 2.1.4.1: H2O (water vapour)
        # here, we only have a Pitzer parameter set
        polarization_parameters["H2O"] = Duan2010.PP_water_vapor

        # section 2.1.4.2: SO2
        # get real part of the permittivity from integrating through
        # the spectral lines
        eps_prime_r_so2 = Duan2010.eps_prime_r_from_spectral_lines(
            Duan2010.T_SO2,
            Duan2010.P_SO2,
            jplspectrallines.tables["SO2"],
            Duan2010.BR_SO2_CO2,
            Duan2010.EPS_PRIME_R_INF_SO2,
            VISAR_FREQUENCY,
        )
        # convert to polarization
        Pnu_SO2 = Duan2010.eq2(eps_prime_r_so2)
        if use_virial_approximation:
            # get virial expansion terms
            A_mu_SO2 = float(HarveyLemmon2005Parameters.get_A_mu(Duan2010.MU_SO2))
            A_epsilon_SO2 = float(
                Duan2010.A_epsilon_from_eq8(
                    Pnu_SO2, A_mu_SO2, Duan2010.RHO_SO2, Duan2010.T_SO2
                )
            )
            # define Harvey & Lemmon parameter set
            polarization_parameters["SO2"] = HarveyLemmon2005Parameters(
                a0=A_epsilon_SO2, A_mu=A_mu_SO2
            )
        else:
            # get molecular polarizability
            alpha_T_SO2 = Duan2010.alpha_T_from_eq14(
                Duan2010.RHO_SO2,
                Duan2010.T_SO2,
                Pnu_SO2,
                Duan2010.MU_SO2,
            )
            # define Pitzer parameter set
            polarization_parameters["SO2"] = Pitzer1983Parameters(
                Duan2010.MU_SO2,
                alpha_T_SO2,
            )

        # section 2.1.4.3: H2SO4 (gaseous)
        # get Pnu from the experiment of Kolodner and Steffes (1998)
        eps_prime_r_h2so4, rho_h2so4 = (
            kolodnersteffes1998.get_eps_prime_r_and_molar_density()
        )
        Pnu_H2SO4 = Duan2010.eq3(eps_prime_r_h2so4)
        if use_virial_approximation:
            # get virial expansion terms
            A_mu_H2SO4 = float(
                HarveyLemmon2005Parameters.get_A_mu(kolodnersteffes1998.MU_H2SO4)
            )
            A_epsilon_H2SO4 = float(
                Duan2010.A_epsilon_from_eq8(
                    Pnu_H2SO4, A_mu_H2SO4, rho_h2so4, kolodnersteffes1998.T_H2SO4
                )
            )
            # define Harvey & Lemmon parameter set
            polarization_parameters["H2SO4"] = HarveyLemmon2005Parameters(
                a0=A_epsilon_H2SO4, A_mu=A_mu_H2SO4
            )
        else:
            # get molecular polarizability
            alpha_T_H2SO4 = Duan2010.alpha_T_from_eq14(
                rho_h2so4,
                kolodnersteffes1998.T_H2SO4,
                Pnu_H2SO4,
                kolodnersteffes1998.MU_H2SO4,
            )
            # define Pitzer parameter set
            polarization_parameters["H2SO4"] = Pitzer1983Parameters(
                kolodnersteffes1998.MU_H2SO4,
                alpha_T_H2SO4,
            )

        # section 2.1.4.4: CO
        Pnu_CO = Duan2010.eq3(Duan2010.EPS_PRIME_R_CO)
        if use_virial_approximation:
            # get virial expansion terms
            A_mu_CO = float(HarveyLemmon2005Parameters.get_A_mu(Duan2010.MU_CO))
            A_epsilon_CO = float(
                Duan2010.A_epsilon_from_eq8(
                    Pnu_CO, A_mu_CO, Duan2010.RHO_CO, Duan2010.T_CO
                )
            )
            # define Harvey & Lemmon parameter set
            polarization_parameters["CO"] = HarveyLemmon2005Parameters(
                a0=A_epsilon_CO, A_mu=A_mu_CO
            )
        else:
            # get molecular polarizability
            alpha_T_CO = Duan2010.alpha_T_from_eq14(
                Duan2010.RHO_CO,
                Duan2010.T_CO,
                Pnu_CO,
                Duan2010.MU_CO,
            )
            # define Pitzer parameter set
            polarization_parameters["CO"] = Pitzer1983Parameters(
                Duan2010.MU_CO,
                alpha_T_CO,
            )

        # section 2.1.4.5: OCS
        # get real part of the permittivity from integrating through
        # the spectral lines
        eps_prime_r_ocs = Duan2010.eps_prime_r_from_spectral_lines(
            Duan2010.T_OCS,
            Duan2010.P_OCS,
            jplspectrallines.tables["OCS"],
            Duan2010.BR_OCS_CO2,
            Duan2010.EPS_PRIME_R_INF_OCS,
            VISAR_FREQUENCY,
        )
        # convert to polarization
        Pnu_OCS = Duan2010.eq2(eps_prime_r_ocs)
        if use_virial_approximation:
            # get virial expansion terms
            A_mu_OCS = float(HarveyLemmon2005Parameters.get_A_mu(Duan2010.MU_OCS))
            A_epsilon_OCS = float(
                Duan2010.A_epsilon_from_eq8(
                    Pnu_OCS, A_mu_OCS, Duan2010.RHO_OCS, Duan2010.T_OCS
                )
            )
            # define Harvey & Lemmon parameter set
            polarization_parameters["OCS"] = HarveyLemmon2005Parameters(
                a0=A_epsilon_OCS, A_mu=A_mu_OCS
            )
        else:
            # get molecular polarizability
            alpha_T_OCS = Duan2010.alpha_T_from_eq14(
                Duan2010.RHO_OCS,
                Duan2010.T_OCS,
                Pnu_OCS,
                Duan2010.MU_OCS,
            )
            # define Pitzer parameter set
            polarization_parameters["OCS"] = Pitzer1983Parameters(
                Duan2010.MU_OCS,
                alpha_T_OCS,
            )

        # done
        return polarization_parameters

    def evaluate_polarization_parameters(self) -> astrotable.QTable:
        """
        Evaluate the model's polarization parameters given its atmospheric quantities.

        Returns
        -------
            Table with evaluated polarizations
        """
        # initialize
        polarizations = astrotable.QTable()
        # loop over species
        for comp in self.molar_densities.keys():
            params = self.polarization_parameters[comp]
            if isinstance(params, HarveyLemmon2005Parameters):
                polarizations[comp] = Duan2010.eq8(
                    self.molar_densities[comp], self.temperature, params
                )
            elif isinstance(params, Pitzer1983Parameters):
                polarizations[comp] = Duan2010.eq14(
                    self.molar_densities[comp], self.temperature, params
                )
            else:
                raise NotImplementedError(
                    "There is no known converting function"
                    f"from {type(params)} parameters to polarization"
                )
        # done
        return polarizations

    def evaluate_cloud_permittivity(
        self, use_cimino_clouds: bool = True, use_cimino_fitted_lookup: bool = False
    ) -> Tuple[Quantity["dimensionless"], Quantity["wavenumber"]]:
        """
        Evaluate the cloud polarization and absorption given the model's
        atmospheric quantities. Follows Sections 2.1.5 and 2.2.5,
        and/or :cite:t:`cimino1982`.

        Parameters
        ----------
        use_cimino_clouds
            Whether to use the polarization and absorption equations for the clouds in
            :cite:t:`cimino1982`, eq. (10) and (16), or sections 2.1.5 and 2.2.5
            in :cite:t:`duan2010`.
        use_cimino_fitted_lookup
            Whether to estimate the complex permittivity of gaseous H2SO4 from
            lookup tables and then pre-fitted analytical extrapolation functions,
            or to numerically inter- and extrapolate.
            This option is only kept for development purposes, since the pre-fitted
            model is flawed. Regardless, this options only has a range delay effect
            on the sub-micrometer scale, and an effect on the two-way attenuation
            on the millidecibel scale.

        Returns
        -------
        cloud_pol
            Polarization of the cloud (accounting for its volume fraction)
        cloud_absorp
            Absorption of the cloud (accounting for its volume fraction)
        """
        # get the complex relative permittivity of the clouds
        if use_cimino_fitted_lookup:
            eps_prime_r_H2SO4_H2O, eps_dprime_r_H2SO4_H2O = (
                Duan2010.get_h2so4_rel_permittivity(
                    self.cloud_concentration,
                    self.temperature,
                    VISAR_FREQUENCY,
                )
            )
        else:
            eps_prime_r_H2SO4_H2O, eps_dprime_r_H2SO4_H2O = (
                cimino1982.get_h2s04_rel_permittivity(
                    self.cloud_concentration,
                    self.temperature,
                    wavelength=VISAR_WAVELENGTH,
                )
            )
        # convert the relative permittivity to polarization
        if use_cimino_clouds:
            # use the equations in the Cimino paper
            # calculate mass of shell and core
            d_shell = self.cloud_mass_density / (1 + cimino1982.RATIO_MASS_DROPLETS)
            d_core = cimino1982.RATIO_MASS_DROPLETS * d_shell
            vol_frac_droplets = (
                d_core / cimino1982.D_DROPLET_CORE
                + d_shell / cimino1982.D_DROPLET_SHELL
            ).decompose()
            # compute complex polarization of droplets using eq. (10)
            cloud_Pnu = cimino1982.get_h2so4_droplet_polarization(
                eps_prime_r_H2SO4_H2O - 1j * eps_dprime_r_H2SO4_H2O
            )
            # save
            cloud_pol = cloud_Pnu.real * vol_frac_droplets
        else:
            # follow section 2.1.5
            # we look up (i.e., interpolate) to get the density
            # of the concentrated droplets
            d_concentr_H2SO4 = Quantity(
                np.interp(
                    self.cloud_concentration,
                    duan2010figures.tables["4"]["Weight Percentage"].to("%").value,
                    duan2010figures.tables["4"]["Density"].value,
                    left=np.nan,
                    right=np.nan,
                ),
                duan2010figures.tables["4"]["Density"].unit,
            )
            # calculate the spreading ratio
            eta_s = (d_concentr_H2SO4 / self.cloud_mass_density).decompose()
            # convert it to polarization
            P_concentr_H2SO4_H2O = Duan2010.eq3(eps_prime_r_H2SO4_H2O)
            # and finally calculate the polarization of the distributed solution
            P_distr_H2SO4_H2O = P_concentr_H2SO4_H2O / eta_s
            # since eta_s is the inverse of the volume fraction, the computed
            # polarization already accounts for its density in the atmosphere
            cloud_pol = P_distr_H2SO4_H2O
        # convert the relative permittivity to absorption
        if use_cimino_clouds:
            if use_cimino_fitted_lookup:
                # approximation used by Duan et al. paper
                cloud_Pnu_imag = np.abs(cloud_Pnu.imag.value)
            else:
                # we can use the actual definition from Cimino
                cloud_Pnu_imag = -cloud_Pnu.imag.value
            # eq. (16)
            cloud_absorp = Quantity(
                0.6
                * np.pi
                * cloud_Pnu_imag
                * vol_frac_droplets.to("cm3/m3").value
                / VISAR_WAVELENGTH.to("cm").value,
                "1/km",
            ).decompose()
        else:
            # follow section 2.2.5
            cloud_absorp = (
                Duan2010.eq25(eps_prime_r_H2SO4_H2O, eps_dprime_r_H2SO4_H2O) / eta_s
            )
        # done
        return cloud_pol.unmasked, cloud_absorp.unmasked

    def evaluate_absorptions(
        self,
        cutoff_so2_frequency: Quantity["frequency"] | None = None,
        use_kolbe_ocs: bool = False,
    ) -> astrotable.QTable:
        """
        Evaluate the absorption models given the model's atmospheric quantities.

        Parameters
        ----------
        cutoff_so2_frequency
            When computing the absorption coefficient of SO2, include all spectral
            lines up to this frequency. If ``None``, use all available ones.
            This option is only kept for development purposes.
        use_kolbe_ocs
            Whether to use the :cite:t:`kolbe1977`, Lorentzian-based approach to
            compute the absorption coefficient from OCS, or not.

        Returns
        -------
            Table with evaluated absorptions
        """

        # initialize
        absorptions = astrotable.QTable()

        # section 2.2.1: CO2, N2, Ar, and H2O
        absorptions["CO2+N2+AR+H2O"] = Duan2010.eq26(
            self.pressure,
            self.temperature,
            self.molar_fractions["CO2"],
            self.molar_fractions["N2"],
            (
                self.molar_fractions["AR"]
                if "AR" in self.molar_fractions.colnames
                else None
            ),
            self.molar_fractions["H2O"],
        )

        # section 2.2.2: SO2
        spectral_lines_so2 = (
            jplspectrallines.tables["SO2"]
            if cutoff_so2_frequency is None
            else jplspectrallines.tables["SO2"][
                jplspectrallines.tables["SO2"]["FREQ"] < cutoff_so2_frequency
            ]
        )
        absorptions["SO2"] = Duan2010.eq27(
            self.temperature,
            self.molar_fractions["SO2"] * self.pressure,
            self.molar_fractions["CO2"] * self.pressure,
            spectral_lines_so2,
            VISAR_FREQUENCY,
            Duan2010.BR_SO2_CO2,
        ).squeeze()

        # section 2.2.3: H2SO4 (gaseous)
        absorptions["H2SO4"] = Duan2010.eq33(
            self.molar_fractions["H2SO4"],
            self.pressure,
            VISAR_FREQUENCY,
            self.temperature,
        )

        # section 2.2.4: OCS
        if use_kolbe_ocs:
            absorptions["OCS"] = Duan2010.kolbe_ocs(
                self.temperature,
                self.molar_fractions["OCS"] * self.pressure,
                jplspectrallines.tables["OCS"],
            )
        else:
            absorptions["OCS"] = Duan2010.eq27(
                self.temperature,
                self.molar_fractions["OCS"] * self.pressure,
                self.molar_fractions["CO2"] * self.pressure,
                jplspectrallines.tables["OCS"],
                VISAR_FREQUENCY,
                Duan2010.BR_OCS_CO2,
            ).squeeze()

        # done
        return absorptions

    def sum_polarizations(self) -> Quantity["dimensionless"]:
        """
        Sum the polarizations already present in the model.
        These have all already been scaled by their volume fraction.

        Returns
        -------
            Total polarization of the atmospheric profile
        """
        # convert to same scale and stack
        polarizations = np.stack(
            [
                self.polarizations[c].to(u.dimensionless_unscaled)
                for c in self.polarizations.colnames
            ],
            axis=-1,
        )
        # replace (only) NaNs
        polarizations = np.nan_to_num(
            polarizations, nan=0, posinf=np.inf, neginf=-np.inf
        )
        # sum and give unit
        polarization = Quantity(polarizations.sum(axis=-1), u.dimensionless_unscaled)
        # done
        return polarization

    def sum_absorptions(self) -> Quantity["dimensionless"]:
        """
        Sum the absorptions already present in the model.

        Returns
        -------
            Total absorption of the atmospheric profile
        """
        # convert to same units and stack
        absorptions = np.stack(
            [self.absorptions[c].to("1/cm") for c in self.absorptions.colnames], axis=-1
        )
        # replace (only) NaNs
        absorptions = np.nan_to_num(absorptions, nan=0, posinf=np.inf, neginf=-np.inf)
        # sum and give unit
        absorption = Quantity(absorptions.sum(axis=-1), "1/cm")
        # done
        return absorption

    @staticmethod
    def eq2(eps_prime_r: float_or_array | Quantity) -> float_or_array | Quantity:
        """
        Calculate the polarization per molar volume of a non-polar material
        from the relative dielectric constant using eq. (2) on page 3.

        Parameters
        ----------
        eps_prime_r
            Relative dielectric constant [-]

        Returns
        -------
            Polarization per molar volume [-]
        """
        return (eps_prime_r - 1) / (eps_prime_r + 2)

    @staticmethod
    def eq3(eps_prime_r: float_or_array | Quantity) -> float_or_array | Quantity:
        """
        Calculate the polarization per molar volume of a polar material
        from the relative dielectric constant using eq. (3) on page 3.

        Parameters
        ----------
        eps_prime_r
            Relative dielectric constant [-]

        Returns
        -------
            Polarization per molar volume [-]
        """
        return (eps_prime_r - 1) * (2 * eps_prime_r + 1) / (9 * eps_prime_r)

    @staticmethod
    def eps_prime_r_from_eq3(Pnu: Quantity) -> Quantity:
        """
        Given the polarization per molar volume, calculate the (positive)
        solution of eq. (3) for the dielectric constant.

        Parameters
        ----------
        Pnu
            Polarization per molar volume [-]

        Returns
        -------
            Relative dielectric constant [-]
        """
        return (1 + 9 * Pnu + 3 * np.sqrt(1 + 2 * Pnu + 9 * Pnu**2)).to(
            u.dimensionless_unscaled
        ) / 4

    @staticmethod
    def eq8(
        rho: Quantity,
        T: Quantity,
        fluid: HarveyLemmon2005Parameters,
    ) -> Quantity:
        """
        Calculate the total polarization using the dielectric
        virial expansion as described in eq. (8) from :cite:t:`duan2010`.

        Parameters
        ----------
        rho
            Molar density [mol/m^3]
        T
            Mixture temperature [K]
        fluid
            Material coefficients

        Returns
        -------
            Polarization [-]
        """
        # convert to cm^3 and K units
        # as required by HarveyLemmon2005Parameters
        rho_mol_cm3 = rho.to("mol/cm3").value
        T_K = T.to("K").value
        # calculate terms
        A_eps = fluid.a0 + fluid.a1 * (T_K / fluid.T0 - 1)
        B_eps = fluid.b0 + fluid.b1 * (fluid.T0 / T_K - 1)
        C_eps = fluid.c0 + fluid.c1 * (fluid.T0 / T_K - 1)
        # evaluate and return dimensionless Quantity
        return Quantity(
            (A_eps + fluid.A_mu / T_K) * rho_mol_cm3
            + B_eps * rho_mol_cm3**2
            + C_eps * rho_mol_cm3 ** (fluid.D + 1)
        )

    @staticmethod
    def A_epsilon_from_eq8(
        Pnu: Quantity, A_mu: float_or_array, rho: Quantity, T: Quantity
    ) -> float_or_array:
        """
        Compute the leading non-polar term in the dielectric virial expansion
        (as described by Harvey & Lemmon, 2005, eq. 5) using the polarization
        per molar volume and the dipolar term, and assuming no temperature dependence.

        Parameters
        ----------
        Pnu
            Polarization per molar volume [-]
        A_mu
            Dipolar term in the virial expansion [cm^3 K/mol]
        rho
            Molar density [mol/cm^3]
        T
            Temperature [K]

        Returns
        -------
            Leading non-polar term in the virial expansion [cm^3/mol]
        """
        # compute, assuming A_mu is already in the right units
        A_epsilon = Pnu / rho - Quantity(A_mu, "cm3 K/mol") / T
        # return value in correct units (since HarveyLemmon2005Parameters
        # cannot handle the Quantity type)
        return A_epsilon.to("cm3/mol").value

    @staticmethod
    def kirkwood_correlation_cgs(
        d: float_or_array,
        T: float_or_array,
        p0: float = 2.68,
        p1: float = 6.69,
        p2: float = 565.0,
        e: float = 0.3,
    ) -> float_or_array:
        """
        Kirkwood correlation factor as described on p. 5.
        Inconsistent units so no :class:`~astropy.units.Quantity` inputs.

        Parameters
        ----------
        d
            Mass density [g/cm^3]
        T
            Temperature [K]
        p0, p1, p2, e
            Factors used in the formula

        Returns
        -------
            Kirkwood correlation factor [-]
        """
        return 1 + p0 * d + p1 * d**5 * ((p2 / T) ** e - 1)

    @staticmethod
    def eq14(
        rho: Quantity,
        T: Quantity,
        pp: Pitzer1983Parameters,
        g: float = 1.0,
    ) -> Quantity:
        """
        Calculate the total polarization as described in
        eq. (14), assuming we know the molecular polarizability and
        molecular dipole moment.

        Parameters
        ----------
        rho
            Molar density [mol/m^3]
        T
            Temperature [K]
        pp
            Material polarization parameters
        g
            Kirkwood correlation factor

        Returns
        -------
            Polarization [-]
        """
        # mass_density / molar_mass = molar_density
        first_term = (4 * np.pi * AVOGADRO * rho) / 3
        second_term = pp.alpha_T + (pp.mu**2 * g) / (3 * BOLTZMANN * T)
        Pnu = first_term * second_term
        # return dimensionless Quantity
        return Pnu.decompose()

    @staticmethod
    def alpha_T_from_eq14(
        rho: Quantity,
        T: Quantity,
        Pnu: Quantity,
        mu: Quantity,
        g: float = 1.0,
    ) -> Quantity:
        """
        Calculate the molecular polarizability as described in eq. (14) on
        p. 5, assuming we know the total polarization at given conditions
        and the molecular dipole moment.

        Parameters
        ----------
        rho
            Molar density [mol/m^3]
        T
            Temperature [K]
        Pnu
            Polarization per molar volume [-]
        mu
            Molecular dipole moment [esu cm = 1e18 D]
        g
            Kirkwood correlation factor

        Returns
        -------
            Molecular polarizability [cm^3]
        """
        # molar_mass / mass_density = 1 / molar_density
        first_term = (3 * Pnu) / (4 * np.pi * AVOGADRO * rho)
        second_term = ((mu**2 * g) / (3 * BOLTZMANN * T)).decompose()
        alpha_T = (first_term - second_term).decompose()
        if alpha_T < 0:
            raise ValueError(
                f"Molecular polarizability cannot be negative ({alpha_T=})."
            )
        # return simplified Quantity
        return alpha_T.to("cm3")

    @staticmethod
    def eq22_mod(
        el_density: Quantity, frequency: Quantity = VISAR_FREQUENCY
    ) -> Quantity:
        """
        Calculate the relative permittivity due to the polarization of
        the ionosphere, i.e., the parenthesis in eq. (22).

        Parameters
        ----------
        el_density
            Electron density [1/m^3]
        frequency
            Frequency at which to calculate the permittivity [Hz]

        Returns
        -------
            Relative permittivity [-]
        """
        # plasma frequency from eq. (23)
        omega_p = np.sqrt((el_density * E_CHARGE**2) / (FREE_SPACE_PERM * E_MASS))
        # convert linear to angular frequency
        omega = 2 * np.pi * frequency
        # calculate relative permittivity
        eps_r = 1 - (omega_p / omega) ** 2
        # done
        return eps_r.to(u.dimensionless_unscaled)

    @staticmethod
    def eq25(
        eps_prime_r: float_or_array | Quantity,
        eps_dprime_r: float_or_array | Quantity,
        lambda_0: Quantity = VISAR_WAVELENGTH,
    ) -> Quantity:
        """
        Converts the real and imaginary parts of the relative permittivity
        to the absorption coefficient using eq. (25) on p. 9.

        Parameters
        ----------
        eps_prime_r
            Dielectric constant (real part of the relative permittivity) [-]
        eps_dprime_r
            Imaginary part of the relative permittivity [-]
        lambda_0
            Wavelength [m]

        Returns
        -------
            Power absorption coefficient [1/m]
        """
        alpha = 2 * np.pi * eps_dprime_r / (lambda_0 * np.sqrt(eps_prime_r))
        return alpha.decompose()

    @staticmethod
    def eps_dprime_r_from_eq25(
        eps_prime_r: Quantity,
        alpha: Quantity,
        lambda_0: Quantity = VISAR_WAVELENGTH,
    ) -> Quantity:
        """
        Converts the total absorption and relative dielectric constant
        to the imaginary part of the permittivity using eq. (25) on p. 9.

        Parameters
        ----------
        eps_prime_r
            Dielectric constant (real part of the relative permittivity) [-]
        alpha
            Power absorption coefficient [1/m]
        lambda_0
            Wavelength [m]

        Returns
        -------
            Imaginary part of the permittivity [-]
        """
        eps_dprime_r = alpha * lambda_0 * np.sqrt(eps_prime_r) / (2 * np.pi)
        return eps_dprime_r.to(u.dimensionless_unscaled)

    @staticmethod
    def eq26(
        P: Quantity,
        T: Quantity,
        f_CO2: Quantity,
        f_N2: Quantity,
        f_Ar: Quantity | None,
        f_H2O: Quantity,
        lambda_0: Quantity = VISAR_WAVELENGTH,
    ) -> Quantity:
        """
        Calculate the total absorption of a mixture of CO2, N2, Ar, and H2O
        following eq. (26) on p. 9.

        Parameters
        ----------
        P
            Pressure [bar]
        T
            Temperature [K]
        f_CO2, f_N2, f_Ar, f_H2O
            Molar fractions [-]
        lamda_0
            Wavelength [m]

        Returns
        -------
            Total absorption [1/cm]
        """
        # convert
        f_CO2 = f_CO2.to(u.dimensionless_unscaled).value
        f_N2 = f_N2.to(u.dimensionless_unscaled).value
        f_Ar = (
            np.zeros_like(f_CO2)
            if f_Ar is None
            else f_Ar.to(u.dimensionless_unscaled).value
        )
        f_H2O = f_H2O.to(u.dimensionless_unscaled).value
        # calculate
        alpha = (
            (P.to("atm").value / lambda_0.to("cm").value) ** 2
            * (273.15 / T.to("K").value) ** 5
            * (
                15.7 * f_CO2**2
                + 3.90 * f_CO2 * f_N2
                + 2.64 * f_CO2 * f_Ar
                + 0.085 * f_N2**2
                + 1330 * f_H2O
            )
            * 1e-8
        )
        return Quantity(alpha / 2, "1/cm")

    @staticmethod
    def eq27(
        T: Quantity["temperature"],
        P_minor: Quantity["pressure"],
        P_major: Quantity["pressure"],
        spectral_lines: astrotable.QTable,
        nu: Quantity["frequency"],
        br_params: BenReuvenParameters,
    ) -> Quantity:
        """
        Calculates all absorptions from a spectral line catalog
        and line broadening coefficients as described in
        eqs. (27-32) on pp. 10f.

        Parameters
        ----------
        T
            Temperature [K]
        P_minor
            Partial pressure of the minor species [torr]
        P_major
            Partial pressure of the major species [torr]
        spectral_lines
            Spectral line catalog for the minor species containing line
            frequencies nu [MHz], line center intensities I [nm^2 MHz],
            and lower state energies El [1/cm]
        nu
            Target frequency of the absorption [Hz]
        br_params
            Parameters for the Ben-Reuven line expression

        Returns
        -------
            Total absorption [1/cm]
        """
        # prepare output
        alpha = np.full((T.size, nu.size), np.nan)
        assert len(T) == len(P_minor) == len(P_major)
        data_valid = np.logical_or(P_minor > 0, P_major > 0)
        # convert all input quantities to unit-defined NumPy arrays
        # so we can make effective use of broadcasting
        # first axis: atmospheric parameters
        T = T.to("K").value[data_valid, None, None]
        P_minor = P_minor.to("torr").value[data_valid, None, None]
        P_major = P_major.to("torr").value[data_valid, None, None]
        # second axis: spectral line catalog
        nu_0 = spectral_lines["FREQ"].to("MHz").value[None, :, None]
        I = spectral_lines["LGINT"].physical.to("nm2 MHz").value[None, :, None]
        El = spectral_lines["ELO"].to("1/cm").value[None, :, None]
        # third axis: frequencies to compute results for
        nu = np.atleast_1d(nu.to("MHz").value)[None, None, :]
        # get Ben-Reuven parameters in correct units
        T_0 = br_params.T_0.to("K").value
        gamma_min_maj = br_params.gamma_min_maj.to("MHz/torr").value
        gamma_min_min = br_params.gamma_min_min.to("MHz/torr").value
        zeta_min_maj = br_params.zeta_min_maj.to("MHz/torr").value
        zeta_min_min = br_params.zeta_min_min.to("MHz/torr").value
        delta_min = br_params.delta_min.to("MHz/torr").value
        m = br_params.m
        n = br_params.n
        # eq. (32) [MHz]
        delta = delta_min * P_minor
        # eq. (31) [MHz]
        zeta = (zeta_min_maj * P_major + zeta_min_min * P_minor) * (T_0 / T) ** m
        # eq. (30) [MHz]
        gamma = (gamma_min_maj * P_major + gamma_min_min * P_minor) * (T_0 / T) ** n
        # eq. (29) [1/MHz]
        F_BR = (
            (2 / np.pi)
            * (nu / nu_0) ** 2
            * (
                (gamma - zeta) * nu**2
                + (gamma + zeta) * ((nu_0 + delta) ** 2 + gamma**2 - zeta**2)
            )
            / (
                (nu**2 - (nu_0 + delta) ** 2 - gamma**2 + zeta**2) ** 2
                + 4 * nu**2 * gamma**2
            )
        )
        # eq. (28) [1/cm]
        alpha_max = Quantity(
            102.46
            * P_minor
            / gamma
            * I
            * (T_0 / T) ** (7 / 2)
            * np.exp(
                -(PLANCK * SPEED_OF_LIGHT / BOLTZMANN).to("cm K").value
                * El
                * (1 / T - 1 / T_0)
            ),
            "1/cm",
        )
        # eq. (27) [1/cm]
        alpha[data_valid, :] = np.sum(alpha_max * np.pi * gamma * F_BR, axis=1)
        # done
        return Quantity(alpha, "1/cm")

    @staticmethod
    def kolbe_ocs(
        T: Quantity,
        P_OCS: Quantity,
        spectral_lines_OCS: astrotable.QTable,
        nu: Quantity = VISAR_FREQUENCY,
    ) -> Quantity:
        """
        Calculates all absorptions for the OCS spectral line catalog
        as described in Section 2.2.4, using data from :cite:t:`kolbe1977`
        and a Lorentzian line shape.

        Parameters
        ----------
        T
            Temperature [K]
        P_OCS
            Partial pressure of OCS [torr]
        spectral_lines_OCS
            OCS spectral line catalog containing line frequencies nu [MHz],
            line center intensities I [nm^2 MHz], and lower state energies El [1/cm]
        nu
            Target frequency of the absorption [Hz]

        Returns
        -------
            Total absorption due to OCS [1/cm]
        """
        # prepare output
        alpha = np.full(len(T), np.nan)
        data_valid = P_OCS > 0
        # convert all input quantities to unit-defined NumPy arrays
        # so we can make effective use of broadcasting
        # column: atmospheric parameters
        T = T.to("K").value[data_valid, None]
        P_OCS = P_OCS.to("torr").value[data_valid, None]
        nu = nu.to("MHz").value
        # SO2 parameters
        T0 = 300  # [K]
        gamma_OCS_OCS = 6.4  # [MHz/torr]
        # read spectral line catalog
        nu_0 = spectral_lines_OCS["FREQ"].to("MHz").value[None, :]
        I = spectral_lines_OCS["LGINT"].physical.to("nm2 MHz").value[None, :]
        El = spectral_lines_OCS["ELO"].to("1/cm").value[None, :]
        # get line widths at frequencies
        gamma = gamma_OCS_OCS * P_OCS
        # Lorentzian line shape function [1/MHz]
        F_L = gamma / (np.pi * ((nu_0 - nu) ** 2 + gamma**2))
        # eq. (28) [1/cm]
        alpha_max = Quantity(
            102.46
            * P_OCS
            / gamma
            * I
            * (T0 / T) ** (7 / 2)
            * np.exp(
                -(PLANCK * SPEED_OF_LIGHT / BOLTZMANN).to("cm K").value
                * El
                * (1 / T - 1 / T0)
            ),
            "1/cm",
        )
        # eq. (27) [1/cm]
        alpha[data_valid] = np.sum(alpha_max * np.pi * gamma * F_L, axis=1)
        # done
        return Quantity(alpha, "1/cm")

    @staticmethod
    def eq33(
        q: float_or_array | Quantity, p: Quantity, f: Quantity, T: Quantity
    ) -> Quantity:
        """
        Calculate the total absorption of H2SO4 given eq. (33) on p. 11.

        Parameters
        ----------
        q
            Number mixing ratio of H2SO4 in the H2SO4/CO2 mixture [-]
        p
            Pressure [bar]
        f
            Frequency [Hz]
        T
            Temperature [K]

        Returns
        -------
            Total absorption due to H2SO4 [1/m]
        """
        # eq. (33) [db/km]
        alpha = Quantity(
            53.601
            * p.to("atm").value ** 1.11
            * f.to("GHz").value ** 1.15
            * (553 / T.to("K").value) ** 3
            * q.to(u.dimensionless_unscaled).value,
            "dB / km",
        )
        # return in physical quantities
        return (alpha / 2).to("1/m")

    @staticmethod
    def get_h2so4_rel_permittivity(
        concentration: Quantity, temperature: Quantity, frequency: Quantity
    ) -> Tuple[Quantity, Quantity]:
        """
        Calculate the complex relative permittivity of gaseous H2SO4.

        Parameters
        ----------
        concentration
            Concentration of H2SO4 [%]
        temperature
            Temperature of the medium [K]
        frequency
            Wavelength for which to compute the permittivity values [Hz]

        Returns
        -------
        eps_prime_r
            Real part of the relative permittivity
        eps_dprime_r
            Imaginary part of the relative permittivity

        Note
        ----
        Here, the imaginary part of the relative atmospheric permittivity
        has the opposite sign as in :cite:t:`duan2010`.
        """
        # input size check
        assert concentration.size == temperature.size
        assert frequency.size in [1, concentration.size]
        # get indices of valid data
        try:
            data_valid = ~concentration.value.mask
        except AttributeError:
            data_valid = np.isfinite(concentration.value)
        conc_idx = np.round(concentration[data_valid]).astype(int)
        # look up values
        eps_r_prime_2650MHz = np.full(data_valid.shape, np.nan)
        eps_r_dprime_2650MHz = np.full(data_valid.shape, np.nan)
        eps_r_prime_2650MHz[data_valid] = Duan2010.EPS_PRIME_R_H2SO4[conc_idx]
        eps_r_dprime_2650MHz[data_valid] = Duan2010.EPS_DPRIME_R_H2SO4[conc_idx]
        # extrapolate in frequency and temperature
        freq_GHz = frequency.to("GHz").value
        eps_prime_r = -((freq_GHz - 2.65) ** 0.72) + eps_r_prime_2650MHz
        eps_dprime_r = np.zeros_like(eps_prime_r)
        mask = concentration > Quantity(95, "%")
        eps_dprime_r[mask] = 0.85 * (temperature[mask].to("K").value - 295) + 25
        eps_dprime_r[~mask] = 1.17 * (temperature[~mask].to("K").value - 295) + 26
        # done
        return Quantity(eps_prime_r, u.dimensionless_unscaled), Quantity(
            eps_dprime_r, u.dimensionless_unscaled
        )

    @staticmethod
    def eps_prime_r_from_spectral_lines(
        T: Quantity["temperature"],
        P: Quantity["pressure"],
        spectral_lines: astrotable.QTable,
        br_params: BenReuvenParameters,
        eps_prime_r_inf: Quantity["dimensionless"],
        nu: Quantity["frequency"],
        freqmin: Quantity["frequency"] = Quantity(0.1, "MHz"),
        freqmax: Quantity["frequency"] = Quantity(4, "THz"),
        freqstep: Quantity["frequency"] = Quantity(0.1, "GHz"),
    ):
        """
        Computes the real part of the relative permittivity by integrating
        through the spectral lines and assuming an infinite convergence value

        Parameters
        ----------
        T
            Temperature [K]
        P
            Pressure [bar]
        spectral_lines
            Spectral line catalog for the minor species containing line
            frequencies nu [MHz], line center intensities I [nm^2 MHz],
            and lower state energies El [1/cm]
        nu
            Target frequency of the absorption [Hz]
        br_params
            Parameters for the Ben-Reuven line expression
        eps_prime_r_inf
            Real part of the relative permittivity at infinite frequency
        freqmin
            Minimum frequency of the integration domain
        freqmax
            Maximum frequency of the integration domain
        freqstep
            Frequency step of the integration domain

        Returns
        -------
            Real part of the relative permittivity
        """
        # get integration domain
        freqrange = Quantity(
            np.arange(
                freqmin.to("GHz").value,
                freqmax.to("GHz").value,
                freqstep.to("GHz").value,
            ),
            "GHz",
        )
        freqdiff = freqrange - nu
        # mask out singularities
        freqdiff[np.abs(freqdiff) < freqstep / 2] = np.nan
        # compute absorption coefficient
        alpha = Duan2010.eq27(
            np.atleast_1d(T),
            np.atleast_1d(P),
            Quantity([0], "atm"),
            spectral_lines,
            freqrange,
            br_params,
        ).squeeze()
        # convert from absorption coefficient to imaginary part of the
        # relative permittivity
        eps_dprime = (
            alpha / freqrange.to("1/cm", equivalencies=u.spectral()) / (2 * np.pi)
        ).decompose()
        # integrate through profile
        delta_eps_prime_r = (
            np.trapezoid(
                np.nan_to_num((eps_dprime / freqdiff).to("1/GHz").value),
                x=freqrange.to("GHz").value,
            )
            / np.pi
        )
        # add to infinite value
        eps_prime_r = eps_prime_r_inf + delta_eps_prime_r
        # done
        return Quantity(eps_prime_r, u.dimensionless_unscaled)


class VariableProfiles(Duan2010):

    profiles_tpd: NDArray[np.floating] | None = None
    """
    Temperature, pressure and mass densities for a range of
    latitudes and local solar times
    """
    profiles_el: NDArray[np.floating] | None = None
    """ Electron density profiles for a range of solar zenith angles """
    latitudes: NDArray[np.floating] | None = None
    """ Latitudes [°] of profiles """
    localtimes: NDArray[np.floating] | None = None
    """ Local solar times [h] of profiles """
    szas: NDArray[np.floating] | None = None
    """ Solar zenith angles [°] of profiles """

    def __init__(
        self,
        use_compressible_gas: bool = True,
        use_keating_temp_press_above100km: bool = False,
        use_keating_co_co2_n2_above_100km: bool = False,
        use_kolste_h2so4: bool = False,
        use_marcq_ocs: bool = False,
        add_ar: bool = False,
        cutoff_so2_frequency: Quantity["frequency"] | None = None,
        use_kolbe_ocs: bool = False,
        use_virial_approximation: bool = True,
        use_cimino_clouds: bool = True,
        use_cimino_fitted_lookup: bool = False,
        min_altitude_spacing: Quantity = Quantity(1, "km"),
        add_3K: bool = False,
    ):
        """
        Warning
        -------
        This class is still experimental, might change or break at any time,
        and does not have usage documentation yet.
        """
        # save init variables for later use
        self._add3K = add_3K
        self._cutoff_so2_frequency = cutoff_so2_frequency
        self._use_kolbe_ocs = use_kolbe_ocs
        self._use_cimino_clouds = use_cimino_clouds
        self._use_cimino_fitted_lookup = use_cimino_fitted_lookup
        # call parent initializer
        super().__init__(
            use_compressible_gas,
            use_keating_temp_press_above100km,
            use_keating_co_co2_n2_above_100km,
            use_kolste_h2so4,
            use_marcq_ocs,
            add_ar,
            cutoff_so2_frequency,
            use_kolbe_ocs,
            use_virial_approximation,
            use_cimino_clouds,
            use_cimino_fitted_lookup,
            min_altitude_spacing,
        )

    def prepare_profiles(
        self, latitudes: Quantity["angle"], localtimes: Quantity["time"]
    ):
        """
        Precompute a range of temperature, pressure, and density profiles,
        for later application to the model.

        Parameters
        ----------
        latitudes
            Latitudes [°] of profiles
        localtimes
            Local solar times [h] of profiles

        Notes
        -----
        Sets :attr:`~VariableProfiles.latitudes`,
        :attr:`~VariableProfiles.localtimes`,
        :attr:`~VariableProfiles.profiles_tpd`, and
        :attr:`~VariableProfiles.profiles_el` attributes.
        """
        # run temperature, pressure, and density model
        self.szas, self.profiles_tpd = seiffkeating(latitudes, localtimes, self._add3K)
        # run ionosphere model, save in log space for easier interpolation
        profiles_el = paetzold2007(self.szas.ravel()).reshape(
            -1, localtimes.size, latitudes.size
        )
        profiles_el[profiles_el == 0] = np.nan
        self.profiles_el = np.log10(profiles_el.value)
        # save input
        self.latitudes = latitudes
        self.localtimes = localtimes
        # done

    def set_iono(self, ilat: int, itime: int, update: bool = True):
        """
        Update the base model with the ionosphere from a
        :class:`~xvamp.reference.Paetzold2007` model at the given
        latitude and local time.

        Parameters
        ----------
        ilat
            Index of latitude [°] of loaded profiles
        itime
            Index of local solar time [h] of loaded profiles
        update
            Whether to update the rest of the model after setting the
            electron density profile

        Notes
        -----
        Overwrites :attr:`~Model.electron_density`, then calls
        :meth:`~Duan2010.update_ionosphere` and
        :meth:`~Duan2010.update_rel_perm_refraction`.
        """
        # input check
        assert self.profiles_el is not None, "Need to run 'prepare_profiles' first!"
        # extract desired values from profile
        profile = self.profiles_el[:, itime, ilat]
        # interpolate to current altitude levels
        electron_density = 10 ** np.interp(
            self.altitude.to("km").value,
            paetzold2007.el_altitude.to("km").value,
            profile.value,
            left=np.nan,
            right=np.nan,
        )
        electron_density[np.isnan(electron_density)] = 0
        self.electron_density = Quantity(electron_density, paetzold2007.UNIT_EL_DENSITY)
        # call update methods
        if update:
            self.update_from_ionosphere()
        # done

    def set_tpd(self, ilat: int, itime: int, update: bool = True):
        """
        Update the base model with the temperature and pressure
        from a :class:`~xvamp.reference.SeiffKeating` model at the
        given latitude and local time.

        Parameters
        ----------
        ilat
            Index of latitude [°] of loaded profiles
        itime
            Index of local solar time [h] of loaded profiles
        update
            Whether to update the rest of the model after setting the
            temperature and pressure profiles

        Notes
        -----
        Overwrites :attr:`~Model.altitude` and :attr:`~Model.temperature`,
        then calls :meth:`~Duan2010.update_densities`,
        :meth:`~Duan2010.update_pol_absorp_atmosphere`,
        :meth:`~Duan2010.update_ionosphere`,
        and :meth:`~Duan2010.update_rel_perm_refraction`.
        """
        # input check
        assert self.profiles_tpd is not None, "Need to run 'prepare_profiles' first!"
        # extract desired values from profile
        profile = self.profiles_tpd[:, :, itime, ilat]
        # convert to their units
        alt_prof = Quantity(profile[:, 0], seiffkeating.UNITS[0])
        temp_prof = Quantity(profile[:, 1], seiffkeating.UNITS[1])
        press_prof = Quantity(profile[:, 2], seiffkeating.UNITS[2])
        dens_prof = Quantity(profile[:, 3], seiffkeating.UNITS[3])
        # get current units
        unit_alt = self.altitude.unit
        unit_temp = self.temperature.unit
        unit_press = self.pressure.unit
        unit_dens = self.mass_density.unit
        # get extrapolated below 0 km
        alt_neg, temp_neg, press_neg, dens_neg = Model.tpd_below_0km(
            Duan2010.VENUS_GAS_CONSTANT
        )
        # combine
        new_alt = np.concatenate(
            [alt_neg.to(unit_alt).value, alt_prof.to(unit_alt).value]
        )
        new_temp = np.concatenate(
            [temp_neg.to(unit_temp).value, temp_prof.to(unit_temp).value]
        )
        new_press = np.concatenate(
            [press_neg.to(unit_press).value, press_prof.to(unit_press).value]
        )
        new_dens = np.concatenate(
            [dens_neg.to(unit_dens).value, dens_prof.to(unit_dens).value]
        )
        # interpolate to current altitudes
        altitude = self.altitude.to(unit_alt).value
        self.temperature = Quantity(
            np.interp(altitude, new_alt, new_temp, left=np.nan, right=None),
            unit_temp,
        )
        self.pressure = Quantity(
            np.interp(altitude, new_alt, new_press, left=np.nan, right=np.nan),
            unit_press,
        )
        self.mass_density = Quantity(
            np.interp(altitude, new_alt, new_dens, left=np.nan, right=np.nan),
            unit_dens,
        )
        # call update methods
        if update:
            self.update_from_densities()
        # done

    def update_from_ionosphere(self):
        """Call all update methods relying on the ionosphere"""
        # call update methods
        self.update_ionosphere()
        self.update_rel_perm_refraction()
        # done

    def update_from_densities(self):
        """Call all update methods relying on the density profiles"""
        # call update methods
        self.update_densities()
        self.update_pol_absorp_atmosphere(
            cutoff_so2_frequency=self._cutoff_so2_frequency,
            use_kolbe_ocs=self._use_kolbe_ocs,
            use_cimino_clouds=self._use_cimino_clouds,
            use_cimino_fitted_lookup=self._use_cimino_fitted_lookup,
        )
        self.update_from_ionosphere()
        # done
