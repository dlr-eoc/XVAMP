"""
Model class that loads all the reference data, maybe adds its own,
and returns permittivity.
"""

# standard imports
import numpy as np
import astropy.units as u
import astropy.table as astrotable
from warnings import warn
from pathlib import Path
from typing import Tuple
from astropy.units import Quantity, Unit
from astropy.table import QTable

# package imports
from ..constants import *
from ..utils import float_or_array
from ..profile import Profile, MultiProfile, check_physical_type
from ..utils.io import read_polarization_parameters
from ..utils.parametersets import (
    HarveyLemmon2005Parameters,
    Pitzer1983Parameters,
    LineShapeParameters,
)
from ..references import (
    seiff_et_al_1985 as seiff1985,
    zasova_et_al_2006 as zasova2006,
    keating_et_al_1985 as keating1985,
    duan_et_al_2010 as duan2010figures,
    kolodner_steffes_1998 as kolodnersteffes1998,
    james_et_al_1997 as james1997,
    jpl_spectral_lines as jplspectrallines,
    cimino_1982 as cimino1982,
    vonzahn_moroz_1985 as zahnmoroz1985,
)
from .model import Model


# Model class that implements the Duan et al. (2010) paper
class Duan2010(Model):

    # general constants
    VENUS_GAS_CONSTANT = Quantity(191.4, "J/kg K")
    """ Venus standard atmospheric gas constant (= R/M) [J/kg K] """
    VENUS_MOLAR_MASS = (
        zahnmoroz1985.CO2_MR * SPEC_MOL_M["CO2"]
        + zahnmoroz1985.N2_MR * SPEC_MOL_M["N2"]
    ).to("kg/mol")
    """ Venus standard atmospheric molar mass [kg/mol] """
    TRANSITION_ATMO_IONO = Quantity(100, "km")
    """
    Altitude at which the computation of the real part of the relative
    permittivity switches from the individual components in the
    atmosphere to the overall effect of the ionosphere
    """

    # constants relating to Argon (Ar)
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
    EPS_PRIME_R_INF_OCS = Quantity(1.005862637533891, u.dimensionless_unscaled)
    """ Estimated dielectric constant of OCS at infinite frequency """
    P_OCS = Quantity(101325, "Pa")
    """ Pressure at which the dielectric constant for OCS was calculated """
    T_OCS = Quantity(273.18, "K")
    """ Temperature at which the dielectric constant for OCS was calculated """
    RHO_OCS = ((P_OCS) / (GAS_CONSTANT * T_OCS)).decompose()
    """ Molar density from ``P_OCS`` and ``T_OCS`` """
    MU_OCS = Quantity(0.71521e-18, ESU_CM)
    """ Permanent dipole moment of OCS [esu cm] """
    BR_SO2_AS_OCS_CO2 = LineShapeParameters(
        T_0=Quantity(300, "K"),
        gamma_min_maj=Quantity(7.2, "MHz/torr"),
        gamma_min_min=Quantity(16, "MHz/torr"),
        m=0.85,
        n=0.85,
    )
    """
    Ben-Reuven line parameters for OCS in CO2 derived from the SO2 in CO2 parameters
    but setting zeta and delta to zero
    """
    BR_OCS_CO2 = LineShapeParameters(
        T_0=Quantity(300, "K"),
        gamma_min_maj=Quantity(4.3, "MHz/torr"),
        gamma_min_min=Quantity(5.9, "MHz/torr"),
        m=0.7,
        n=0.7,
    )
    """
    Ben-Reuven line parameters for OCS in CO2 based on visual inspection of
    :cite:t:`bouanich1988` and :cite:t:`lavrentieva2020`
    """
    L_OCS = LineShapeParameters(
        T_0=Quantity(300, "K"), gamma_min_min=Quantity(6.4, "MHz/torr")
    )
    """ Lorentzian line parameters for OCS from :cite:t:`kolbe1977` """

    # constants relating to sulfur dioxide (SO2)
    EPS_PRIME_R_INF_SO2 = Quantity(1.005862637533891, u.dimensionless_unscaled)
    """ Estimated dielectric constant of SO2 at infinite frequency """
    P_SO2 = Quantity(101325, "Pa")
    """ Pressure at which the dielectric constant for SO2 was calculated """
    T_SO2 = Quantity(273.15, "K")
    """ Temperature at which the dielectric constant for SO2 was calculated """
    RHO_SO2 = ((P_SO2) / (GAS_CONSTANT * T_SO2)).decompose()
    """ Molar density from ``P_SO2`` and ``T_SO2`` """
    MU_SO2 = Quantity(1.633e-18, ESU_CM)
    """ Permanent dipole moment of SO2 [esu cm] """
    BR_SO2_CO2 = LineShapeParameters(
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

    :meta hide-value:
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

    :meta hide-value:
    """

    # for the numerical detail
    MIN_ALTITUDE_SPACING = Quantity(1, "km")
    """
    Minimum height spacing between altitude nodes. Only becomes relevant if the
    loaded profiles of the physical and chemical quantities are not dense enough
    to ensure an accurate numerical integration.
    """

    def __init__(
        self,
        profile_TPD: MultiProfile | str = "duan",
        profile_CO2: Profile | None = zahnmoroz1985.co2_molar_fraction,
        profile_N2: Profile | None = zahnmoroz1985.n2_molar_fraction,
        profile_H2O: Profile | None = duan2010figures.h2o_molar_fraction,
        profile_SO2: Profile | None = duan2010figures.so2_molar_fraction,
        profile_CO: Profile | None = duan2010figures.co_molar_fraction,
        profile_H2SO4: Profile | None = duan2010figures.h2so4_molar_fraction,
        profile_OCS: Profile | None = duan2010figures.ocs_molar_fraction,
        profile_Ar: Profile | None = None,
        use_clouds_from: str = "cimino",
        ocs_abspol_from: str = "duan",
        use_eps_prime_r_inf: bool = True,
        load_polarization_parameters: bool | str | Path = True,
        use_compressible_gas: bool = True,
        use_keating_temp_press_above100km: bool = False,
        use_virial_approximation: bool = True,
        cutoff_so2_frequency: Quantity["frequency"] | None = None,
        use_cimino_fitted_lookup: bool = False,
    ) -> None:
        """
        Initialize the :cite:t:`duan2010` model. All parameters are set such that
        they correspond to the Matlab ``config.atm_recipe = 'all_standard'`` setting.

        Parameters
        ----------
        profile_TPD
            Which temperature, pressure, and density profile to use:

            - ``"duan"``: A combination of :cite:t:`seiff1985` and :cite:t:`zasova2006`
              as described in the paper, Section 3.1 (i.e., including the 3 K offset).
            - ``"seiff:x"``: A specific profile of :cite:t:`seiff1985` for a given
              latitude *x* (valid values: 30, 45, 60, 75, 85) in degrees.

            Note that these preconfigured profiles are all downward-continued to
            negative altitudes, and are influenced by the the parameters
            ``use_compressible_gas`` and ``use_keating_temp_press_above100km``.
            Alternatively, a :class:`xvamp.profile.MultiProfile` with the data columns
            ``"temperature"``, ``"pressure"``, and optionally ``"mass_density"``
            (and the index being the altitude).
        profile_CO2
            CO2 molar fraction profile.
        profile_N2
            N2 molar fraction profile.
        profile_H2O
            H2O molar fraction profile.
        profile_SO2
            SO2 molar fraction profile.
        profile_CO
            CO molar fraction profile.
        profile_H2SO4
            H2SO4 molar fraction profile. Preconfigured options are:

            - :attr:`~xvamp.references.duan2010figures.h2so4_molar_fraction` or
              :attr:`~xvamp.references.duan2010figures.h2so4_3212_molar_fraction`
              from :cite:t:`duan2010` and the reference code.
            - :attr:`~xvamp.references.kolodnersteffes1998.h2so4_mr_mean` (mean) or
              :attr:`~xvamp.references.kolodnersteffes1998.h2so4_mr_3212` (where
              ``3212``, ``3213`` and ``3214`` are individual orbits) from
              :cite:t:`kolodner1998`, Figs. 7-9. This option adds about a tenth of a
              dB attenuation and removes about 4 mm of delay.
            - :attr:`~xvamp.references.jenkins2002.h2so4_molar_fraction_0ppm_so2` (where
              ``0``, ``50``, ``100``, ``150``, ``200`` are assumptions about the SO2
              content) from :cite:t:`jenkins2002`. This changes the
              attenuation by about a tenth of a dB and the delay by some millimeters.
            - :attr:`~xvamp.references.magellan321x.h2so4_mr_x_3212` (where
              ``3212``, ``3213`` and ``3214`` are individual orbits) from
              :cite:t:`jenkins1996a`.  This changes the
              attenuation by about a dB and the delay of some millimeters.
        profile_OCS
            OCS molar fraction profile. Preconfigured options are:

            - :attr:`~xvamp.references.duan2010figures.co_molar_fraction`
            - :attr:`~xvamp.references.marcq2006.ocs_mr` from :cite:t:`marcq2006`

            This has a range delay effect on the sub-millimeter scale, and an effect
            on the two-way attenuation on the millidecibel scale.
        profile_Ar
            Argon molar fraction profile. The default is not to add Argon to the
            mixture, but a preconfigured (constant) profile is
            :attr:`~xvamp.referenceszahnmoroz1985.ar_molar_fraction`.
            This has a range delay effect on the sub-micrometer scale, and an effect
            on the two-way attenuation on the tens of microdecibel scale.
        use_clouds_from
            Define which cloud polarization and absorption model to use:

            - ``"cimino"``: :cite:t:`cimino1982`, eq. (10) and (16)
            - ``"duan"``: :cite:t:`duan2010`, sections 2.1.5 and 2.2.5
            - ``"none"``: Ignore all cloud effects

            See the notes on the importance of this parameter at
            :ref:`implementation:Cloud polarization and absorption`.
        use_compressible_gas
            Whether to use the gas compressibility factor when deriving the mass
            density for the 0-100 km altitude range, or assume the ideal gas law.
            This only affects the attenuation of the cloud layer, since all other
            species quantities are derived from the pressure profile, which is directly
            loaded from :cite:t:`seiff1985` and :cite:t:`zasova2006`.
            The attenuation difference is about 2 millidecibels.
            If a :class:`xvamp.profile.MultiProfile` is passed as the ``profile_TPD``
            parameter and contains a mass density, ``use_compressible_gas`` is ignored.
        ocs_abspol_from
            Define which model to use to compute the absorption and polarization
            profiles of OCS.

            - ``"duan"``: Using a Ben-Reuven line shape derived from SO2 (default)
            - ``"kolbe"``: Using a Lorentzian line shape as described in the paper
              and following :cite:t:`kolbe1977`
            - ``"bbld"``: Using a Ben-Reuven line shape with parameters derived
              approximately from :cite:t:`bouanich1988` and :cite:t:`lavrentieva2020`.

            Since OCS is such a minor constituent, the different options have a
            sub-millimeter effect on the delay and a milli-decibel effect on the
            attenuation. If changing the default, then also set
            ``load_polarization_parameters=False``, as the setting affects the
            polarization parameters.
        use_eps_prime_r_inf
            If ``True``, when computing the real part of the relative permittivity
            of SO2 and OCS, a value of the real relative permittivity at infinite
            frequency is set to an assumed value (rather than using the theoretical
            value of unity). This only has an effect if
            ``load_polarization_parameters=False``, because the polarization parameters
            resulting from the real part of the relative permittivity are stored.
            This option has a centimeter-level effect on the delay and changes the
            attenuation by micro-decibels.
        load_polarization_parameters
            By default, the polarization parameters are loaded from a prepackaged
            configuration file (in ``"data/default_polarization_parameters.toml"``).
            If set to ``False``, they are recomputed with the current settings.
            If set to a filename, the parameters are loaded from there.

        Other Parameters
        ----------------
        use_keating_temp_press_above100km
            Only used if ``profile_TPD`` is not a :class:`xvamp.profile.MultiProfile`.
            Whether to use the temperature profile from :cite:t:`keating1985`
            above 100 km, and get its matching pressure profile from the
            ideal gas law.
            This option has no effect on the model, since the transition between
            atmosphere- and ionosphere-dominated permittivity profiles is at 100 km,
            and the ionosphere is modeled differently. It is only useful if one
            wants to load these quantities for later plotting.
        use_virial_approximation
            Whether to use the leading terms of the virial approximation to calculate
            the total polarization of the polar species :cite:p:`harvey2005`,
            or to use the polarization relationship by :cite:t:`pitzer1983`.
            These two approaches are numerically fully equivalent.
        cutoff_so2_frequency
            When computing the absorption coefficient of SO2, include all spectral
            lines up to this frequency. If ``None``, use all available ones.
            This option is only kept for development purposes.
        use_cimino_fitted_lookup
            Whether to estimate the complex permittivity of gaseous H2SO4 from
            lookup tables and then pre-fitted analytical extrapolation functions,
            or to numerically inter- and extrapolate.
            This option is only kept for development purposes, since the pre-fitted
            model is flawed. Regardless, this options only has a range delay effect
            on the sub-micrometer scale, and an effect on the two-way attenuation
            on the millidecibel scale.
        """

        # part 1: physical quantities

        # temperature, pressure and optionally density
        if isinstance(profile_TPD, MultiProfile):
            prof_tpd = profile_TPD
            # check if temperature is present and has the right units
            if not "temperature" in prof_tpd.data_names:
                raise ValueError(
                    "MultiProfile passed for 'profile_TPD' "
                    "must contain a 'temperature' entry."
                )
            check_physical_type(
                prof_tpd.temperature, "temperature", "length", "profile_TPD.temperature"
            )
            # check pressure
            if not "pressure" in prof_tpd.data_names:
                raise ValueError(
                    "MultiProfile passed for 'profile_TPD' "
                    "must contain a 'pressure' entry."
                )
            check_physical_type(prof_tpd.pressure, "pressure", "profile_TPD.pressure")
            # check mass density
            if "mass_density" in prof_tpd.data_names:
                use_compressible_gas = True
                check_physical_type(
                    prof_tpd.mass_density, "mass density", "profile_TPD.mass_density"
                )
        else:
            prof_tpd = Duan2010.get_tpd(
                profile_TPD=profile_TPD,
                use_compressible_gas=use_compressible_gas,
                use_keating_temp_press_above100km=use_keating_temp_press_above100km,
            )

        # part 2: compositional profiles

        # get the mixing ratios of the chemical species
        dict_prof_species = {}
        if profile_CO2 is not None:
            dict_prof_species["CO2"] = profile_CO2
        if profile_N2 is not None:
            dict_prof_species["N2"] = profile_N2
        if profile_H2O is not None:
            dict_prof_species["H2O"] = profile_H2O
        if profile_SO2 is not None:
            dict_prof_species["SO2"] = profile_SO2
        if profile_CO is not None:
            dict_prof_species["CO"] = profile_CO
        if profile_H2SO4 is not None:
            dict_prof_species["H2SO4"] = profile_H2SO4
        if profile_OCS is not None:
            dict_prof_species["OCS"] = profile_OCS
        if profile_Ar is not None:
            dict_prof_species["AR"] = profile_Ar
        # check their units
        for sname, sprof in dict_prof_species.items():
            check_physical_type(sprof, "dimensionless", "length", f"profile_{sname}")

        # get the electron density
        prof_electrons = duan2010figures.electron_density

        # get cloud profiles
        if use_clouds_from != "none":
            prof_cloud_concentration = james1997.cloud_concentration
            prof_cloud_mass_mixing_ratio = james1997.cloud_mass_mixing_ratio

        # part 3: interpolate all physical quantities, mixing ratios, electron density,
        # and cloud profile onto the same altitude levels, ensuring a minimum spacing
        # of altitude values

        # get all altitude levels and add a minimum spacing
        km = Unit("km")
        min_alt_spacing_km = Duan2010.MIN_ALTITUDE_SPACING.to_value(km)
        joint_alt = (
            [
                prof_tpd.index_to(km),
                prof_electrons.index_to(km),
                np.arange(
                    -7, 375 + min_alt_spacing_km / 2, min_alt_spacing_km, dtype=float
                ),
            ]
            + [p.index_to(km) for p in dict_prof_species.values()]
            + (
                [
                    prof_cloud_concentration.index_to(km),
                    prof_cloud_mass_mixing_ratio.index_to(km),
                ]
                if use_clouds_from != "none"
                else []
            )
        )
        # combine for joint altitude profile
        self.altitude = Quantity(np.unique(np.concatenate(joint_alt)), km)

        # evaluate all Profiles
        # temperature, pressure, and density
        self.temperature = prof_tpd.temperature(self.altitude)
        self.pressure = prof_tpd.pressure(self.altitude)
        if use_compressible_gas:
            self.mass_density = prof_tpd.mass_density(self.altitude)
        # electrons
        self.electron_density = prof_electrons(self.altitude)
        # clouds
        if use_clouds_from != "none":
            self.cloud_concentration = prof_cloud_concentration(self.altitude)
            self.cloud_mass_mixing_ratio = prof_cloud_mass_mixing_ratio(self.altitude)
        else:
            self.cloud_mass_mixing_ratio = Quantity(0, u.dimensionless_unscaled)
        # species
        self.molar_fractions = QTable(
            {spec: prof(self.altitude) for spec, prof in dict_prof_species.items()}
        )

        # part 4: computation of total and per-species densities

        self.update_densities()
        # sets self.mass_density (if not already present), self.number_density,
        # self.molar_density, self.[molar_densities,mass_densities], and
        # self.cloud_mass_density

        # part 5: get individual contributions to polarization and absorption
        # for each species and the clouds in the atmosphere, as well as the
        # resulting real part of the relative permittivity

        # the computation of the polarization parameters is independent of
        # the loaded atmospheric profiles

        # warn if we should recompute the polarization parameters
        if (load_polarization_parameters == True) and (
            (ocs_abspol_from != "duan") or (not use_eps_prime_r_inf)
        ):
            post_warn = (
                "while loading the default polarization parameters "
                f"({load_polarization_parameters=}) will yield inconsistent results. "
                "Either recompute the polarization parameters, "
                "or load them from an appropriate custom file."
            )
            if ocs_abspol_from != "duan":
                warn(f"Choosing the non-default {ocs_abspol_from=} {post_warn}")
            if not use_eps_prime_r_inf:
                warn(f"Choosing the non-default {use_eps_prime_r_inf=} {post_warn}")

        # check if we should recompute them
        if load_polarization_parameters == False:
            self.polarization_parameters = Duan2010.compute_polarization_parameters(
                ocs_abspol_from=ocs_abspol_from,
                use_eps_prime_r_inf=use_eps_prime_r_inf,
                use_virial_approximation=use_virial_approximation,
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
            ocs_abspol_from=ocs_abspol_from,
            use_clouds_from=use_clouds_from,
            use_cimino_fitted_lookup=use_cimino_fitted_lookup,
        )
        # sets self.polarization[s], self.absorption[s], and self.eps_prime_r_atmo

        # part 6: ionosphere

        self.update_ionosphere()
        # sets self.eps_prime_r_iono

        # part 7: combine all contributions

        self.update_rel_perm_refraction()
        # sets self.relative_permittivity and self.refraction

        # done

    def update_densities(self):
        """
        Compute the total and specific mass, number, and molar densities
        from the total pressure and temperature, and the molar fractions.
        Also computes the cloud mass density from the atmospheric profile
        and the cloud concentration and mass mixing ratio.
        If the mass density has not been set yet, it is derived from the
        ideal gas law.

        Notes
        -----
        Reads: :attr:`~xvamp.models.model.Model.pressure`,
        :attr:`~xvamp.models.model.Model.temperature`,
        :attr:`~xvamp.models.model.Model.molar_fractions`,
        and :attr:`~xvamp.models.model.Model.cloud_mass_mixing_ratio`

        Writes: :attr:`~xvamp.models.model.Model.number_density`,
        :attr:`~xvamp.models.model.Model.mass_densities`,
        :attr:`~xvamp.models.model.Model.molar_density`,
        :attr:`~xvamp.models.model.Model.molar_densities`,
        :attr:`~xvamp.models.model.Model.cloud_mass_density` and
        (if not already present) :attr:`~xvamp.models.model.Model.mass_density`
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

        # cloud density
        self.cloud_mass_density = (
            self.cloud_mass_mixing_ratio * self.mass_density
        ).decompose()

        # done

    def update_pol_absorp_atmosphere(
        self,
        cutoff_so2_frequency: Quantity["frequency"] | None = None,
        ocs_abspol_from: str = "duan",
        use_clouds_from: str = "cimino",
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
        ocs_abspol_from
            Define which model to use to compute the absorption and polarization
            profiles of OCS.

            - ``"duan"``: Using a Ben-Reuven line shape derived from SO2 (default)
            - ``"kolbe"``: Using a Lorentzian line shape as described in the paper
              and following :cite:t:`kolbe1977`
            - ``"bbld"``: Using a Ben-Reuven line shape with parameters derived
              approximately from :cite:t:`bouanich1988` and :cite:t:`lavrentieva2020`.

        use_clouds_from
            Define which cloud polarization and absorption model to use:

            - ``"cimino"``: :cite:t:`cimino1982`, eq. (10) and (16)
            - ``"duan"``: :cite:t:`duan2010`, sections 2.1.5 and 2.2.5
            - ``"none"``: Ignore all cloud effects

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

        Notes
        -----
        Reads: :attr:`~xvamp.models.model.Model.polarization_parameters`,
        :attr:`~xvamp.models.model.Model.temperature`,
        :attr:`~xvamp.models.model.Model.pressure`,
        :attr:`~xvamp.models.model.Model.molar_fractions`,
        :attr:`~xvamp.models.model.Model.molar_densities`,
        :attr:`~xvamp.models.model.Model.mass_densities`,
        :attr:`~xvamp.models.model.Model.cloud_concentration`,
        and :attr:`~xvamp.models.model.Model.cloud_mass_density`.

        Writes: :attr:`~xvamp.models.model.Model.polarizations`,
        :attr:`~xvamp.models.model.Model.polarization`,
        :attr:`~xvamp.models.model.Model.absorptions`,
        :attr:`~xvamp.models.model.Model.absorption`, and
        :attr:`~xvamp.models.model.Model.eps_prime_r_atmo`.
        """

        # sections 2.1.3-2.1.4: non-polar and polar components
        # convert polarization parameters to actual polarizations
        self.polarizations = self.evaluate_polarization_parameters()

        # sections 2.2.1-2.2.4: absorptions from species
        self.absorptions = self.evaluate_absorptions(
            cutoff_so2_frequency=cutoff_so2_frequency,
            ocs_abspol_from=ocs_abspol_from,
        )

        # sections 2.1.5 and 2.2.5: clouds
        # add quantities to existing QTable
        if use_clouds_from != "none":
            self.polarizations["cloud"], self.absorptions["cloud"] = (
                self.evaluate_cloud_permittivity(
                    use_clouds_from=use_clouds_from,
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
        Reads: :attr:`~xvamp.models.model.Model.electron_density`.

        Writes: :attr:`~xvamp.models.model.Model.eps_prime_r_iono`.
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
        Reads: :attr:`~xvamp.models.model.Model.altitude`,
        :attr:`~xvamp.models.model.Model.eps_prime_r_atmo`,
        :attr:`~xvamp.models.model.Model.eps_prime_r_iono`,
        and :attr:`~xvamp.models.model.Model.absorption`.

        Writes: :attr:`~xvamp.models.model.Model.relative_permittivity`
        and :attr:`~xvamp.models.model.Model.refraction`.
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
        profile_TPD: str = "duan",
        use_compressible_gas: bool = True,
        use_keating_temp_press_above100km: bool = False,
    ) -> MultiProfile:
        """
        Build the temperature, pressure, and mass density profiles.

        Parameters
        ----------
        profile_TPD
            Which temperature, pressure, and density profile to use:

            - ``"duan"``: A combination of :cite:t:`seiff1985` and :cite:t:`zasova2006`
              as described in the paper, Section 3.1 (i.e., including the 3 K offset).
            - ``"seiff:x"``: A specific profile of :cite:t:`seiff1985` for a given
              latitude *x* (valid values: 30, 45, 60, 75, 85) in degrees.

            Note that these preconfigured profiles are all downward-continued to
            negative altitudes.
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
            :class:`~xvamp.profile.Multiprofile` with altitude as the index and
            temperature and pressure as data columns. If ``use_compressible_gas=True``,
            also has the mass density as a data column.
        """

        # start with the basic profile from Seiff et al. (1985) for the deep atmosphere
        alt = [seiff1985.tables["1-1"]["z"]]
        temp = [seiff1985.tables["1-1"]["T"]]
        press = [seiff1985.tables["1-1"]["p"]]
        if use_compressible_gas:
            dens = [seiff1985.tables["1-1"]["ρ"]]

        # now, determine which continuation to make
        match profile_TPD.split(":"):

            # the default model, combining Seiff et al. (1985) and Zasova et al. (2006)
            case ["duan"]:

                # p. 13: "for the lower atmosphere, [...] the temperature curve at
                # latitude of 75° in the work of Seiff et al. (1985) is used after being
                # increased by 3 K"
                seiff_lat_table = "1-2d"
                # p. 14: "In the simulation, the middle atmosphere temperature and
                # pressure profiles are using the column of Ls = 200°–270° in
                # Table 5 of Zasova et al. (2006)"
                ix_seiff_below_zasova = (
                    seiff1985.tables[seiff_lat_table]["z"]
                    < zasova2006.tables["5"]["H"][-1]
                )
                alt.extend(
                    [
                        seiff1985.tables[seiff_lat_table]["z"][ix_seiff_below_zasova],
                        zasova2006.tables["5"]["H"][::-1],
                    ]
                )
                temp = [
                    temp[0] + 3 * u.K,
                    seiff1985.tables[seiff_lat_table]["T"][ix_seiff_below_zasova]
                    + 3 * u.K,
                    zasova2006.tables["5"]["Ls = 200°-270°, T"][::-1],
                ]
                press.extend(
                    [
                        seiff1985.tables[seiff_lat_table]["p"][ix_seiff_below_zasova],
                        zasova2006.tables["5"]["Ls = 200°-270°, P"][::-1],
                    ]
                )
                # interpolate the pressure levels of Zasova et al. (2006) onto
                # compressible density profile from Seiff et al. (1985)
                if use_compressible_gas:
                    dens.extend(
                        [
                            seiff1985.tables[seiff_lat_table]["ρ"][
                                ix_seiff_below_zasova
                            ],
                            Quantity(
                                np.interp(
                                    press[-1].to("bar").value,
                                    seiff1985.tables[seiff_lat_table]["p"]
                                    .to("bar")
                                    .value[::-1],
                                    seiff1985.tables[seiff_lat_table]["ρ"].value[::-1],
                                ),
                                seiff1985.tables[seiff_lat_table]["ρ"].unit,
                            ),
                        ]
                    )

                # extrapolate to negative altitudes
                alt_neg, temp_neg, press_neg, dens_neg = Model.tpd_below_0km(
                    Duan2010.VENUS_GAS_CONSTANT,
                    add_3K=True,
                )

            # use a specific Seiff et al. (1985) profile directly
            case ["seiff", x] if int(x) in [30, 45, 60, 75, 85]:

                # simply add the corresponding table to the lists
                seiff_lat_table = seiff1985.LAT_TABLES[
                    seiff1985.LAT.value.tolist().index(int(x))
                ]
                alt.append(seiff1985.tables[seiff_lat_table]["z"])
                temp.append(seiff1985.tables[seiff_lat_table]["T"])
                press.append(seiff1985.tables[seiff_lat_table]["p"])
                if use_compressible_gas:
                    dens.append(seiff1985.tables[seiff_lat_table]["ρ"])

                # extrapolate to negative altitudes
                alt_neg, temp_neg, press_neg, dens_neg = Model.tpd_below_0km(
                    Duan2010.VENUS_GAS_CONSTANT,
                    add_3K=False,
                )

            case _:
                raise ValueError(f"Unknown TPD model {profile_TPD=}")

        # insert negative profiles into lists
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
            alt.extend(
                [
                    keating1985.tables["4-15"]["ALT"][-2:0:-1],
                    keating1985.tables["4-5"]["ALT"],
                ]
            )
            temp.extend(
                [
                    keating1985.tables["4-15"]["T"][-2:0:-1],
                    keating1985.tables["4-5"]["T"],
                ]
            )
            press.extend(
                [
                    keating1985.tables["4-15"]["P"][-2:0:-1],
                    keating1985.tables["4-7"]["P"],
                ]
            )
            if use_compressible_gas:
                dens.extend(
                    [
                        keating1985.tables["4-15"]["RHO"][-2:0:-1],
                        keating1985.tables["4-5"]["RHO"],
                    ]
                )
        # otherwise, we use a previously-fitted extrapolating function for pressure,
        # continue the temperature as a constant, and use the ideal gas law to get
        # mass density
        else:
            extrap_alt_km = np.arange(101, 376)
            alt.append(Quantity(extrap_alt_km, "km"))
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
                )
            )
            if use_compressible_gas:
                dens.append(press[-1] / (Duan2010.VENUS_GAS_CONSTANT * temp[-1]))

        # combine into MultiProfile
        # concatenate all profiles
        altitude = np.concatenate(alt)
        temperature = np.concatenate(temp)
        pressure = np.concatenate(press)
        # convert pressure into log space since that's where we want it to be
        # interpolated
        with np.errstate(invalid="raise"):
            pressure = Quantity(np.log10(pressure.value), pressure.unit)
            if use_compressible_gas:
                mass_density = np.concatenate(dens)
                mass_density = Quantity(np.log10(mass_density.value), mass_density.unit)
        # combine all the data columns
        data = [temperature, pressure]
        data_names = ["temperature", "pressure"]
        log_list = [False, True]
        upper_list = [None, np.nan]
        if use_compressible_gas:
            data.append(mass_density)
            data_names.append("mass_density")
            log_list.append(True)
            upper_list.append(np.nan)
        # convert everything
        tpd = MultiProfile(
            index=altitude,
            data=QTable(data, names=data_names),
            log=log_list,
            lower=np.nan,
            upper=upper_list,
        )

        # done
        return tpd

    @staticmethod
    def compute_polarization_parameters(
        ocs_abspol_from: str = "duan",
        use_eps_prime_r_inf: bool = True,
        use_virial_approximation: bool = True,
    ) -> dict[str, HarveyLemmon2005Parameters | Pitzer1983Parameters]:
        """
        Get the polarization parameters of the different species.
        Follows Section 2.1.

        Parameters
        ----------
        ocs_abspol_from
            Define which model to use to compute the absorption and polarization
            profiles of OCS.

            - ``"duan"``: Using a Ben-Reuven line shape derived from SO2 (default)
            - ``"kolbe"``: Using a Lorentzian line shape as described in the paper
              and following :cite:t:`kolbe1977`
            - ``"bbld"``: Using a Ben-Reuven line shape with parameters derived
              approximately from :cite:t:`bouanich1988` and :cite:t:`lavrentieva2020`.

        use_eps_prime_r_inf
            If ``True``, when computing the real part of the relative permittivity
            of SO2 and OCS, a value of the real relative permittivity at infinite
            frequency is set to an assumed value (rather than using the theoretical
            value of unity).
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
        # AR
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
            VISAR_FREQUENCY,
            eps_prime_r_inf=(
                Duan2010.EPS_PRIME_R_INF_SO2 if use_eps_prime_r_inf else 1.0
            ),
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
        Pnu_H2SO4 = Duan2010.eq3(kolodnersteffes1998.eps_prime_r_h2so4)
        if use_virial_approximation:
            # get virial expansion terms
            A_mu_H2SO4 = float(
                HarveyLemmon2005Parameters.get_A_mu(kolodnersteffes1998.MU_H2SO4)
            )
            A_epsilon_H2SO4 = float(
                Duan2010.A_epsilon_from_eq8(
                    Pnu_H2SO4,
                    A_mu_H2SO4,
                    kolodnersteffes1998.rho_h2so4,
                    kolodnersteffes1998.T_H2SO4,
                )
            )
            # define Harvey & Lemmon parameter set
            polarization_parameters["H2SO4"] = HarveyLemmon2005Parameters(
                a0=A_epsilon_H2SO4, A_mu=A_mu_H2SO4
            )
        else:
            # get molecular polarizability
            alpha_T_H2SO4 = Duan2010.alpha_T_from_eq14(
                kolodnersteffes1998.rho_h2so4,
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
        match ocs_abspol_from:
            case "kolbe":
                eps_prime_r_ocs = Duan2010.eps_prime_r_from_spectral_lines(
                    Duan2010.T_OCS,
                    Duan2010.P_OCS,
                    jplspectrallines.tables["OCS"],
                    Duan2010.L_OCS,
                    VISAR_FREQUENCY,
                    use_ben_reuven=False,
                    eps_prime_r_inf=(
                        Duan2010.EPS_PRIME_R_INF_OCS if use_eps_prime_r_inf else 1.0
                    ),
                )
            case "duan":
                eps_prime_r_ocs = Duan2010.eps_prime_r_from_spectral_lines(
                    Duan2010.T_OCS,
                    Duan2010.P_OCS,
                    jplspectrallines.tables["OCS"],
                    Duan2010.BR_SO2_AS_OCS_CO2,
                    VISAR_FREQUENCY,
                    eps_prime_r_inf=(
                        Duan2010.EPS_PRIME_R_INF_OCS if use_eps_prime_r_inf else 1.0
                    ),
                )
            case "bbld":
                eps_prime_r_ocs = Duan2010.eps_prime_r_from_spectral_lines(
                    Duan2010.T_OCS,
                    Duan2010.P_OCS,
                    jplspectrallines.tables["OCS"],
                    Duan2010.BR_OCS_CO2,
                    VISAR_FREQUENCY,
                    eps_prime_r_inf=(
                        Duan2010.EPS_PRIME_R_INF_OCS if use_eps_prime_r_inf else 1.0
                    ),
                )
            case _:
                raise ValueError(f"Unknown OCS model {ocs_abspol_from=}")
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
        self, use_clouds_from: str = "cimino", use_cimino_fitted_lookup: bool = False
    ) -> Tuple[Quantity["dimensionless"], Quantity["wavenumber"]]:
        """
        Evaluate the cloud polarization and absorption given the model's
        atmospheric quantities. Follows Sections 2.1.5 and 2.2.5,
        and/or :cite:t:`cimino1982`.

        Parameters
        ----------
        use_clouds_from
            Define which cloud polarization and absorption model to use:

            - ``"cimino"``: :cite:t:`cimino1982`, eq. (10) and (16)
            - ``"duan"``: :cite:t:`duan2010`, sections 2.1.5 and 2.2.5
            - ``"none"``: Ignore all cloud effects

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

        Returns
        -------
        cloud_pol
            Polarization of the cloud (accounting for its volume fraction)
        cloud_absorp
            Absorption of the cloud (accounting for its volume fraction)
        """
        # early return if we ignore clouds
        if use_clouds_from == "none":
            return Quantity(
                np.zeros(self.altitude.size), u.dimensionless_unscaled
            ), Quantity(np.zeros(self.altitude.size), "1/cm")
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
        match use_clouds_from:
            case "cimino":
                # compute complex polarization of droplets and volume fraction
                # from the Cimino paper
                cloud_Pnu, vol_frac_droplets = (
                    cimino1982.get_h2so4_droplet_polarization_volfrac(
                        self.cloud_mass_density,
                        eps_prime_r_H2SO4_H2O - 1j * eps_dprime_r_H2SO4_H2O,
                    )
                )
                # save polarization
                cloud_pol = cloud_Pnu.real * vol_frac_droplets
                # convert the relative permittivity to absorption
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
            case "duan":
                # follow section 2.1.5
                # we look up (i.e., interpolate) to get the density
                # of the concentrated droplets
                d_concentr_H2SO4 = Quantity(
                    np.interp(
                        self.cloud_concentration.to_value("%"),
                        duan2010figures.tables["4"]["Weight Percentage"].to("%").value,
                        duan2010figures.tables["4"]["Density"].value,
                        left=np.nan,
                        right=np.nan,
                    ),
                    duan2010figures.tables["4"]["Density"].unit,
                )
                # calculate the spreading ratio
                # (inverse actually to avoid divide by zero)
                eta_s_inv = (self.cloud_mass_density / d_concentr_H2SO4).decompose()
                # convert it to polarization
                P_concentr_H2SO4_H2O = Duan2010.eq3(eps_prime_r_H2SO4_H2O)
                # and finally calculate the polarization of the distributed solution
                P_distr_H2SO4_H2O = P_concentr_H2SO4_H2O * eta_s_inv
                # since eta_s is the inverse of the volume fraction, the computed
                # polarization already accounts for its density in the atmosphere
                cloud_pol = P_distr_H2SO4_H2O
                # convert the relative permittivity to absorption
                # follow section 2.2.5
                cloud_absorp = (
                    Duan2010.eq25(eps_prime_r_H2SO4_H2O, eps_dprime_r_H2SO4_H2O)
                    * eta_s_inv
                )
        # done
        return np.nan_to_num(cloud_pol), np.nan_to_num(cloud_absorp)

    def evaluate_absorptions(
        self,
        cutoff_so2_frequency: Quantity["frequency"] | None = None,
        ocs_abspol_from: str = "duan",
    ) -> astrotable.QTable:
        """
        Evaluate the absorption models given the model's atmospheric quantities.

        Parameters
        ----------
        cutoff_so2_frequency
            When computing the absorption coefficient of SO2, include all spectral
            lines up to this frequency. If ``None``, use all available ones.
            This option is only kept for development purposes.
        ocs_abspol_from
            Define which model to use to compute the absorption and polarization
            profiles of OCS.

            - ``"duan"``: Using a Ben-Reuven line shape derived from SO2 (default)
            - ``"kolbe"``: Using a Lorentzian line shape as described in the paper
              and following :cite:t:`kolbe1977`
            - ``"bbld"``: Using a Ben-Reuven line shape with parameters derived
              approximately from :cite:t:`bouanich1988` and :cite:t:`lavrentieva2020`.


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
        absorptions["SO2"] = Duan2010.absorption_ben_reuven(
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
        match ocs_abspol_from:
            case "kolbe":
                absorptions["OCS"] = Duan2010.absorption_lorentz(
                    self.temperature,
                    self.molar_fractions["OCS"] * self.pressure,
                    jplspectrallines.tables["OCS"],
                    VISAR_FREQUENCY,
                    Duan2010.L_OCS,
                ).squeeze()
            case "duan":
                absorptions["OCS"] = Duan2010.absorption_ben_reuven(
                    self.temperature,
                    self.molar_fractions["OCS"] * self.pressure,
                    self.molar_fractions["CO2"] * self.pressure,
                    jplspectrallines.tables["OCS"],
                    VISAR_FREQUENCY,
                    Duan2010.BR_SO2_AS_OCS_CO2,
                ).squeeze()
            case "bbld":
                absorptions["OCS"] = Duan2010.absorption_ben_reuven(
                    self.temperature,
                    self.molar_fractions["OCS"] * self.pressure,
                    self.molar_fractions["CO2"] * self.pressure,
                    jplspectrallines.tables["OCS"],
                    VISAR_FREQUENCY,
                    Duan2010.BR_OCS_CO2,
                ).squeeze()
            case _:
                raise ValueError(f"Unknown OCS model {ocs_abspol_from=}")

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
        (as described by :cite:t:`harvey2005`, eq. 5) using the polarization
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
    def absorption_ben_reuven(
        T: Quantity["temperature"],
        P_minor: Quantity["pressure"],
        P_major: Quantity["pressure"],
        spectral_lines: astrotable.QTable,
        nu: Quantity["frequency"],
        ls_params: LineShapeParameters,
    ) -> Quantity:
        """
        Calculates the absorption by summing contributions from a spectral line catalog
        and using Ben-Reuven line broadening coefficients as described in
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
        ls_params
            Line shape parameters for the Ben-Reuven expression

        Returns
        -------
            Total absorption [1/cm]
        """
        # prepare input
        T = np.atleast_1d(T)
        P_minor = np.atleast_1d(P_minor)
        P_major = np.atleast_1d(P_major)
        nu = np.atleast_1d(nu)
        # prepare output
        alpha = np.full((T.size, nu.size), np.nan)
        assert T.shape == P_minor.shape == P_major.shape
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
        nu = nu.to("MHz").value[None, None, :]
        # get Ben-Reuven parameters in correct units
        T_0 = ls_params.T_0.to("K").value
        gamma_min_maj = ls_params.gamma_min_maj.to("MHz/torr").value
        gamma_min_min = ls_params.gamma_min_min.to("MHz/torr").value
        zeta_min_maj = ls_params.zeta_min_maj.to("MHz/torr").value
        zeta_min_min = ls_params.zeta_min_min.to("MHz/torr").value
        delta_min = ls_params.delta_min.to("MHz/torr").value
        m = ls_params.m
        n = ls_params.n
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
    def absorption_lorentz(
        T: Quantity["temperature"],
        P: Quantity["pressure"],
        spectral_lines: astrotable.QTable,
        nu: Quantity["frequency"],
        ls_params: LineShapeParameters,
    ) -> Quantity:
        """
        Calculates the absorption by summing contributions from a spectral line catalog
        as described in eqs. (27-32) on pp. 10f but using Lorentzian line broadening
        coefficients.

        Parameters
        ----------
        T
            Temperature [K]
        P
            Partial pressure [torr]
        spectral_lines
            Spectral line catalog for the species containing line
            frequencies nu [MHz], line center intensities I [nm^2 MHz],
            and lower state energies El [1/cm]
        nu
            Target frequency of the absorption [Hz]
        ls_params
            Line shape parameters; only `gamma_min_min` is used as the line width

        Returns
        -------
            Total absorption [1/cm]
        """
        # prepare input
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)
        nu = np.atleast_1d(nu)
        # prepare output
        alpha = np.full((T.size, nu.size), np.nan)
        assert T.shape == P.shape
        data_valid = P > 0
        # convert all input quantities to unit-defined NumPy arrays
        # so we can make effective use of broadcasting
        # first axis: atmospheric parameters
        T = T.to("K").value[data_valid, None, None]
        P = P.to("torr").value[data_valid, None, None]
        # second axis: spectral line catalog
        nu_0 = spectral_lines["FREQ"].to("MHz").value[None, :, None]
        I = spectral_lines["LGINT"].physical.to("nm2 MHz").value[None, :, None]
        El = spectral_lines["ELO"].to("1/cm").value[None, :, None]
        # third axis: frequencies to compute results for
        nu = nu.to("MHz").value[None, None, :]
        # get line widths at frequencies
        T0 = ls_params.T_0.to("K").value
        gamma = ls_params.gamma_min_min.to("MHz/torr").value * P
        # Lorentzian line shape function [1/MHz]
        F_L = gamma / (np.pi * ((nu_0 - nu) ** 2 + gamma**2))
        # eq. (28) [1/cm]
        alpha_max = Quantity(
            102.46
            * P
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
        Calculate the total absorption of H2SO4 given eq. (33) on p. 11 in
        :cite:t:`duan2010`, which in turn is eq. (18) in :cite:t:`kolodner1998`.

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
        ls_params: LineShapeParameters,
        nu: Quantity["frequency"],
        freqstep: Quantity["frequency"] = Quantity(0.1, "GHz"),
        freqmin: Quantity["frequency"] | None = None,
        freqmax: Quantity["frequency"] | None = None,
        use_ben_reuven: bool = True,
        eps_prime_r_inf: float = 1.0,
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
        ls_params
            Parameters for the Ben-Reuven line expression
        nu
            Target frequency of the absorption [Hz]
        freqstep
            Frequency step of the integration domain
        freqmin
            Minimum frequency of the integration domain
            (defaults to minimum frequency of ``spectral_lines``).
            Below that, five log-spaced samples at lower orders of magnitude are
            added for numerical stability
        freqmax
            Maximum frequency of the densely-sampled integration domain
            (defaults to maximum frequency of ``spectral_lines``).
            Above that, five log-spaced samples at higher orders of magnitude are
            added for numerical stability.
        use_ben_reuven
            If ``True``, use the Ben-Reuven line expression, else use a Lorentzian
            line shape for the computation of the absorption.
        eps_prime_r_inf
            Real part of the relative permittivity at infinite frequency,
            theoretically ``1``.

        Returns
        -------
            Real part of the relative permittivity
        """
        # get integration domain
        if freqmin is None:
            freqmin = np.min(spectral_lines["FREQ"])
        freqmin = min(freqmin, freqstep)
        if freqmax is None:
            freqmax = np.max(spectral_lines["FREQ"])
        freqmin_Hz = freqmin.to("Hz").value
        freqmax_Hz = freqmax.to("Hz").value
        freqstep_Hz = freqstep.to("Hz").value
        freqsing_Hz = nu.to("Hz").value
        freqrange_raw = np.arange(freqmin_Hz, freqmax_Hz, freqstep_Hz)
        # offset the frequency range to have the singularity exactly between two samples
        freqrange_raw += (freqsing_Hz - freqmin_Hz + freqstep_Hz / 2) % freqstep_Hz
        assert freqrange_raw[0] <= freqsing_Hz <= freqrange_raw[-1]
        # create samples close to the singularity
        singtol_Hz = 1e-5 * (freqstep_Hz / 2)
        singfreqs_raw = np.geomspace(singtol_Hz, freqstep_Hz / 2, num=5, endpoint=False)
        insert_middle = np.r_[
            freqsing_Hz - singfreqs_raw[::-1],
            freqsing_Hz + singfreqs_raw,
        ]
        # create samples before and after the main frequency range
        insert_before = np.geomspace(
            1e-5 * freqrange_raw[0],
            freqrange_raw[0],
            num=5,
            endpoint=False,
        )
        insert_after = np.geomspace(
            1e1 * freqrange_raw[-1],
            1e5 * freqrange_raw[-1],
            num=5,
            endpoint=True,
        )
        # combine the different frequency ranges
        i_aftersing = np.argmax(freqrange_raw > freqsing_Hz)
        extrange = np.r_[
            insert_before,
            freqrange_raw[:i_aftersing],
            insert_middle,
            freqrange_raw[i_aftersing:],
            insert_after,
        ]
        freqrange = Quantity(extrange, "Hz")
        # compute absorption coefficient
        if use_ben_reuven:
            alpha = Duan2010.absorption_ben_reuven(
                np.atleast_1d(T),
                np.atleast_1d(P),
                Quantity([0], "atm"),
                spectral_lines,
                freqrange,
                ls_params,
            ).squeeze()
        else:
            alpha = Duan2010.absorption_lorentz(
                np.atleast_1d(T),
                np.atleast_1d(P),
                spectral_lines,
                freqrange,
                ls_params,
            ).squeeze()
        # convert from absorption coefficient to imaginary part of the
        # relative permittivity
        eps_dprime = (
            alpha / freqrange.to("1/cm", equivalencies=u.spectral()) / (2 * np.pi)
        ).decompose()
        # use Kramers-Krönig equation to compute the real part of the relative
        # permittivity from the imaginary part
        freqsqdiff = freqrange**2 - nu**2
        integrand = (freqrange * eps_dprime / freqsqdiff).to("1/GHz").value
        eps_prime_r = (
            eps_prime_r_inf
            + 2 * np.trapezoid(integrand, x=freqrange.to("GHz").value) / np.pi
        )
        # done
        return Quantity(eps_prime_r, u.dimensionless_unscaled)


class Duan2010Verification(Duan2010):
    """
    The same as :class:`~Duan2010` except that the defaults follow the Matlab
    ``config.atm_recipe = 'model_verification'`` setting.
    """

    def __init__(
        self,
        profile_TPD="seiff:75",
        profile_CO2=zahnmoroz1985.co2_molar_fraction,
        profile_N2=zahnmoroz1985.n2_molar_fraction,
        profile_H2O=duan2010figures.h2o_old_molar_fraction,
        profile_SO2=duan2010figures.so2_old_molar_fraction,
        profile_CO=duan2010figures.co_old_molar_fraction,
        profile_H2SO4=kolodnersteffes1998.h2so4_mr_3212,
        profile_OCS=duan2010figures.ocs_old_molar_fraction,
        profile_Ar=None,
        use_clouds_from="none",
        ocs_abspol_from="duan",
        use_eps_prime_r_inf=True,
        load_polarization_parameters=True,
        use_compressible_gas=True,
        use_keating_temp_press_above100km=False,
        use_virial_approximation=True,
        cutoff_so2_frequency=None,
        use_cimino_fitted_lookup=False,
    ):
        super().__init__(
            profile_TPD,
            profile_CO2,
            profile_N2,
            profile_H2O,
            profile_SO2,
            profile_CO,
            profile_H2SO4,
            profile_OCS,
            profile_Ar,
            use_clouds_from,
            ocs_abspol_from,
            use_eps_prime_r_inf,
            load_polarization_parameters,
            use_compressible_gas,
            use_keating_temp_press_above100km,
            use_virial_approximation,
            cutoff_so2_frequency,
            use_cimino_fitted_lookup,
        )
