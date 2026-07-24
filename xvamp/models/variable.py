"""
Module containing the model with variable profiles.
"""

# standard imports
import numpy as np
from astropy.units import Quantity
from numpy.typing import NDArray

# package imports
from ..reference import paetzold2007, seiffkeating


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
        use_tpd_from: str = "duan",
        use_compressible_gas: bool = True,
        use_keating_temp_press_above100km: bool = False,
        use_keating_co_co2_n2_above_100km: bool = False,
        use_simple_h2o: bool = False,
        use_simple_so2: bool = False,
        use_simple_co: bool = False,
        use_h2so4_from: str = "duan",
        use_ocs_from: str = "duan",
        add_ar: bool = False,
        cutoff_so2_frequency: Quantity["frequency"] | None = None,
        ocs_abspol_from: str = "duan",
        use_eps_prime_r_inf: bool = True,
        use_virial_approximation: bool = True,
        use_clouds_from: str = "cimino",
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
        self._ocs_abspol_from = ocs_abspol_from
        self._use_cimino_clouds = use_clouds_from
        self._use_cimino_fitted_lookup = use_cimino_fitted_lookup
        # call parent initializer
        super().__init__(
            use_tpd_from,
            use_compressible_gas,
            use_keating_temp_press_above100km,
            use_keating_co_co2_n2_above_100km,
            use_simple_h2o,
            use_simple_so2,
            use_simple_co,
            use_h2so4_from,
            use_ocs_from,
            add_ar,
            cutoff_so2_frequency,
            ocs_abspol_from,
            use_eps_prime_r_inf,
            use_virial_approximation,
            use_clouds_from,
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
        Overwrites :attr:`~xvamp.models.model.Model.electron_density`, then calls
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
        Overwrites :attr:`~xvamp.models.model.Model.altitude`
        and :attr:`~xvamp.models.model.Model.temperature`,
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
            Duan2010.VENUS_GAS_CONSTANT, self._add3K
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
            ocs_abspol_from=self._ocs_abspol_from,
            use_clouds_from=self._use_cimino_clouds,
            use_cimino_fitted_lookup=self._use_cimino_fitted_lookup,
        )
        self.update_from_ionosphere()
        # done
