"""
Module that provides the background, reference atmospheric properties.
"""

# standard imports
import numpy as np
from astropy.units import Quantity, Unit

# package imports
from ..constants import GAS_CONSTANT
from ..geometry import get_sza
from ..utils.interpolate import BoundedInterpolatingBasis


class SeiffKeating:

    UNITS = [Unit("km"), Unit("K"), Unit("bar"), Unit("kg/m3")]
    """ Units of the returned profiles """

    def __init__(self, seiff: Seiff1985, keating: Keating1985):
        """
        Class that combines the temperature, pressure, and density profiles from the
        two references :cite:t:`seiff1985`, which contains latitudinal variations,
        and :cite:t:`keating1985`, which contains variations with solar zenith angle.

        Parameters
        ----------
        seiff
            Initialized :class:`~xvamp.reference.Seiff1985` model
        keating
            Initialized :class:`~xvamp.reference.Keating1985` model
        """
        # save models
        self.seiff = seiff
        self.keating = keating
        # initialize individual interpolators
        self.interp_33km_100km = BoundedInterpolatingBasis(
            lower=0, upper=90, knots=self.seiff.LAT.to("°").value
        )
        self.interp_100km_150km = BoundedInterpolatingBasis(
            lower=0, upper=180, knots=np.array(self.keating.SZA[[0, -1]].to("°").value)
        )
        self.interp_150km_250km = BoundedInterpolatingBasis(
            lower=0, upper=180, knots=self.keating.SZA.to("°").value
        )
        # done

    def __call__(
        self,
        latitude: Quantity,
        localtime: Quantity,
        add_3K: bool = False,
    ):
        """
        Interpolate the profiles from :cite:t:`seiff1985` and :cite:t:`keating1985`.

        Parameters
        ----------
        latitude
            Latitude [°] of desired profiles
        localtime
            Local solar time [h] of desired profiles
        add_3K
            This refers to the 3 K addition done in the :cite:t:`Duan2010` model
            when combining the :cite:t:`seiff1985` and :cite:t:`zasova2006` profiles.
            If ``True``, the 3 K are added to all latitudes, if ``False``, to none of
            them.

        Returns
        -------
        sza
            Solar zenith angle [°] of profiles
        profile
            Temperature, pressure, and density profiles
        """
        # get solar zenith angle
        sza = (
            np.rad2deg(
                get_sza(
                    localtime.to("h").value[:, None], latitude.to("rad").value[None, :]
                )
            )
            % 360
        )
        # get input table 0-32 km
        ref_0km_32km = (
            self.seiff.tables["1-1"].as_array().view((float, 7))[:, :4, None, None]
        )
        # adjust temperature if desired
        if add_3K:
            ref_0km_32km[:, 1] += 3
        # compute data at reference altitudes
        # 33-100 km
        basis_33km_100km = self.interp_33km_100km(np.abs(latitude.to("°").value))
        ref_33km_100km = np.einsum(
            "ijk,lk->ijl", self.seiff.dcube_33km_100km, basis_33km_100km
        )
        # adjust temperature if desired
        if add_3K:
            ref_33km_100km[:, 1, :] += 3
        # 100-150 km
        basis_100km_150km = self.interp_100km_150km(sza.ravel())
        ref_100km_150km = np.einsum(
            "ijk,lk->ijl",
            self.keating.dcube_100km_150km[:, [0, 2, 11, 1], :],
            basis_100km_150km,
        )
        # convert density from [g/cm3] to [kg/m3]
        ref_100km_150km[:, 3] *= 1e3
        # convert pressure from [mbar] to [bar]
        ref_100km_150km[:, 2] /= 1e3
        # 150-250 km
        basis_150km_250km = self.interp_150km_250km(sza.ravel())
        ref_150km_250km = np.einsum(
            "ijk,lk->ijl",
            self.keating.dcube_150km_250km[:, [0, 2, 10, 1], :],
            basis_150km_250km,
        )
        # convert mean molecular weight column into pressure [bar] column
        ref_150km_250km[:, 2] = (
            ref_150km_250km[:, 3]  # [g / cm3]
            * GAS_CONSTANT  # [J / K mol]
            * ref_150km_250km[:, 1]  # [K]
            / ref_150km_250km[:, 2]  # [g / mol]
        ) * 10
        # convert density from [g/cm3] to [kg/m3]
        ref_150km_250km[:, 3] *= 1e3
        # reshape
        ref_0km_32km = np.broadcast_to(
            ref_0km_32km,
            (len(self.seiff.tables["1-1"]), 4, localtime.size, latitude.size),
        )
        ref_33km_100km = np.broadcast_to(
            ref_33km_100km[:, :, None, :],
            (ref_33km_100km.shape[0], 4, localtime.size, latitude.size),
        )
        ref_100km_150km = ref_100km_150km.reshape(
            ref_100km_150km.shape[0], 4, localtime.size, latitude.size
        )
        ref_150km_250km = ref_150km_250km.reshape(
            ref_150km_250km.shape[0], 4, localtime.size, latitude.size
        )
        # combine
        profile = np.concatenate(
            [
                ref_0km_32km,
                ref_33km_100km,
                ref_100km_150km[-2:0:-1, :, :, :],
                ref_150km_250km,
            ],
            axis=0,
        )
        # done
        return Quantity(sza, "°"), profile
