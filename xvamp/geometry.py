"""
Provides functions around the satellite's viewing geometry.
"""

# standard imports
import astropy.units as u
import numpy as np
from typing import Tuple
from scipy.integrate import cumulative_trapezoid
from astropy.units import Quantity, Unit, UnitConversionError

# package imports
from .utils import float_or_array
from .constants import VENUS_RADIUS


def get_sza(lst: float_or_array, lat: float_or_array) -> float_or_array:
    """
    Calculate the solar zenith angle.

    Parameters
    ----------
    lst
        Local solar time [hours]
    lat
        Latitude [rad]

    Returns
    -------
        Solar zenith angle [rad]

    Note
    ----
    Formula is from :cite:t:`keating1985`, eq.(8).
    Assumes that the orbital tilt is approximately 0°.
    """
    return np.arccos(np.cos((lst - 12) / 24 * 2 * np.pi) * np.cos(lat))


def _make_broadcastable_input(
    *inputs: Quantity | float_or_array,
) -> Tuple[Quantity | float_or_array, ...]:
    """
    Parse all inputs to a commons length.

    Parameters
    ----------
    *args
        Input floats, arrays, or Quantities.

    Returns
    -------
        All inputs but cast to 1D shape

    Raises
    ------
    ValueError
        If the inputs cannot be cast to the same length, or an input is more than
        one-dimensional
    """
    # cast everything to 1D
    out = tuple(np.atleast_1d(inp) for inp in inputs)
    # check dimensions
    for i, o in enumerate(out):
        if o.ndim != 1:
            raise ValueError(f"Argument #{i} is {o.ndim}-dimensional")
    # get common size
    nout = max(o.size for o in out)
    # check if all inputs are either scalars or nout long
    for i, o in enumerate(out):
        if o.size not in [1, nout]:
            raise ValueError(f"Argument #{i} is {o.size} long, expected 1 or {nout}")
    # done
    return out


def height_to_radius(
    height: Quantity["length"] | float_or_array,
    planet_radius: Quantity["length"] | float_or_array = VENUS_RADIUS,
    unit: Unit = u.km,
) -> Quantity["length"]:
    """
    Add a planetary radius to a height and return the radius as a
    :class:`astropy.unit.Quantity`.

    Parameters
    ----------
    height
        Height to be converted to a radius.
        If not a Quantity, the ``unit`` keyword must match.
    planet_radius
        Planet radius to use.
        If not a Quantity, the ``unit`` keyword must match.
    unit
        :class:`~astropy.units.Unit` of the output.

    Returns
    -------
        Equivalent radius of the height input
    """
    try:
        return Quantity(planet_radius + height, unit)
    except UnitConversionError:
        return planet_radius + Quantity(height, unit)


def geometry_from_central_angle(
    central_angle: Quantity["angle"] | float_or_array,
    height_terrain: Quantity["length"] | float_or_array,
    height_platform: Quantity["length"] | float_or_array,
) -> Tuple[Quantity["length"], Quantity["angle"], Quantity["angle"]]:
    """
    Calculate the geometric (in vacuum) range, look angle, and incidence angle
    from the central angle and the law of cosines.

    Parameters
    ----------
    central_angle
        Central angle(s) of the platform in [rad], if not a
        :class:`~astropy.units.Quantity`.
    height_terrain
        Height(s) of the terrain relative to the mean planet radius in [km],
        if not a :class:`~astropy.units.Quantity`.
    height_platform
        Height(s) of the platform relative to the mean planet radius in [km],
        if not a :class:`~astropy.units.Quantity`.

    Returns
    -------
    geometric_range
        Geometric range from the platform to the surface [km].
    geometric_look_angle
        Geometric look angle from the platform to the surface [rad].
    geometric_incidence_angle
        Geometric incidence angle from the surface to the platform [rad].
    """
    # input check
    central_angle, height_terrain, height_platform = _make_broadcastable_input(
        central_angle, height_terrain, height_platform
    )
    # parse input to common units
    if not isinstance(central_angle, Quantity):
        central_angle = Quantity(central_angle, "rad")
    rp = height_to_radius(height_platform)
    rt = height_to_radius(height_terrain)
    # avoid duplicate operations
    rp_sq = rp**2
    rt_sq = rt**2
    # get geometric range
    geometric_range = np.sqrt(rp_sq + rt_sq - 2 * rp * rt * np.cos(central_angle))
    r_sq = geometric_range**2
    # get geometric look angle
    geometric_look_angle = np.arccos(
        (rp_sq + r_sq - rt_sq) / (2 * rp * geometric_range)
    )
    # get geometric incidence angle
    geometric_incidence_angle = Quantity(np.pi, "rad") - np.arccos(
        (rt_sq + r_sq - rp_sq) / (2 * rt * geometric_range)
    )
    # done
    return geometric_range, geometric_look_angle, geometric_incidence_angle


def get_cross_track_displacement(
    look_angle: Quantity["angle"] | float_or_array,
    central_angle: Quantity["angle"] | float_or_array,
    height_terrain: Quantity["length"] | float_or_array,
    height_platform: Quantity["length"] | float_or_array,
) -> Quantity["length"]:
    """
    Calculate the cross-track displacement between a given central angle and one
    derived from a look angle assuming vacuum propagation.

    Parameters
    ----------
    look_angle
        Look angle(s) of the instrument in [rad], if not a
        :class:`~astropy.units.Quantity`
    central_angle
        Central angle(s) of the platform in [rad], if not a
        :class:`~astropy.units.Quantity`.
    height_terrain
        Height(s) of the terrain relative to the mean planet radius in [km],
        if not a :class:`~astropy.units.Quantity`.
    height_platform
        Height(s) of the platform relative to the mean planet radius in [km],
        if not a :class:`~astropy.units.Quantity`.

    Returns
    -------
    cross_track_displacement
        Distance along the spherical planet between the true point on the ground and
        the point if the look angle went through vacuum
    """
    # input check
    look_angle, central_angle, height_terrain, height_platform = (
        _make_broadcastable_input(
            look_angle, central_angle, height_terrain, height_platform
        )
    )
    # parse input to common units
    if not isinstance(look_angle, Quantity):
        look_angle = Quantity(look_angle, "rad")
    if not isinstance(central_angle, Quantity):
        central_angle = Quantity(central_angle, "rad")
    rp = height_to_radius(height_platform)
    rt = height_to_radius(height_terrain)
    # avoid duplicate operations
    rp_sq = rp**2
    rt_sq = rt**2
    # get cross-track displacement
    vacuum_range = rp * np.cos(look_angle) - np.sqrt(
        rt_sq - (rp * np.sin(look_angle)) ** 2
    )
    vacuum_central_angle = np.arccos((rp_sq + rt_sq - vacuum_range**2) / (2 * rp * rt))
    cross_track_displacement = (
        (vacuum_central_angle - central_angle) / Quantity(1, "rad") * VENUS_RADIUS
    )
    # done
    return cross_track_displacement


def get_brightness_temperature(
    altitude: Quantity["length"],
    temperature: Quantity["temperature"],
    refraction: Quantity["dimensionless"],
    absorption: Quantity["dB/km"],
    look_angle: Quantity["angle"],
    surface_brightness: Quantity["temperature"],
) -> Quantity["temperature"]:
    """
    Calculate the brightness temperature of the planet from an assumed surface
    brightness as well as the temperature, refraction, and absorption profiles
    (across their entirety, i.e., assuming that the surface is the first and the
    platform is the last altitude array element, respectively).
    Assumes a scatter-free medium and follows Section 6-6.4 of :cite:t:`ulaby2013`.

    Parameters
    ----------
    altitude
        Altitude values of the profiles
    temperature
        Temperature profile
    refraction
        Refraction profile
    absorption
        Absorption coefficient profile
    look_angle
        Observer angle
    surface_brightness
        Assumed surface brightness temperature

    Returns
    -------
        Brightness temperature
    """
    # force units for straightforward numerical integration later
    altitude_km = altitude.to("km").value
    height_platform = altitude_km[-1]
    T_K = temperature.to("K").value
    refractions = refraction.value
    refraction_0 = refractions[-1]
    absorption_Np_km = absorption.to("Np/km").value
    venus_radius = VENUS_RADIUS.to("km").value
    look_angle_rad = look_angle.to("rad").value
    surface_brightness_K = surface_brightness.to("K").value
    # use Snell's law for a spherical shell to compute the secant of the
    # look angle for all platform and evaluation altitude layers
    sine_look_angle = (
        (venus_radius + height_platform)
        / (venus_radius + altitude_km)
        * (refraction_0 / refractions)
        * np.sin(look_angle_rad)
    )
    secant_look_angle = 1 / np.sqrt(1 - sine_look_angle**2)
    # get an expression for the differential optical thickness
    # for each layer of the spherical atmosphere.
    # start from eq. (6.61) setting κ_e = κ_a (scatter-free assumption)
    # and noting κ_a = 2α, eq. (6.92)
    dtau_dz = 2 * absorption_Np_km * secant_look_angle
    # note that this expression uses the geometric curved path length
    # drho = sec(θ) dz rather than the optical curved path length
    # droh = n(z) sec(θ) dz to be consistent with Ulaby & Long 2013.
    # cumulatively integrate to get an expression for tau(0, z')
    # as needed in eq. (6.73)
    tau = cumulative_trapezoid(dtau_dz, x=altitude_km, initial=0)
    # rearrange to get the expression for tau(z', height_platform)
    tau = tau[-1] - tau
    # get upward emission contribution, second term in eq. (6.73)
    T_up = np.trapezoid(T_K * np.exp(-tau) * dtau_dz, x=altitude_km)
    # combine everything
    T_brightness = surface_brightness_K * np.exp(-tau[0]) + T_up
    # done
    return Quantity(T_brightness, "K")
