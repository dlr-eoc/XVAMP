"""
Utility module for the atmospheric model.
"""

# standard imports
from typing import Any
from datetime import datetime
from importlib.resources import files as res_files
import astropy.units as u
import numpy as np
import tomlkit as tok
from typing import Tuple
from dataclasses import dataclass, field
from pathlib import Path
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.integrate import cumulative_trapezoid
from astropy.units import Quantity, UnitConversionError
from astropy.table import Table, QTable

# package imports
from . import data
from .constants import *

# type shorthands
float_or_array = float | NDArray[np.double]
""" Either a single float number or an array of float numbers """
complex_or_array = complex | NDArray[np.cdouble]
""" Either a single complex number or an array of complex numbers """


def read_unit_csv(path: Path) -> QTable:
    """
    Given a path to a ``.csv`` file containing column names of the format
    "Name [Unit]", read the data and return a :class:`~astropy.table.QTable`
    with the units correctly set.

    Parameters
    ----------
    path
        Path to the file

    Returns
    -------
        Table with quantities and units
    """
    # read file
    table = Table.read(path, converters={"*": float}, encoding="utf8")
    # extract column names and units
    matches = HEADERPATTERN.findall("\n".join(table.colnames))
    colnames, unitstr = zip(*matches)
    units = [u.Unit(s) for s in unitstr]
    if len(colnames) != len(table.columns):
        raise ValueError(f"Only found the following headers:\n{colnames}\n{units}")
    # convert Table to QTable
    return QTable(table, names=colnames, units=units)


def read_unit_fwf(
    path: Path,
    names: list[str],
    formats: list[str],
    widths: list[int],
    units: list[str],
    converters: dict = {},
) -> QTable:
    """
    Given a path to a fixed-width text file and lists describing the column
    names, widths, and units, return a :class:`~astropy.table.QTable` matching
    data with units.

    Parameters
    ----------
    path
        Path to the file
    names
        List of column names
    formats
        NumPy-readable format strings for each column
    widths
        List of column widths
    units
        List of astropy-readable unit strings
    converters
        Column data converter functions passed on to :func:`~numpy.genfromtxt`

    Returns
    -------
        Table with quantities and units
    """
    # read data
    data = np.genfromtxt(
        path, dtype=formats, delimiter=widths, converters=converters, encoding="utf8"
    )
    # build QTable
    return QTable(data, names=names, units=units)


@dataclass
class HarveyLemmon2005Parameters:
    """
    Parameters for mixture components from :cite:t:`harvey2005`,
    as represented in :cite:t:`duan2010`, Table 1 for eq. (8).
    Parameters are NOT converted to astropy :class:`~astropy.units.Quantity`
    because of the unknown exponent.
    """

    a0: float = 0
    """ [cm^3/mol] """
    a1: float = 0
    """ [cm^3/mol] """
    b0: float = 0
    """ [cm^6/mol^2] """
    b1: float = 0
    """ [cm^6/mol^2] """
    c0: float = 0
    """ [cm^(3(D+1))/mol^-(D+1)] """
    c1: float = 0
    """ [cm^(3(D+1))/mol^-(D+1)] """
    D: float = 0
    """ [-] """
    T0: float = 273.16
    """ Temperature [K] """
    A_mu: float = 0
    """ Dipolar term in the virial expansion [cm^3 K/mol] """

    @staticmethod
    def get_A_mu(mu: Quantity) -> float_or_array:
        """
        Compute the dipolar term in the dielectric virial expansion,
        assuming CGS units in the input, but SI in the output.

        Parameters
        ----------
        mu
            Permanent dipole moment [esu cm]

        Returns
        -------
            Dipolar term in the virial expansion [cm^3 K/mol]
        """
        return ((4 * np.pi * AVOGADRO * mu**2) / (9 * BOLTZMANN)).to("cm3 K/mol").value


@dataclass
class Pitzer1983Parameters:
    """
    Parameters for the :cite:t:`pitzer1983` model to calculate the polarization
    per molar volume as given by :cite:t:`duan2010` on p. 5, eq. (14).
    """

    mu: Quantity[ESU_CM]
    """ Molecular dipole moment [esu cm = 1e18 D] """
    alpha_T: Quantity["cm3"]
    """ Molecular polarizability [cm^3] """


@dataclass
class LineShapeParameters:
    """
    Line shape parameters compatible with the Ben-Reuven line shape function, following
    the notation from :cite:t:`duan2010`, eqs. (27-32) on p. 10f.
    Can be used for Lorentzian line shapes if only specifying
    :attr:`~LineShapeParameters.gamma_min_min`.
    """

    T_0: Quantity["K"]
    """ Reference temperature of broadening coefficients [K] """
    gamma_min_min: Quantity["MHz/torr"]
    """ Self-broadened linewidth parameter [MHz/torr] """
    gamma_min_maj: Quantity["MHz/torr"] = Quantity(0, "MHz/torr")
    """ Foreign-broadened linewidth parameter [MHz/torr] """
    zeta_min_min: Quantity["MHz/torr"] = Quantity(0, "MHz/torr")
    """ Self-coupling linewidth parameter [MHz/torr] """
    zeta_min_maj: Quantity["MHz/torr"] = Quantity(0, "MHz/torr")
    """ Foreign-coupling parameter [MHz/torr] """
    delta_min: Quantity["MHz/torr"] = Quantity(0, "MHz/torr")
    """ Frequency shift parameter [MHz/torr] """
    m: float = 0.0
    """ Temperature dependence of the coupling [-] """
    n: float = 0.0
    """ Temperature dependence of the linewidth [-] """


def write_polarization_parameters(
    polarization_parameters: dict[
        str, HarveyLemmon2005Parameters | Pitzer1983Parameters
    ],
    filename: str | Path,
):
    """
    Write a TOML file containing all the polarization parameters.

    Parameters
    ----------
    polarization_parameters
        Dictionary that containes the parameter objects for each species
    filename
        Full file name to write the polarization parameters to.
    """
    # get current time
    tz = datetime.now().astimezone().tzinfo
    now = datetime.now(tz).isoformat(timespec="seconds")
    # start document with general information
    doc = tok.document()
    doc.add(tok.comment(f"Polarization parameters written by XVAMP on {now}"))
    doc.add(tok.nl())
    doc.add(tok.comment("Each following table describes the parameter set for a"))
    doc.add(tok.comment("given species by describing its type and then all"))
    doc.add(tok.comment("individual parameters required by the set type."))
    # loop over species
    for spec in sorted(polarization_parameters.keys()):
        # get parameter set
        pp = polarization_parameters[spec]
        # create new table
        tab = tok.table()
        # write class type
        tab["__name__"] = type(pp).__name__
        # loop over individual parameters
        for k, v in pp.__dict__.items():
            # save with units if it has one
            if isinstance(v, Quantity):
                tab[k] = [float(v.value), str(v.unit)]
            else:
                tab[k] = float(v)
        # add to document
        doc[spec] = tab
    # write to file
    with open(filename, mode="w") as fp:
        tok.dump(doc, fp)
    # done


def read_polarization_parameters(
    filename: str | Path | None = None,
) -> dict[str, HarveyLemmon2005Parameters | Pitzer1983Parameters]:
    """
    Read a TOML file containing all the polarization parameters.

    Parameters
    ----------
    filename
        Full file name to read the polarization parameters from.
        If ``None``, the XVAMP defaults will be loaded.

    Returns
    -------
        Dictionary that containes the parameter objects for each species
    """
    # check if we should load defaults
    if filename is None:
        filename = res_files(data) / "default_polarization_parameters.toml"
    # load file
    with open(filename, mode="r") as fp:
        doc = tok.load(fp)
    # unwrap TOML object
    doc = doc.unwrap()
    # initialize empty dictionary
    polarization_parameters = {}
    # loop over tables
    for spec, pp in doc.items():
        # get type
        t = pp.pop("__name__")
        match t:
            case "HarveyLemmon2005Parameters":
                parclass = HarveyLemmon2005Parameters
            case "Pitzer1983Parameters":
                parclass = Pitzer1983Parameters
            case _:
                raise ValueError(
                    f"Unrecognized parameter class {t} in file '{filename}'."
                )
        # convert parameters into Quantities if necessary
        for k, v in pp.items():
            if isinstance(v, list):
                pp[k] = Quantity(v[0], v[1])
        # instantiate parameter set and save to dictionary
        polarization_parameters[spec] = parclass(**pp)
    # done
    return polarization_parameters


def fill_df(
    df: DataFrame,
    interpolators: dict = {},
    log_list: list = [],
    ffill_list: list = [],
    bfill_list: list = [],
    zero_list: list = [],
) -> DataFrame:
    """
    Inter- and extrapolate a :class:`~pandas.DataFrame`
    in-place based on its index values.

    First, it fills missing values from a dictionary of interpolating functions.
    Second, it interpolates between valid values in linear (default) or log space.
    Third, it fills columns forwards and backwards based on their last and first
    value, respectively.
    Fourth, values that are still missing are set to zero.

    Parameters
    ----------
    df
        Input :class:`~pandas.DataFrame` with missing values set to ``NaN``
    interpolators
        Dictionary that contains interpolating functions, indexed by
        column name
    log_list
        List of columns to interpolate in logarithmic space
    ffill_list
        List of columns to fill the last value forward
    bfill_list
        List of columns to fill the first value backward
    zero_list
        List of columns where to replace leftover ``NaN`` with zeros

    Returns
    -------
        Output :class:`~pandas.DataFrame`
    """

    # 1. call interpolators
    for col, interp in interpolators.items():
        df.loc[:, col] = interp(df.index).value

    # 2. interpolate between valid values
    # convert log quantities, avoiding numerical issues
    df.replace({c: 0 for c in log_list}, 1e-100, inplace=True)
    df.loc[:, log_list] = np.log10(df.loc[:, log_list])
    # interpolate
    df.interpolate("index", limit_area="inside", inplace=True)
    # convert log quantities back to linear space
    df.loc[:, log_list] = 10 ** df.loc[:, log_list]
    df.replace({c: 1e-100 for c in log_list}, 0, inplace=True)

    # 3. apply forward and backward fill
    for col in ffill_list:
        df.loc[:, col] = df.loc[:, col].ffill(limit_area="outside")
    for col in bfill_list:
        df.loc[:, col] = df.loc[:, col].bfill(limit_area="outside")

    # 4. set still missing values to zero
    df.fillna({col: 0.0 for col in zero_list}, inplace=True)

    # done
    return df


def cast_to_np(input: Any | Quantity, unit: str) -> Any | NDArray[np.floating]:
    """
    Convert a :class:`~astropy.units.Quantity` into a NumPy array of [unit],
    or simply return the input if it's not a :class:`~astropy.units.Quantity`.
    """
    try:
        return input.to(unit).value
    except AttributeError:
        return input


def interpolate_nodes(
    altitudes: NDArray[np.floating],
    nodes: NDArray[np.floating],
    scale: float = 1.0,
    log: bool = False,
    left_constant: bool = False,
) -> NDArray[np.floating]:
    """
    Linearly interpolate between nodes, apply a scaling factor,
    and transform from logarithmic space.

    Parameters
    ----------
    altitudes
        Altitudes in [km] at which to compute values
    nodes
        Nodes defining the interpolation [km, -]
    scale
        Scaling factor to apply to data
    log
        Set to ``True`` if the nodes are in logarithmic space,
        so that the output is transformed back to linear space
    left_constant
        If ``True``, set the left (lower) values outside of the
        interpolating range to the leftmost valid value (instead of
        setting it to zero, which is the default).

    Returns
    -------
        Interpolated and padded values
    """
    if log:
        out = scale * 10 ** np.interp(
            altitudes,
            nodes[:, 0],
            nodes[:, 1],
            left=None if left_constant else np.nan,
            right=np.nan,
        )
        out[np.isnan(out)] = 0
    else:
        out = scale * np.interp(
            altitudes,
            nodes[:, 0],
            nodes[:, 1],
            left=None if left_constant else 0,
            right=0,
        )
    return out


@dataclass
class BoundedInterpolatingBasis:
    """
    Smooth interpolator for models defined on bounded input.
    Uses sine and cosine squared as basis functions.
    Assumes constant values between the boundaries and the closest
    adjacent knots, and transitions between knots.

    Parameters
    ----------
    lower
        Lower boundary
    upper
        Upper boundary
    knots
        Array of knot values
    """

    lower: float
    upper: float
    knots: NDArray[np.floating]

    def __post_init__(self):
        # input check
        assert self.lower < self.upper
        assert self.knots.ndim == 1
        assert np.logical_and(
            np.all(self.lower < self.knots), np.all(self.upper > self.knots)
        )

    def __len__(self):
        """Number of basis functions"""
        return self.knots.size

    def __call__(self, nodes: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the basis functions that interpolates
        between knot-anchored models at given input nodes.

        Parameters
        ----------
        nodes
            Locations at which to compute the values of the
            basis functions

        Returns
        -------
            Coefficients that linearly combine the different models
        """
        # input check
        assert nodes.ndim == 1
        assert np.logical_and(np.all(self.lower <= nodes), np.all(self.upper >= nodes))
        # get distance to nodes
        dist = nodes[:, None] - self.knots[None, :]
        # initialize output
        coef = np.zeros(dist.shape)
        # coefficients after nodes
        temp = 1 - (dist / np.diff(self.knots, append=np.nan)[None, :])
        temp_mask = np.logical_and(temp > 0, temp <= 1)
        coef[temp_mask] = np.sin(temp[temp_mask] * np.pi / 2) ** 2
        # coefficients before nodes
        temp = -dist / np.diff(self.knots, prepend=np.nan)[None, :]
        temp_mask = np.logical_and(temp > 0, temp <= 1)
        coef[temp_mask] = np.cos(temp[temp_mask] * np.pi / 2) ** 2
        # coefficients between boundaries and adjacent knots
        coef[dist[:, 0] <= 0, 0] = 1
        coef[dist[:, -1] >= 0, -1] = 1
        # done
        return coef


@dataclass
class PeriodicInterpolatingBasis:
    """
    Smooth interpolator for models defined on periodic input.
    Uses sine and cosine squared as basis functions.
    Assumes transitions between knots (including across the boundaries),
    except where ``const_between_indices`` is set.

    Parameters
    ----------
    lower
        Lower boundary
    upper
        Upper boundary, wraps to lower one
    knots
        Array of knot values
    const_between_indices
        Force a constant value between these knot indices
        (reduces the number of basis functions created)
    """

    # input parameters
    lower: float
    upper: float
    knots: NDArray[np.floating]
    const_between_indices: list[tuple] = field(default_factory=list)

    # internal values
    period: float = field(init=False)
    output_columns: list[tuple] = field(init=False)

    def __post_init__(self):
        # input check
        assert self.lower < self.upper
        assert self.knots.ndim == 1
        assert np.logical_and(
            np.all(self.lower < self.knots), np.all(self.upper > self.knots)
        )
        assert isinstance(self.const_between_indices, list)
        assert all(
            isinstance(t, tuple)
            and (len(t) == 2)
            and all(tt < self.knots.size for tt in t)
            and (t[1] == (t[0] + 1) % self.knots.size)
            for t in self.const_between_indices
        )
        # compute period
        self.period = self.upper - self.lower
        # get true output columns
        self.output_columns = list(range(self.knots.size))
        for t in self.const_between_indices:
            self.output_columns.remove(t[1] % self.knots.size)

    def __len__(self):
        """Number of basis functions"""
        return len(self.output_columns)

    def __call__(self, nodes: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the basis functions that linearly interpolate
        between knot-anchored models at given input nodes.

        Parameters
        ----------
        nodes
            Locations at which to compute the values of the
            basis functions

        Returns
        -------
            Coefficients that linearly combine the different models
        """
        # input check
        assert nodes.ndim == 1
        # fold input into wrapped range
        nodes = (nodes - self.lower) % self.period + self.lower
        assert np.logical_and(np.all(self.lower <= nodes), np.all(self.upper >= nodes))
        # get distance to nodes
        dist = (nodes[:, None] - self.knots[None, :]) % self.period
        dist[dist > self.period / 2] -= self.period
        # initialize output
        coef = np.zeros(dist.shape)
        # coefficients after nodes
        temp = 1 - (
            dist / np.diff(np.r_[self.knots, self.knots[0] + self.period])[None, :]
        )
        temp_mask = np.logical_and(temp > 0, temp <= 1)
        coef[temp_mask] = np.sin(temp[temp_mask] * np.pi / 2) ** 2
        # coefficients before nodes
        temp = -dist / np.diff(np.r_[self.knots[-1] - self.period, self.knots])[None, :]
        temp_mask = np.logical_and(temp > 0, temp <= 1)
        coef[temp_mask] = np.cos(temp[temp_mask] * np.pi / 2) ** 2
        # sum the columns for the range that should remain constant
        for t in self.const_between_indices:
            coef[:, t[0]] += coef[:, t[1]]
        coef = coef[:, self.output_columns]
        # done
        return coef


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
    central_angle = np.atleast_1d(central_angle)
    assert central_angle.ndim == 1
    height_terrain = np.atleast_1d(height_terrain)
    assert height_terrain.ndim == 1
    height_platform = np.atleast_1d(height_platform)
    assert height_platform.ndim == 1
    nout = np.max([central_angle.size, height_terrain.size, height_platform.size])
    assert central_angle.size in [
        1,
        nout,
    ], f"{central_angle.size=}, expected 1 or {nout}."
    assert height_terrain.size in [
        1,
        nout,
    ], f"{height_terrain.size=}, expected 1 or {nout}."
    assert height_platform.size in [
        1,
        nout,
    ], f"{height_platform.size=}, expected 1 or {nout}."
    # parse input to common units
    if not isinstance(central_angle, Quantity):
        central_angle = Quantity(central_angle, "rad")
    try:
        rp = VENUS_RADIUS + height_platform
    except UnitConversionError:
        rp = VENUS_RADIUS + Quantity(height_platform, "km")
    try:
        rt = VENUS_RADIUS + height_terrain
    except UnitConversionError:
        rt = VENUS_RADIUS + Quantity(height_terrain, "km")
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
