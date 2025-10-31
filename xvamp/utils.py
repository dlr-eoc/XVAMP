"""
Utility module for the atmospheric model.
"""

# standard imports
from typing import Any
import astropy.units as u
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from numpy.typing import NDArray
from pandas import DataFrame
from astropy.units import Quantity, UnitConversionError
from astropy.table import Table, QTable

# package imports
from .constants import HEADERPATTERN, VENUS_RADIUS

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

    Returns
    -------
        Table with quantities and units
    """
    # read data
    data = np.genfromtxt(path, dtype=formats, delimiter=widths, encoding="utf8")
    # build QTable
    return QTable(data, names=names, units=units)


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


def geometric_range_from_central_angle(
    height_terrain: Quantity | float_or_array,
    height_platform: Quantity | float_or_array,
    central_angle: Quantity | float_or_array,
) -> Quantity:
    """
    Calculate the geometric range from the law of cosines.

    Parameters
    ----------
    height_terrain
        Height of the terrain relative to the mean planet radius in [km],
        if not a :class:`~astropy.units.Quantity`.
    height_platform
        Height of the platform relative to the mean planet radius in [km],
        if not a :class:`~astropy.units.Quantity`.
    central_angle
        Central angle of the platform in [rad], if not a
        :class:`~astropy.units.Quantity`. Must be of broadcastable shape
        to ``height_platform`` and ``height_terrain``.

    Returns
    -------
        Geometric range from the platform to the surface [km].
    """
    # input check
    assert height_terrain.ndim < 2
    assert height_platform.ndim < 2
    height_terrain = np.atleast_1d(height_terrain)
    height_platform = np.atleast_1d(height_platform)
    # parse input to common units
    try:
        radius_platform = VENUS_RADIUS + height_platform[:, None]
    except UnitConversionError:
        radius_platform = VENUS_RADIUS + Quantity(height_platform[:, None], "km")
    try:
        radius_terrain = VENUS_RADIUS + height_terrain[None, :]
    except UnitConversionError:
        radius_terrain = VENUS_RADIUS + Quantity(height_terrain[None, :], "km")
    # return law of cosines
    return np.sqrt(
        radius_platform**2
        + radius_terrain**2
        - 2 * radius_platform * radius_terrain * np.cos(central_angle)
    )
