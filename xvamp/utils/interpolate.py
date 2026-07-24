"""
Interpolation helper functions.
"""

# standard imports
import numpy as np
from dataclasses import dataclass, field
from numpy.typing import NDArray
from pandas import DataFrame
from warnings import deprecated


@deprecated(
    "'utils.interpolate.fill_df' will be removed as it is no longer needed",
    category=FutureWarning,
)
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
