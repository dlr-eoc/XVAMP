"""
Interpolation helper functions.
"""

# standard imports
import numpy as np
from dataclasses import dataclass, field
from numpy.typing import NDArray


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
