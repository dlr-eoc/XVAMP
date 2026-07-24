"""
Utility module for the atmospheric model.
"""

# standard imports
from __future__ import annotations
import numpy as np
from typing import List, Any
from numpy.typing import NDArray
from pandas import DataFrame
from astropy.units import Quantity, Unit, UnitsError, get_physical_type
from astropy.table import QTable


# helper function
def cast_to_np(input: Any | Quantity, unit: str) -> Any | NDArray[np.floating]:
    """
    Convert a :class:`~astropy.units.Quantity` into a NumPy array of [unit],
    or simply return the input if it's not a :class:`~astropy.units.Quantity`.
    """
    try:
        return input.to_value(unit)
    except AttributeError:
        return input


# unit checker helper
def check_physical_type(
    p: Profile,
    data_physical_type: str,
    index_physical_type: str | None = None,
    name: str | None = None,
):
    """
    Check whether a Profile has data (and optionally, index) of the desired
    :mod:`~astropy.units.physical` type.

    Parameters
    ----------
    p
        :class:`~profile` to check
    data_physical_type
        Desired physical type of the data
    index_physical_type
        Desired physical type of the index
    name
        Add this name to the raised error if a check fails

    Raises
    ------
    UnitsError
        If the data (and/or index) is of the wrong physical type
    """
    msg_suffix = f" in {name}" if name is not None else ""
    if (index_physical_type is not None) and (
        get_physical_type(p.index_unit) != get_physical_type(index_physical_type)
    ):
        raise UnitsError("Wrong index unit" + msg_suffix)
    if get_physical_type(p.data_unit) != get_physical_type(data_physical_type):
        raise UnitsError("Wrong data unit" + msg_suffix)


class Profile:

    index: NDArray[np.floating]
    """ Index nodes of data """
    data: NDArray[np.floating]
    """ Data values at the indices """
    index_unit: Unit
    """ Unit of :attr:`~index` """
    data_unit: Unit
    """ Unit of :attr:`~data` """
    log: bool
    """ Whether :attr:`~data` is saved in logarithmic space """
    lower_constant: bool
    """ Sets downward continuation to constant rather than 0 """
    upper_constant: bool
    """ Sets downward continuation to constant rather than 0 """

    def __init__(
        self,
        index: NDArray[np.floating] | Quantity,
        data: NDArray[np.floating] | Quantity,
        index_unit: Unit | str | None = None,
        data_unit: Unit | str | None = None,
        scale: float = 1.0,
        log: bool = False,
        lower: float | None = 0.0,
        upper: float | None = 0.0,
        check: bool = True,
    ):
        """
        Class providing interfaces to loading and interpolating generic atmospheric
        profiles.

        Parameters
        ----------
        index
            Index nodes (e.g., altitudes) at which ``data`` values are present.
            If not a :class:`~astropy.units.Quantity`, ``index_unit`` must be set.
        data
            Data nodes (e.g., pressure or mixing ratio) at the ``index`` locations.
            If not a :class:`~astropy.units.Quantity`, ``data_unit`` must be set.
        index_unit
            Unit of ``index``. Ignored if ``index`` is a
            :class:`~astropy.units.Quantity`, required if it is not.
        data_unit
            Unit of ``index``. Ignored if ``index`` is a
            :class:`~astropy.units.Quantity`, required if it is not.
        scale
            Scaling factor to apply to data (in linear space)
        log
            Set to ``True`` if the input nodes are in logarithmic space,
            so that the output is transformed back to linear space
        lower
            Set the lower (left) values outside of the interpolating range to this
            value (default ``0``). Set to ``None`` to use the leftmost valid value
            (see :func:`numpy.interp` ``left`` parameter with different default).
        upper
            Set the upper (right) values outside of the interpolating range to this
            value (default ``0``). Set to ``None`` to use the rightmost valid value
            (see :func:`numpy.interp` ``right`` parameter with different default).
        check
            Check the input shapes and index monotonicity.
        """
        # save index
        if isinstance(index, Quantity):
            self.index = index.value
            self.index_unit = index.unit
        else:
            if index_unit is None:
                raise ValueError(
                    "Must define 'index_unit' if 'index' is not a Quantity"
                )
            self.index = index
            self.index_unit = Unit(index_unit)
        # save data
        if isinstance(data, Quantity):
            self.data = data.value
            self.data_unit = data.unit
        else:
            if data_unit is None:
                raise ValueError("Must define 'data_unit' if 'data' is not a Quantity")
            self.data = data
            self.data_unit = Unit(data_unit)
        # perform input checks
        if check:
            # enforce array type
            self.index = np.atleast_1d(self.index).astype(float)
            if not self.index.ndim == 1:
                raise ValueError(
                    f"'index' must be one-dimensional, got shape {self.index.shape}"
                )
            self.data = np.atleast_1d(self.data).astype(float)
            if not self.data.ndim == 1:
                raise ValueError(
                    f"'data' must be one-dimensional, got shape {self.data.shape}"
                )
            # check their sizes
            if not self.index.size == self.data.size:
                raise ValueError(
                    "Mismatching array sizes "
                    f"(index: {self.index.size}, data: {self.data.size})"
                )
            # check monotonicity of index
            index_pos_diff = np.diff(self.index) > 0
            if not np.all(index_pos_diff):
                err_subset = np.flatnonzero(~index_pos_diff)
                err_subset = np.unique(
                    np.clip(
                        np.r_[err_subset - 1, err_subset, err_subset + 1],
                        a_min=0,
                        a_max=self.index.size - 1,
                    )
                )
                err_df = DataFrame(
                    index=err_subset,
                    data={
                        "index": self.index[err_subset],
                        "data": self.data[err_subset],
                    },
                )
                raise ValueError(
                    f"Index not strictly monotonically increasing:\n{err_df}"
                )
        # apply scaling to data
        if log:
            self.data += np.log10(scale)
        else:
            self.data *= scale
        # save settings
        self.log = log
        self.lower = lower
        self.upper = upper
        # done

    def __len__(self) -> int:
        return self.index.size

    def __str__(self) -> str:
        return (
            f"Profile of length {len(self)} with\n"
            f"- index_unit={self.index_unit}\n"
            f"- data_unit={self.data_unit}\n"
            f"- log={self.log}\n"
            f"- lower={self.lower}\n"
            f"- upper={self.upper}"
        )

    def __call__(self, new_index: NDArray[np.floating] | Quantity) -> Quantity:
        """
        Linearly interpolates the profile data (either in linear or logarithmic space,
        depending on how it is stored, see :attr:`~log`) onto a new index given the
        down- and upward continuation settings in :attr:`~lower` and :attr:`~upper`.

        Parameters
        ----------
        new_index
            New index values (if not a :class:`~astropy.units.Quantity`, must already
            be in the unit of this profile [:attr:`~index_unit`])

        Returns
        -------
            New data values in [:attr:`~data_unit`]
        """
        # make sure we have the right index units
        new_index = cast_to_np(new_index, self.index_unit)
        # continue depending on whether we're interpolating in logarithmic space or not
        if self.log:
            if (self.lower is not None) or (self.upper is not None):
                # need an array for where to set values after interpolating
                if self.lower is not None:
                    i_lower = new_index < self.index[0]
                if self.upper is not None:
                    i_upper = new_index > self.index[-1]
            out = 10 ** np.interp(
                new_index,
                self.index,
                self.data,
                left=None if self.lower is None else np.nan,
                right=None if self.upper is None else np.nan,
            )
            if self.lower is not None:
                out[i_lower] = self.lower
            if self.upper is not None:
                out[i_upper] = self.upper
        else:
            out = np.interp(
                new_index,
                self.index,
                self.data,
                left=self.lower,
                right=self.upper,
            )
        # cast as Quantity and return
        return Quantity(out, self.data_unit)

    @property
    def index_as_quantity(self) -> Quantity:
        """Return the index as a :class:`~astropy.unit.Quantity`."""
        return Quantity(self.index, self.index_unit)

    def index_to(self, unit: Unit | str | None = None) -> NDArray[np.floating]:
        """
        Return the index as as an array in a given unit.

        Parameters
        ----------
        unit
            If not a string or :class:`astropy.unit.Unit`, the :attr:`~index_unit`
            is assumed.
        """
        return self.index_as_quantity.to_value(unit)

    @property
    def as_quantity(self) -> Quantity:
        """Return the data as a :class:`~astropy.unit.Quantity`."""
        return Quantity(self.data, self.data_unit)

    def to(self, unit: Unit | str | None = None) -> NDArray[np.floating]:
        """
        Return the data as as an array in a given unit.

        Parameters
        ----------
        unit
            If not a string or :class:`astropy.unit.Unit`, the :attr:`~data_unit`
            is assumed.
        """
        return self.as_quantity.to_value(unit)


class MultiProfile:

    index: NDArray[np.floating]
    """ Index nodes of data """
    index_unit: Unit
    """ Unit of :attr:`~index` """
    # each individual profile will be set as an attribute
    data_names: List[str]
    """ Names of the individual profiles """

    def __init__(
        self,
        index: NDArray[np.floating] | Quantity,
        data: NDArray[np.floating] | Quantity | QTable,
        index_unit: Unit | str | None = None,
        data_units: List[Unit] | Unit | str | None = None,
        data_names: List[str] | None = None,
        scales: List[float] | float = 1.0,
        log: List[bool] | bool = False,
        lower: List[float] | float | None = 0.0,
        upper: List[float] | float | None = 0.0,
    ):
        """
        Class providing an interface to define multiple :class:`~Profile`s with a
        shared index.

        Parameters
        ----------
        index
            Index nodes (e.g., altitudes) at which ``data`` values are present.
            If not a :class:`~astropy.units.Quantity`, ``index_unit`` must be set.
        data
            Data nodes (e.g., pressure or mixing ratio) at the ``index`` locations.
            If not a :class:`~astropy.units.Quantity`, ``data_units`` must be set.
            If ``data`` is a 2D NumPy array, ``index`` applies to the first axis
            (matching the :class:`~astropy.table.QTable` layout).
        index_unit
            Unit of ``index``. Ignored if ``index`` is a
            :class:`~astropy.units.Quantity`, required if it is not.
        data_units
            Unit(s) of ``data``. Ignored if ``data`` is a
            :class:`~astropy.units.Quantity` or :class:`~astropy.table.QTable`,
            required if it is not.
            If a single unit and the data is 2D, the unit is applied to all.
        data_names
            List of names of the data column(s). Required if ``data`` is not a
            :class:`~astropy.table.QTable`, otherwise it is optional and would override
            the column names.
        scales
            Scaling factor to apply to data (in linear space).
            If a single factor and the data is 2D, the factor is applied to all.
        log
            Set to ``True`` if the input data nodes are in logarithmic space,
            so that the output is transformed back to linear space.
            If a single flag and the data is 2D, the flag is applied to all.
        lower
            Set the lower (left) values outside of the interpolating range to this
            value (default ``0``). Set to ``None`` to use the leftmost valid value (see
            :func:`numpy.interp` ``left`` parameter with different default).
            If a single flag and the data is 2D, the flag is applied to all.
        upper
            Set the upper (right) values outside of the interpolating range to this
            value (default ``0``). Set to ``None`` to use the rightmost valid value (see
            :func:`numpy.interp` ``right`` parameter with different default).
            If a single flag and the data is 2D, the flag is applied to all.
        """
        # save index
        if isinstance(index, Quantity):
            self.index = index.value
            self.index_unit = index.unit
        elif isinstance(index, np.ndarray):
            if index_unit is None:
                raise ValueError("Must define 'index_unit' if 'index' is a NumPy array")
            self.index = index
            self.index_unit = Unit(index_unit)
        else:
            raise TypeError(f"Cannot parse input 'index' of type {type(index)}")
        # check index shape
        self.index = np.atleast_1d(self.index).astype(float)
        if not self.index.ndim == 1:
            raise ValueError(
                f"'index' must be one-dimensional, got shape {self.index.shape}"
            )
        # parse data
        if isinstance(data, QTable):
            data_units = [data[c].unit for c in data.columns]
            if data_names is None:
                data_names = data.colnames
            data = data.to_pandas().to_numpy()
        elif isinstance(data, Quantity):
            data_units = [data.unit]
            data = data.value
        elif isinstance(data, np.ndarray):
            if data_units is None:
                raise ValueError("Must define 'data_units' if 'data' is a NumPy array")
            if isinstance(data_units, list):
                data_units = [Unit(du) for du in data_units]
            else:
                data_units = [Unit(data_units)]
        else:
            raise TypeError(f"Cannot parse input 'data' of type {type(index)}")
        # data is now an array and data_units is now a list (of unknown length)
        # make 2D by having each profile be a row
        if data.ndim < 2:
            data = data.reshape(1, -1).astype(float)
        elif data.ndim > 2:
            raise ValueError(
                f"'data' must be one- or two-dimensional, got shape {data.shape}"
            )
        else:
            # it's already 2D but we make sure it's subprofile-contiguous
            data = np.ascontiguousarray(data.T, dtype=float)
        # readability variable for later
        n_cols = data.shape[0]
        # extend the data unit list if necessary
        if len(data_units) == 1:
            data_units = [data_units[0]] * n_cols
        elif len(data_units) != n_cols:
            raise ValueError(
                f"The 'data_units' supplied ({data_units}) are not a single "
                f"one and do not match the number of data columns ({n_cols})"
            )
        # compare the index and data shapes
        if not self.index.size == data.shape[1]:
            raise ValueError(
                "Mismatching array sizes "
                f"(index: {self.index.size}, data: {data.T.shape})"
            )
        # check monotonicity of index
        index_pos_diff = np.diff(self.index) > 0
        if not np.all(index_pos_diff):
            err_subset = np.flatnonzero(~index_pos_diff)
            err_subset = np.unique(
                np.clip(
                    np.r_[err_subset - 1, err_subset, err_subset + 1],
                    a_min=0,
                    a_max=self.index.size - 1,
                )
            )
            err_df = DataFrame(
                index=err_subset,
                data={
                    "index": self.index[err_subset],
                    "data": data[:, err_subset],
                },
            )
            raise ValueError(f"Index not strictly monotonically increasing:\n{err_df}")
        # check scales, log, lower, and upper flags
        if not isinstance(scales, list):
            scales = [scales] * n_cols
        elif len(scales) != n_cols:
            raise ValueError(
                f"The 'scales' supplied ({scales}) are not a single one and do not "
                f"match the number of data columns ({n_cols})"
            )
        if not isinstance(log, list):
            log = [log] * n_cols
        elif len(log) != n_cols:
            raise ValueError(
                f"The 'log' flags supplied ({log}) are not a single one and do not "
                f"match the number of data columns ({n_cols})"
            )
        if not isinstance(lower, list):
            lower = [lower] * n_cols
        elif len(lower) != n_cols:
            raise ValueError(
                f"The 'lower' flags supplied ({lower}) are not a single one and do "
                f"not match the number of data columns ({n_cols})"
            )
        if not isinstance(upper, list):
            upper = [upper] * n_cols
        elif len(upper) != n_cols:
            raise ValueError(
                f"The 'upper' flags supplied ({upper}) are not a single one and do "
                f"not match the number of data columns ({n_cols})"
            )
        # create the individual subprofiles
        if not (
            isinstance(data_names, list)
            and all(isinstance(n, str) for n in data_names)
            and len(data_names) == n_cols
        ):
            raise ValueError(
                "'data_names' is not a list of strings "
                f"matching the number of data columns ({n_cols})"
            )
        # assign quick access to the data columns by name
        for i, n in enumerate(data_names):
            if hasattr(self, n):
                raise AttributeError(
                    f"Cannot name data column #{i} '{n}' because the "
                    "attribute already exists"
                )
            else:
                object.__setattr__(
                    self,
                    n,
                    Profile(
                        index=self.index,
                        data=data[i, :],
                        index_unit=self.index_unit,
                        data_unit=data_units[i],
                        scale=scales[i],
                        log=log[i],
                        lower=lower[i],
                        upper=upper[i],
                        check=False,
                    ),
                )
        # save list of names
        self.data_names = data_names
        # done

    def __len__(self) -> int:
        return len(self.data_names)

    def __str__(self) -> str:
        return (
            f"MultiProfile with index of length {len(self.index)} "
            f"[{self.index_unit}] and the {len(self)} Profiles"
            + "".join(
                (f"\n- {n} [{getattr(self, n).data_unit}]" for n in self.data_names)
            )
        )

    def __call__(self, new_index: NDArray[np.floating] | Quantity) -> QTable:
        """
        Linearly interpolates all subprofiles onto a new index.

        Parameters
        ----------
        new_index
            New index values (if not a :class:`~astropy.units.Quantity`, must already
            be in the unit of this profile [:attr:`~index_unit`])

        Returns
        -------
            New data values in their respective units, where each column corresponds
            to the individual subprofiles as ordered in :attr:`data_names`
        """
        return QTable(data={n: getattr(self, n)(new_index) for n in self.data_names})

    @property
    def index_as_quantity(self) -> Quantity:
        """Return the index as a :class:`~astropy.unit.Quantity`."""
        return Quantity(self.index, self.index_unit)

    def index_to(self, unit: Unit | str | None = None) -> NDArray[np.floating]:
        """
        Return the index as as an array in a given unit.

        Parameters
        ----------
        unit
            If not a string or :class:`astropy.unit.Unit`, the :attr:`~index_unit`
            is assumed.
        """
        return self.index_as_quantity.to_value(unit)
