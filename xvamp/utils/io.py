"""
Utility module for the atmospheric model.
"""

# standard imports
from datetime import datetime
from importlib.resources import files as res_files
import astropy.units as u
import numpy as np
import tomlkit as tok
from pathlib import Path
from astropy.units import Quantity
from astropy.table import Table, QTable

# package imports
from .. import data
from ..constants import HEADERPATTERN
from .parametersets import HarveyLemmon2005Parameters, Pitzer1983Parameters


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
