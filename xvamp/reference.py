"""
Module that provides the background, reference atmospheric properties.
"""

# standard imports
import numpy as np
import astropy.units as u
from importlib.resources import files as res_files
from typing import Literal, Tuple
from numpy.typing import NDArray
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator, Akima1DInterpolator
from astropy.units import Quantity, Unit
from astropy.table import Table, QTable, hstack, vstack, join

# package imports
from . import data
from .constants import ESU_CM, GAS_CONSTANT, SPEC_MOL_M
from .utils import (
    BoundedInterpolatingBasis,
    cast_to_np,
    get_sza,
    interpolate_nodes,
    read_unit_csv,
    read_unit_fwf,
)

# classes get defined here and then initialized at the end


class Reference:
    """
    Abstract base class for reference atmospheric models.
    """

    tables: dict[str, QTable]
    """ Dictionary of all tables associated with this reference """

    pass


class Cimino1982(Reference):
    """
    Reference class that provides the reference profiles from :cite:t:`cimino1982`.
    """

    BASEFOLDER = "cimino_1982"
    """ Base folder for data """

    TABLES = [f"fig{n}raw" for n in ["7", "8", "9"]]
    """ Table numbers to load """

    LAMBDA_S = 11.32
    """ Wavelength of S-band experiment [cm] """

    LAMBDA_X = 3.56
    """ Wavelength of X-band experiment [cm] """

    EPS_DROPLETS_CORE = Quantity(2.9, u.dimensionless_unscaled)
    """ Estimated dielectric constant of the cloud droplet cores """

    RATIO_RADIUS_DROPLETS = Quantity(0.97, u.dimensionless_unscaled)
    """ Ratio of the core radius to the total droplet radius  """

    D_DROPLET_CORE = Quantity(2.5, "g/cm3")
    """ Mass density of the droplet core """

    D_DROPLET_SHELL = Quantity(2.0, "g/cm3")
    """ Mass density of the droplet shell """

    RATIO_MASS_DROPLETS = (
        (D_DROPLET_CORE * RATIO_RADIUS_DROPLETS**3)
        / (D_DROPLET_SHELL * (1 - RATIO_RADIUS_DROPLETS**3))
    ).decompose()
    """ Ratio of the core mass to the shell mass"""

    EXTRAPOLATE_DOWN_TO = 200.0
    """ Extrapolate the data down to this temperature [K] """

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load super-raw data files
        self.tables = {t: Table.read(datafolder / f"table{t}.csv") for t in self.TABLES}

        # rename and join the two tables for the imaginary part
        t_real = self.tables["fig7raw"]
        t_imaginary = hstack([self.tables["fig8raw"], self.tables["fig9raw"]])

        # get all different temperatures for each band (same for real & imaginary)
        temps = {
            "s": np.unique([cn[2:-2] for cn in t_real.colnames if cn[0] == "s"]),
            "x": np.unique([cn[2:-2] for cn in t_real.colnames if cn[0] == "x"]),
        }

        # make readable tables
        self.x_y_pairs = {}
        for tbl, part in zip([t_real, t_imaginary], ["Real", "Imaginary"]):
            for band in ["s", "x"]:
                # extract data for each temperature
                list_raw = []
                for t in temps[band]:
                    # get raw data
                    raw = np.ma.stack(
                        [
                            tbl[f"{band}/{t}/x"].value,
                            np.full(len(tbl[f"{band}/{t}/x"]), float(t)),
                            tbl[f"{band}/{t}/y"].value,
                        ],
                        axis=1,
                    )
                    # remove potentially missing values
                    if np.any(raw.mask):
                        raw = raw[~(raw.mask[:, 0]), :]
                    raw = raw.data
                    # linearly extrapolate dielectric values to 100% concentration
                    raw_slope = (raw[-1, 2] - raw[-2, 2]) / (raw[-1, 0] - raw[-2, 0])
                    new_vals = np.r_[
                        100, raw[-1, 1], raw_slope * (100 - raw[-1, 0]) + raw[-1, 2]
                    ]
                    # attach new value and save
                    list_raw.append(
                        np.concatenate([raw, new_vals.reshape(1, 3)], axis=0)
                    )
                # extrapolate to a lower temperature
                subtemps = np.array([float(t) for t in temps[band][:3]])
                local_interps = []
                for itemp in range(3):
                    local_interps.append(
                        Akima1DInterpolator(
                            list_raw[itemp][:, 0], list_raw[itemp][:, 2]
                        )
                    )
                concs = list_raw[0][:, 0]
                interp_vals = np.stack(
                    [interp(concs) for interp in local_interps], axis=1
                )
                new_vals = np.clip(
                    np.array(
                        [
                            np.polynomial.Polynomial.fit(
                                subtemps, interp_vals[ic, :], deg=2
                            )(self.EXTRAPOLATE_DOWN_TO)
                            for ic in range(concs.size)
                        ]
                    ),
                    a_min=0,
                    a_max=None,
                )
                new_table = np.stack(
                    [concs, np.full_like(concs, self.EXTRAPOLATE_DOWN_TO), new_vals],
                    axis=1,
                )
                list_raw.insert(0, new_table)
                # concatenate all temperatures
                raw_concat = np.concatenate(list_raw, axis=0)
                assert np.all(np.isfinite(raw_concat))
                # combine the different columns into a QTable with units
                self.tables[f"{part.lower()}_{band}"] = QTable(
                    raw_concat,
                    names=[
                        "Concentration H2SO4",
                        "Temperature",
                        f"{part} Dielectric Constant",
                    ],
                    units=["%", "K", ""],
                )
                # for later inter- and extrapolation, keep a quick access
                # to the raw data
                self.x_y_pairs[(part, band)] = (
                    self.tables[f"{part.lower()}_{band}"][
                        "Concentration H2SO4", "Temperature"
                    ]
                    .as_array()
                    .view((float, 2)),
                    self.tables[f"{part.lower()}_{band}"][
                        f"{part} Dielectric Constant"
                    ].value,
                )

        # build 4 2D interpolators, one for each wavelength and part,
        # which will be accessed by the global interpolator
        self.get_real_sband = LinearNDInterpolator(*self.x_y_pairs[("Real", "s")])
        self.get_real_xband = LinearNDInterpolator(*self.x_y_pairs[("Real", "x")])
        self.get_imaginary_sband = LinearNDInterpolator(
            *self.x_y_pairs[("Imaginary", "s")]
        )
        self.get_imaginary_xband = LinearNDInterpolator(
            *self.x_y_pairs[("Imaginary", "x")]
        )

    def get_h2s04_rel_permittivity(
        self,
        concentration: Quantity,
        temperature: Quantity,
        wavelength: Literal["s", "x"] | Quantity = "x",
    ) -> Tuple[Quantity, Quantity]:
        """
        Calculate the real and imaginary parts of the relative permittivity of
        H2SO4 clouds.

        Parameters
        ----------
        concentration
            Concentration of H2SO4 [%]
        temperature
            Temperature of the medium [K]
        wavelength
            Wavelength for which to compute the permittivity values [cm].
            If ``'s'`` or ``'x'`` (for S-band, 11.32 cm, or X-band, 3.56 cm),
            then only the corresponding table is used; otherwise, the
            values are inter- or extrapolated between the two tables.

        Returns
        -------
        eps_prime_r
            Real part of the relative permittivity
        eps_dprime_r
            Imaginary part of the relative permittivity

        Note
        ----
        Even if passing the exact same wavelength as one of the two
        tables will result in interpolation, possibly returning NaNs
        if the input concentration or temperature values are outside
        one of the two table's data ranges.

        Here, the imaginary part of the relative atmospheric permittivity
        has the opposite sign as in :cite:t:`duan2010`.
        """
        # convert input to NumPy arrays with the right units
        ct = np.stack([concentration.to("%").value, temperature.to("K").value], axis=1)
        # evaluate the interpolators according to the desired wavelength
        if wavelength == "s":
            # evaluate the S-band interpolators and return
            eps_prime_r = self.get_real_sband(ct)
            eps_dprime_r = self.get_imaginary_sband(ct)
            return eps_prime_r, eps_dprime_r
        elif wavelength == "x":
            # evaluate the X-band interpolators and return
            eps_prime_r = self.get_real_xband(ct)
            eps_dprime_r = self.get_imaginary_xband(ct)
            return eps_prime_r, eps_dprime_r
        else:
            # convert specific wavelength to cm
            l = wavelength.to("cm").value
            # evaluate all interpolators
            real_s = self.get_real_sband(ct)
            real_x = self.get_real_xband(ct)
            imaginary_s = self.get_imaginary_sband(ct)
            imaginary_x = self.get_imaginary_xband(ct)
            # get wavelength slopes
            delta_lambda = Cimino1982.LAMBDA_S - Cimino1982.LAMBDA_X
            real_slope = (real_s - real_x) / delta_lambda
            imaginary_slope = (imaginary_s - imaginary_x) / delta_lambda
            # extrapolate
            eps_prime_r = Quantity(
                real_slope * (l - Cimino1982.LAMBDA_X) + real_x,
                u.dimensionless_unscaled,
            )
            eps_dprime_r = Quantity(
                imaginary_slope * (l - Cimino1982.LAMBDA_X) + imaginary_x,
                u.dimensionless_unscaled,
            )
            # done
            return eps_prime_r, eps_dprime_r

    @staticmethod
    def get_h2so4_droplet_polarization(eps_r_shell: Quantity) -> Quantity:
        """
        Compute the droplet polarization assuming a shell-like structure,
        following eq. (10).

        Parameters
        ----------
        eps_r_shell
            Complex relative permittivity of the shell [-]

        Returns
        -------
            Polarization per molar volume of the H2SO4 cloud droplets [-]

        Note
        ----
        Here, the imaginary part of the relative atmospheric permittivity
        has the opposite sign as in :cite:t:`duan2010`.
        """
        # readability
        eps_c_plus_2s = Cimino1982.EPS_DROPLETS_CORE + 2 * eps_r_shell
        eps_c_minus_s = Cimino1982.EPS_DROPLETS_CORE - eps_r_shell
        q3 = Cimino1982.RATIO_RADIUS_DROPLETS**3
        # calculate polarization
        eps_r_valid = np.isfinite(eps_r_shell)
        Pnu_enum = (eps_r_shell - 1) * eps_c_plus_2s + q3 * (
            2 * eps_r_shell + 1
        ) * eps_c_minus_s
        Pnu_denom = (eps_r_shell + 2) * eps_c_plus_2s + q3 * (
            2 * eps_r_shell - 2
        ) * eps_c_minus_s
        Pnu = np.full_like(Pnu_enum, np.nan)
        Pnu[eps_r_valid] = Pnu_enum[eps_r_valid] / Pnu_denom[eps_r_valid]
        Pnu = Quantity(Pnu, u.dimensionless_unscaled)
        return Pnu


class Duan2010Figures(Reference):
    """
    Reference class that provides the reference profiles from :cite:t:`duan2010`.
    """

    BASEFOLDER = "duan_et_al_2010"
    """ Base folder for data """

    TABLES = [f"fig{n}" for n in ["6b", "7a", "7b", "7d", "8a", "8b", "9a", "9b"]] + [
        "4"
    ]
    """ Table numbers to load """

    ELECTRON_DENSITY_NODES = np.array(
        [
            [100, -2],
            [120, -2],
            [125, 0],
            [130, np.log10(1.3)],
            [140, np.log10(4)],
            [170, np.log10(0.7)],
            [250, np.log10(0.13)],
            [260, np.log10(0.02)],
            [375, np.log10(0.02)],
        ]
    )
    """
    Nodes in that define the electron density profile as pairs of
    altitude [km] and 1e11*log(electron density [1/m3])
    """

    H2O_FRACTION_NODES = np.array(
        [
            [4, 0],
            [5, 30],
            [60, 30],
            [60 + 1e-6, 10],
            [70 - 1e-6, 10],
            [70, 1.7],
            [75, 1.7],
            [85, 0.9],
            [88, 0.8],
            [92, 1.4],
            [98, 1.8],
            [99, 1.8],
            [110, 1.1],
            [112, 1.4],
            [113, 1.4],
            [116, 1.5],
            [117, 0],
        ]
    )
    """
    Nodes in that define the H2O molar fraction profile as pairs of
    altitude [km] and fraction [ppm]
    """

    SO2_FRACTION_NODES = np.array(
        [
            [15, 25],
            [33, 60],
            [37, 90],
            [40, 110],
            [40, 110],
            [42, 150],
            [44, 220],
            [48, 150],
            [50, 90],
            [56, 40],
            [61, 40],
            [62, 0],
        ]
    )
    """
    Nodes in that define the SO2 molar fraction profile as pairs of
    altitude [km] and fraction [ppm]
    """

    H2SO4_FRACTION_NODES = np.array(
        [
            [29, 0],
            [30, 0],
            [40, 1],
            [44, 2.4],
            [46, 9],
            [50, 1.7],
            [52, 1.7],
            [56, 0],
            [57, 0],
        ]
    )
    """
    Nodes in that define the H2SO4 molar fraction profile as pairs of
    altitude [km] and fraction [ppm]
    """

    CO_FRACTION_NODES = np.array(
        [
            [42, np.log10(28)],
            [52, np.log10(42)],
            [64, np.log10(52)],
            [68, np.log10(20)],
            [70, 1],
            [71, 1],
            [73, np.log10(30)],
            [77, np.log10(30)],
            [87, 1],
            [90, 1],
            [126, 4],
            [127, -100],
        ]
    )
    """
    Nodes in that define the CO molar fraction profile as pairs of
    altitude [km] and fraction [ppm]
    """

    OCS_FRACTION_NODES = np.array(
        [[29, 0], [30, 14], [33, 14], [34, 0], [37, 0], [38, 0.35], [40, 0.35], [41, 0]]
    )
    """
    Nodes in that define the OCS molar fraction profile as pairs of
    altitude [km] and fraction [ppm]
    """

    # unit definitions
    UNIT_EL_DENSITY = Unit("1/m3")
    """ Unit of the electron density profile [1/m3] """
    UNIT_MF = Unit("ppm")
    """ Unit of all molar fractions [ppm] """

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load raw data files
        self.tables = {
            t: read_unit_csv(datafolder / f"table{t}.csv") for t in self.TABLES
        }

    @staticmethod
    def get_electron_density(altitudes: NDArray[np.floating] | Quantity) -> Quantity:
        """
        Computes the electron number density as a linear
        interpolation between defined nodes.

        Parameters
        ----------
        altitudes
            Altitude values
            (if not a :class:`~astropy.units.Quantity`, must already be in [km])

        Returns
        -------
            Electron number density
        """
        # interpolate, pad, and add unit
        return Quantity(
            interpolate_nodes(
                cast_to_np(altitudes, "km"),
                Duan2010Figures.ELECTRON_DENSITY_NODES,
                1e11,
                log=True,
            ),
            Duan2010Figures.UNIT_EL_DENSITY,
        )

    @staticmethod
    def get_h2o_density(altitudes: NDArray[np.floating] | Quantity) -> Quantity:
        """
        Computes the H2O (water vapor) molar fraction as a
        linear interpolation between defined nodes.

        Parameters
        ----------
        altitudes
            Altitude values
            (if not a :class:`~astropy.units.Quantity`, must already be in [km])

        Returns
        -------
            H2O (water vapor) molar fraction
        """
        # interpolate, pad, and add unit
        return Quantity(
            interpolate_nodes(
                cast_to_np(altitudes, "km"), Duan2010Figures.H2O_FRACTION_NODES
            ),
            Duan2010Figures.UNIT_MF,
        )

    @staticmethod
    def get_so2_density(altitudes: NDArray[np.floating] | Quantity) -> Quantity:
        """
        Computes the SO2 molar fraction as a
        linear interpolation between defined nodes.

        Parameters
        ----------
        altitudes
            Altitude values
            (if not a :class:`~astropy.units.Quantity`, must already be in [km])

        Returns
        -------
            SO2 molar fraction
        """
        # interpolate and pad according to settings
        so2_interpolated = interpolate_nodes(
            cast_to_np(altitudes, "km"),
            Duan2010Figures.SO2_FRACTION_NODES,
            left_constant=True,
        )
        # add unit and return
        return Quantity(so2_interpolated, Duan2010Figures.UNIT_MF)

    @staticmethod
    def get_h2so4_density(altitudes: NDArray[np.floating] | Quantity) -> Quantity:
        """
        Computes the H2SO4 molar fraction as a
        linear interpolation between defined nodes.

        Parameters
        ----------
        altitudes
            Altitude values
            (if not a :class:`~astropy.units.Quantity`, must already be in [km])

        Returns
        -------
            H2SO4 molar fraction
        """
        # interpolate, pad, and add unit
        return Quantity(
            interpolate_nodes(
                cast_to_np(altitudes, "km"), Duan2010Figures.H2SO4_FRACTION_NODES
            ),
            Duan2010Figures.UNIT_MF,
        )

    @staticmethod
    def get_co_density(altitudes: NDArray[np.floating] | Quantity) -> Quantity:
        """
        Computes the CO molar fraction as a
        log-linear interpolation between defined nodes.

        Parameters
        ----------
        altitudes
            Altitude values
            (if not a :class:`~astropy.units.Quantity`, must already be in [km])

        Returns
        -------
            CO molar fraction
        """
        # interpolate and pad according to settings
        co_interpolated = interpolate_nodes(
            cast_to_np(altitudes, "km"),
            Duan2010Figures.CO_FRACTION_NODES,
            log=True,
            left_constant=True,
        )
        # add unit and return
        return Quantity(co_interpolated, Duan2010Figures.UNIT_MF)

    @staticmethod
    def get_ocs_density(altitudes: NDArray[np.floating] | Quantity) -> Quantity:
        """
        Computes the OCS molar fraction as a
        log-linear interpolation between defined nodes.

        Parameters
        ----------
        altitudes
            Altitude values
            (if not a :class:`~astropy.units.Quantity`, must already be in [km])

        Returns
        -------
            OCS molar fraction
        """
        # interpolate, pad, and add unit
        return Quantity(
            interpolate_nodes(
                cast_to_np(altitudes, "km"), Duan2010Figures.OCS_FRACTION_NODES
            ),
            Duan2010Figures.UNIT_MF,
        )


class James1997(Reference):
    """
    Reference class that provides the reference profiles from :cite:t:`james1997`.
    """

    BASEFOLDER = "james_et_al_1997"
    """ Base folder for data """

    TABLES = ["fig4bdroplets", "fig4bnuclei", "fig7"]
    """ Table numbers to load """

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load raw data files
        self.tables = {
            t: read_unit_csv(datafolder / f"table{t}.csv") for t in self.TABLES
        }

        # combine the two tables for Fig. 4 to get the mass mixing ratio
        # of the liquid part of the clouds
        clouds = join(
            self.tables["fig4bnuclei"],
            self.tables["fig4bdroplets"],
            keys="Altitude",
            join_type="outer",
        ).to_pandas(index="Altitude")
        # move to log space
        clouds = np.log10(clouds)
        # interpolate values in between bounds
        clouds = clouds.interpolate("index", limit_area="inside")
        # back to linear space
        clouds = 10**clouds
        # extrapolate nuclei: to zero above
        clouds.iloc[:, 0] = clouds.iloc[:, 0].fillna(0)
        # calculate liquid ratio
        mmr_name = "mass mixing ratio clouds"
        clouds[mmr_name] = np.fmax(clouds.iloc[:, 1] - clouds.iloc[:, 0], 0)
        # move index back to column for export
        clouds.reset_index(names="altitude", inplace=True)
        # save as QTable with ppm units
        self.tables["clouds"] = QTable.from_pandas(
            clouds[["altitude", mmr_name]],
            units={"altitude": u.Unit("km"), mmr_name: u.Unit("1e-6")},
        )


class JPLSpectralLines(Reference):
    """
    Reference class that provides spectral lines from :cite:t:`pickett1998`.
    """

    BASEFOLDER = "jpl_spectral_lines"
    """ Base folder for data """

    TABLES = {"OCS": "c060001.cat", "SO2": "c064002.cat"}
    """ Tables to load """

    COLUMNS = [
        "FREQ",
        "ERR",
        "LGINT",
        "DR",
        "ELO",
        "GUP",
        "TAG",
        "QNFMT",
        "QN'",
        'QN"',
    ]
    """ Column names in the catalog """

    FORMATS = ["f8", "f8", "f8", "i", "f8", "i", "i", "i", "U12", "U12"]
    """ Column formats """

    WIDTHS = [13, 8, 8, 2, 10, 3, 7, 4, 12, 12]
    """ Fixed widths of the catalog columns """

    UNITS = ["MHz", "MHz", "dex(nm2 MHz)", "", "cm-1", "", "", "", "", ""]
    """ Column units """

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load raw data files
        self.tables = {
            species: read_unit_fwf(
                datafolder / filename,
                names=self.COLUMNS,
                formats=self.FORMATS,
                widths=self.WIDTHS,
                units=self.UNITS,
            )
            for species, filename in self.TABLES.items()
        }


class Keating1985(Reference):
    """
    Reference class that provides the reference profiles from :cite:t:`keating1985`.
    """

    BASEFOLDER = "keating_et_al_1985"
    """ Base folder for data """

    SZA_TABLES = ["4-4", "4-9", "4-10", "4-11", "4-12", "4-13", "4-5"]
    """ Table numbers that build the SZA-defined datacube between 150-250km """

    TABLES = ["4-6", "4-7", "4-15", "4-16"] + SZA_TABLES
    """ Table numbers to load """

    SZA = Quantity([16, 34, 61, 90, 119, 146, 164], "°")
    """ Solar zenith angles for tables 4-[4, 9-13, 5] [°] """

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load raw data files
        self.tables = {
            t: read_unit_csv(datafolder / f"table{t}.csv") for t in self.TABLES
        }

        # combine the three tables that each together build the standard profiles
        # from 100-250km for noon and midnight
        self.tables["day"] = vstack(
            [
                hstack([self.tables["4-4"], self.tables["4-6"]], join_type="exact"),
                self.tables["4-16"][1:],
            ]
        )
        self.tables["day"].sort("ALT")
        self.tables["night"] = vstack(
            [
                hstack([self.tables["4-5"], self.tables["4-7"]], join_type="exact"),
                self.tables["4-15"][1:],
            ]
        )
        self.tables["night"].sort("ALT")

        # build the datacube for 150-250km for the seven samplings of
        # solar zenith angle (which stands in for time)
        self.dcube_150km_250km = np.stack(
            [self.tables[t].as_array().view((float, 11)) for t in self.SZA_TABLES],
            axis=2,
        )
        """
        Physical and species quantities as a function of altitude and
        solar zenith angle between 150-250km
        """
        self.units_150km_250km = [
            self.tables["4-4"].columns[i].unit
            for i in range(self.dcube_150km_250km.shape[1])
        ]
        """ Units of :attr:`~Keating1985.dcube_150km_250km` """
        self.names_150km_250km = self.tables["4-4"].colnames
        """ Column (axis 1) names of of :attr:`~Keating1985.dcube_150km_250km` """

        # build the (smaller) datacube for 100-150km, which only contains
        # noon and midnight
        self.dcube_100km_150km = np.stack(
            [self.tables[t].as_array().view((float, 16)) for t in ["4-16", "4-15"]],
            axis=2,
        )
        """
        Physical and species quantities as a function of altitude and
        midnight/noon between 100-150km
        """
        self.units_100km_150km = [
            self.tables["4-15"].columns[i].unit
            for i in range(self.dcube_100km_150km.shape[1])
        ]
        """ Units of :attr:`~Keating1985.dcube_100km_150km` """
        self.names_100km_150km = self.tables["4-15"].colnames
        """ Column (axis 1) names of of :attr:`~Keating1985.dcube_100km_150km` """


class KolodnerSteffes1998(Reference):
    """
    Reference class that provides the reference profiles from
    :cite:t:`kolodner1998`.
    """

    BASEFOLDER = "kolodner_steffes_1998"
    """ Base folder for data """

    TABLES = ["fig789"]
    """ Table numbers to load """

    # parameters of the experiment
    MU_H2SO4 = Quantity(2.72e-18, ESU_CM)
    """ Molecular dipole moment for gaseous sulfuric acid """
    N_H2SO4 = Quantity((340.64 + 245.36) / 2, "Nunit")
    """ Refractivity of the gaseous sulfuric acid """
    D_H2SO4_L = Quantity(1.8305, "g/ml")
    """ Mass density of the sulfuric acid solution before it evaporates """
    DISS_H2SO4 = 0.461
    """ Dissociation constant of vaporized H2SO4 """
    V_H2SO4 = Quantity((4.12 + 3.18) / 2, "cm3")
    """ Volume of the H2SO4 solution which vaporizes """
    V_VESSEL = Quantity(31, "l")
    """ Volume of the pressure vessel """
    T_H2SO4 = Quantity(553, "K")
    """ Temperature of the experiment of :cite:t:`kolodner1998` """

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load raw data files
        self.tables = {
            t: read_unit_csv(datafolder / f"table{t}.csv") for t in self.TABLES
        }

        # average the three different orbits to have a single H2SO4 abundance
        # profile from X-band
        h2so4_sum = QTable.from_pandas(
            self.tables["fig789"]
            .to_pandas(index="altitude")
            .mean(axis=1)
            .clip(lower=0)
            .to_frame("mixing ratio of H2SO4"),
            units={"mixing ratio of H2SO4": self.tables["fig789"]["3212"].unit},
        )
        self.tables["H2SO4 X-band"] = hstack(
            [self.tables["fig789"]["altitude"], h2so4_sum]
        )

    def get_eps_prime_r_and_molar_density(
        self,
    ) -> Tuple[Quantity["dimensionless"], Quantity["molar concentration"]]:
        """
        Computes the real part of the relative permittivity and the molar density
        of H2SO4 in the experiment as described in Section 3.2.

        Returns
        -------
        eps_prime_r
            Real part of the relative permittivity of H2SO4 in the experiment
        rho
            Molar density of H2SO4 in the experiment
        """
        # real part of the relative permittivity
        eps_prime_r = (self.N_H2SO4.to(u.dimensionless_unscaled)) ** 2
        # Number of moles of pure H2SO4 liquid which vaporizes
        nvap = (self.V_H2SO4 * self.D_H2SO4_L / SPEC_MOL_M["H2SO4"]).decompose()
        # Number of moles of H2SO4 vapor
        nmol = nvap * (1 - self.DISS_H2SO4)
        # Molar density of gaseous sulfuric acid
        rho = (nmol / self.V_VESSEL).to("mol/cm3")
        # done
        return eps_prime_r, rho


class Magellan321X(Reference):
    """
    Reference class that provides the profiles from the Magellan orbits no. 3212,
    3213, and 3214 from :cite:t:`jenkins1996`.
    """

    BASEFOLDER = "magellan321x"
    """ Base folder for data """

    TABLES = ["mgn_abs", "mgn_rtpd"]
    """ Table numbers to load """

    DESC_ABS = [
        ("WAVELENGTH", "", "U1", 3),
        ("ORBIT_NUMBER", "", "i", 5),
        ("ALTITUDE", "km", "f8", 7),
        ("ABSORPTIVITY", "dB/km", "f8", 9),
        ("ABSORP_DEV", "dB/km", "f8", 8),
        ("H2SO4_VOLMIX", "ppm", "f8", 6),
        ("H2SO4_VM_DEV", "ppm", "f8", 6),
        ("LATITUDE", "°", "f8", 7),
        ("LONGITUDE", "°", "f8", 8),
        ("ZENITH_ANGLE", "°", "f8", 8),
        ("LOCAL_TIME", "h", "f8", 7),
        ("ERT", "s", "f8", 10),
    ]
    """ Data format description of the ``mgn_abs.dat`` file """

    DESC_RTPD = [
        ("WAVELENGTH", "", "U1", 3),
        ("ORBIT_NUMBER", "", "i", 5),
        ("ALTITUDE", "km", "f8", 7),
        ("REFRACTIVITY", "Nunit", "f8", 9),
        ("REFRACT_DEV", "Nunit", "f8", 6),
        ("TEMPERATURE", "K", "f8", 7),
        ("TEMP_DEV", "K", "f8", 6),
        ("PRESSURE", "bar", "f8", 9),
        ("PRESS_DEV", "bar", "f8", 9),
        ("DENSITY", "kg/m3", "f8", 8),
        ("DENS_DEV", "kg/m3", "f8", 8),
        ("LATITUDE", "°", "f8", 7),
        ("LONGITUDE", "°", "f8", 8),
        ("ZENITH_ANGLE", "°", "f8", 8),
        ("LOCAL_TIME", "h", "f8", 7),
        ("ERT", "s", "f8", 10),
    ]
    """ Data format description of the ``mgn_rtpd.dat`` file """

    STR_CONVERTER = {0: lambda s: s.strip()}
    """ Convenience converter to strip whitespace from the wavelength field """

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load raw data files
        names_abs, units_abs, formats_abs, widths_abs = zip(*self.DESC_ABS)
        names_rtpd, units_rtpd, formats_rtpd, widths_rtpd = zip(*self.DESC_RTPD)
        self.tables = {
            "mgn_abs": QTable(
                np.genfromtxt(
                    datafolder / "mgn_abs.dat",
                    dtype=formats_abs,
                    delimiter=widths_abs,
                    encoding="utf8",
                    converters=self.STR_CONVERTER,
                ),
                names=names_abs,
                units=units_abs,
            ),
            "mgn_rtpd": QTable(
                np.genfromtxt(
                    datafolder / "mgn_rtpd.dat",
                    dtype=formats_rtpd,
                    delimiter=widths_rtpd,
                    encoding="utf8",
                    converters=self.STR_CONVERTER,
                ),
                names=names_rtpd,
                units=units_rtpd,
            ),
        }


class Marcq2006(Reference):
    """
    Reference class that provides the reference profiles from :cite:t:`marcq2006`.
    """

    BASEFOLDER = "marcq_et_al_2006"
    """ Base folder for data """

    TABLES = ["fig8"]
    """ Table numbers to load """

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load raw data files
        self.tables = {
            t: read_unit_csv(datafolder / f"table{t}.csv") for t in self.TABLES
        }


class Paetzold2007(Reference):
    """
    Reference class that provides the reference profiles from :cite:t:`patzold2007`.
    """

    BASEFOLDER = "paetzold_et_al_2007"
    """ Base folder for data """

    DOYS = [196, 200, 202, 212, 218, 233, 234, 239]
    """ Day-of-years of the electron density profiles """

    SZAS = Quantity([50, 56, 59, 80, 92.4, 113.0, 113.4, 113.5], "°")
    """ Solar zenith angles [°] of the elctron density profiles """

    TABLES = [
        f"fig{t}-doy{d}"
        for t, d in zip(["4a", "4b", "4b", "4b", "5a", "5b", "5c", "5d"], DOYS)
    ]
    """ Table numbers to load """

    UNIT_EL_DENSITY = Unit("1/m3")
    """ Output unit of the electron density profile [1/m3]"""

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load raw data files
        self.tables = {
            t: read_unit_csv(datafolder / f"table{t}.csv") for t in self.TABLES
        }

        # build inter- and extrapolator for first 6 angles
        self.interpolator = BoundedInterpolatingBasis(
            0, 180, knots=self.SZAS[:6].to("°").value
        )

        # build joint datacube for first 6 angles
        # get unique altitude levels
        el_altitude = np.unique(
            np.concatenate([self.tables[t]["Altitude"].value for t in self.TABLES[:6]])
        )
        # interpolate all data onto common altitudes
        el_density = np.zeros((el_altitude.size, 6))
        for i, t in enumerate(self.TABLES[:6]):
            el_density[:, i] = np.interp(
                el_altitude,
                self.tables[t]["Altitude"].value,
                self.tables[t]["Electron density"].to(self.UNIT_EL_DENSITY).value,
                left=0,
                right=None,
            )
        # add units and save
        self.el_altitude = Quantity(el_altitude, self.tables[t]["Altitude"].unit)
        self.el_density = Quantity(el_density, self.UNIT_EL_DENSITY)

    def __call__(self, sza: Quantity["angle"]) -> Quantity:
        """
        Interpolate the profiles from solar zenith angles between 50° and 113°.

        Parameters
        ----------
        sza
            Solar zenith angle [°]

        Returns
        -------
            Electron density profile
        """
        sza_coefs = self.interpolator(sza.to("°").value)
        el_density = Quantity(
            np.einsum(
                "ij,kj->ik",
                np.clip(self.el_density.value, a_min=0, a_max=None),
                sza_coefs,
            ),
            self.el_density.unit,
        )
        return el_density

    @staticmethod
    def get_electron_density(altitudes: NDArray[np.floating] | Quantity) -> Quantity:
        """
        Computes the electron number density as a linear
        interpolation between defined nodes.

        Parameters
        ----------
        altitudes
            Altitude values
            (if not a :class:`~astropy.units.Quantity`, must already be in [km])

        Returns
        -------
            Electron number density
        """
        # interpolate, pad, and add unit
        return Quantity(
            interpolate_nodes(
                cast_to_np(altitudes, "km"),
                Duan2010Figures.ELECTRON_DENSITY_NODES,
                1e11,
                log=True,
            ),
            Duan2010Figures.UNIT_EL_DENSITY,
        )


class Seiff1985(Reference):
    """
    Reference class that provides the model from :cite:t:`seiff1985`.
    """

    BASEFOLDER = "seiff_et_al_1985"
    """ Base folder for data """

    LAT_TABLES = ["1-2a", "1-2b", "1-2c", "1-2d", "1-2e"]
    """ Table numbers of the 33-100km range for different latitudes"""

    TABLES = ["1-1", "1-3"] + LAT_TABLES
    """ Table numbers to load """

    LAT = Quantity([30, 45, 60, 75, 85], "°")
    """ Latitudes of tables 1-2[a-e]"""

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load raw data files
        self.tables = {
            t: read_unit_csv(datafolder / f"table{t}.csv") for t in self.TABLES
        }

        # build the datacube for 33-100km for the five samplings of latitude
        self.dcube_33km_100km = np.stack(
            [
                self.tables[t].as_array().view((float, 8 if t == "1-2a" else 4))[:, :4]
                for t in self.LAT_TABLES
            ],
            axis=2,
        )
        """
        Physical quantities as a function of altitude and
        latitude between 33-100km
        """
        self.units_33km_100km = [
            self.tables["1-2b"].columns[i].unit
            for i in range(self.dcube_33km_100km.shape[1])
        ]
        """ Units of :attr:`~Seiff1985.dcube_33km_100km` """
        self.names_33km_100km = self.tables["1-2b"].colnames
        """ Column (axis 1) names of of :attr:`~Seiff1985.dcube_33km_100km` """


class Zasova2006(Reference):
    """
    Reference class that provides the model from :cite:t:`zasova2006`.
    """

    BASEFOLDER = "zasova_et_al_2006"
    """ Base folder for data """

    TABLES = ["2", "3", "4", "5", "6"]
    """ Table numbers to load """

    def __init__(self) -> None:
        """
        Initialize the model from the raw data.
        """
        # parent class
        super().__init__()

        # get local installation folder paths
        datafolder = res_files(data) / self.BASEFOLDER

        # load raw data files
        self.tables = {
            t: read_unit_csv(datafolder / f"table{t}.csv") for t in self.TABLES
        }


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


def stratton1968(
    T: Quantity["temperature"],
    P_CO2: Quantity["pressure"],
    P_N2: Quantity["pressure"],
    P_H2O: Quantity["pressure"],
) -> Quantity:
    """
    Compute the refractivity from the fitted function in :cite:t:`stratton1968`.

    Parameters
    ----------
    T
        Temperature [K]
    P_CO2
        Partial pressure of CO2 [mbar]
    P_N2
        Partial pressure of N2 [mbar]
    P_H2O
        Partial pressure of H2O [mbar]

    Returns
    -------
        Refractivity [Nunit]
    """
    # readability
    T_K = T.to("K").value
    # compute
    refractivity = (
        134.9 * P_CO2.to("mbar").value / T_K
        + 80.29 * P_N2.to("mbar").value / T_K
        + 16.57 * (1 + 5748 / T_K) * P_H2O.to("mbar").value / T_K
    )
    # return with units
    return Quantity(refractivity, "Nunit")


# load datasets
cimino1982 = Cimino1982()
""" Preloaded :class:`~Cimino1982` dataset """
duan2010figures = Duan2010Figures()
""" Preloaded :class:`~Duan2010Figures` dataset """
james1997 = James1997()
""" Preloaded :class:`~James1997` dataset """
jplspectrallines = JPLSpectralLines()
""" Preloaded :class:`~JPLSpectralLines` dataset """
keating1985 = Keating1985()
""" Preloaded :class:`~Keating1985` dataset """
kolodnersteffes1998 = KolodnerSteffes1998()
""" Preloaded :class:`~KolodnerSteffes1998` dataset """
magellan321x = Magellan321X()
""" Preloaded :class:`~Magellan321X` dataset """
marcq2006 = Marcq2006()
""" Preloaded :class:`~Marcq2006` dataset """
paetzold2007 = Paetzold2007()
""" Preloaded :class:`~Paetzold2007` dataset """
seiff1985 = Seiff1985()
""" Preloaded :class:`~Seiff1985` dataset """
zasova2006 = Zasova2006()
""" Preloaded :class:`~Zasova2006` dataset """
seiffkeating = SeiffKeating(seiff1985, keating1985)
""" Preloaded :class:`~SeiffKeating` dataset """
