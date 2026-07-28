"""
Module that provides the reference profiles from :cite:t:`cimino1982`,
as well as interpolators for the profiles.
"""

# standard imports
from typing import Literal
import numpy as np
from importlib.resources import files as res_files
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table, QTable, hstack
from scipy.interpolate import LinearNDInterpolator, Akima1DInterpolator

# data content
TABLES = [f"fig{n}raw" for n in ["7", "8", "9"]]
""" Table numbers to load """

# values from the paper
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

# get local installation folder paths
datafolder = res_files()
""" Resolved current folder location """
# load super-raw data files
tables = {t: Table.read(datafolder / f"table{t}.csv") for t in TABLES}
""" Loaded tables """


# function to build interpolators
def _get_interpolators() -> tuple[
    LinearNDInterpolator,
    LinearNDInterpolator,
    LinearNDInterpolator,
    LinearNDInterpolator,
]:

    # rename and join the two tables for the imaginary part
    t_real = tables["fig7raw"]
    t_imaginary = hstack([tables["fig8raw"], tables["fig9raw"]])

    # get all different temperatures for each band (same for real & imaginary)
    temps = {
        "s": np.unique([cn[2:-2] for cn in t_real.colnames if cn[0] == "s"]),
        "x": np.unique([cn[2:-2] for cn in t_real.colnames if cn[0] == "x"]),
    }

    # make readable tables
    x_y_pairs = {}
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
                list_raw.append(np.concatenate([raw, new_vals.reshape(1, 3)], axis=0))
            # extrapolate to a lower temperature
            subtemps = np.array([float(t) for t in temps[band][:3]])
            local_interps = []
            for itemp in range(3):
                local_interps.append(
                    Akima1DInterpolator(list_raw[itemp][:, 0], list_raw[itemp][:, 2])
                )
            concs = list_raw[0][:, 0]
            interp_vals = np.stack([interp(concs) for interp in local_interps], axis=1)
            new_vals = np.clip(
                np.array(
                    [
                        np.polynomial.Polynomial.fit(
                            subtemps, interp_vals[ic, :], deg=2
                        )(EXTRAPOLATE_DOWN_TO)
                        for ic in range(concs.size)
                    ]
                ),
                a_min=0,
                a_max=None,
            )
            new_table = np.stack(
                [concs, np.full_like(concs, EXTRAPOLATE_DOWN_TO), new_vals],
                axis=1,
            )
            list_raw.insert(0, new_table)
            # concatenate all temperatures
            raw_concat = np.concatenate(list_raw, axis=0)
            assert np.all(np.isfinite(raw_concat))
            # combine the different columns into a QTable with units
            tables[f"{part.lower()}_{band}"] = QTable(
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
            x_y_pairs[(part, band)] = (
                tables[f"{part.lower()}_{band}"]["Concentration H2SO4", "Temperature"]
                .as_array()
                .view((float, 2)),
                tables[f"{part.lower()}_{band}"][f"{part} Dielectric Constant"].value,
            )

    # build 4 2D interpolators, one for each wavelength and part,
    # which will be accessed by the global interpolator
    get_real_sband = LinearNDInterpolator(*x_y_pairs[("Real", "s")])
    get_real_xband = LinearNDInterpolator(*x_y_pairs[("Real", "x")])
    get_imaginary_sband = LinearNDInterpolator(*x_y_pairs[("Imaginary", "s")])
    get_imaginary_xband = LinearNDInterpolator(*x_y_pairs[("Imaginary", "x")])

    # done
    return get_real_sband, get_real_xband, get_imaginary_sband, get_imaginary_xband


# get the individual interpolators
_get_real_sband, _get_real_xband, _get_imaginary_sband, _get_imaginary_xband = (
    _get_interpolators()
)


# define general interpolating function
def get_h2s04_rel_permittivity(
    concentration: Quantity,
    temperature: Quantity,
    wavelength: Literal["s", "x"] | Quantity = "x",
) -> tuple[Quantity, Quantity]:
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
        Real part of the relative permittivity [-]
    eps_dprime_r
        Imaginary part of the relative permittivity [-]

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
        eps_prime_r = _get_real_sband(ct)
        eps_dprime_r = _get_imaginary_sband(ct)
        return (
            Quantity(eps_prime_r, u.dimensionless_unscaled),
            Quantity(eps_dprime_r, u.dimensionless_unscaled),
        )
    elif wavelength == "x":
        # evaluate the X-band interpolators and return
        eps_prime_r = _get_real_xband(ct)
        eps_dprime_r = _get_imaginary_xband(ct)
        return (
            Quantity(eps_prime_r, u.dimensionless_unscaled),
            Quantity(eps_dprime_r, u.dimensionless_unscaled),
        )
    else:
        # convert specific wavelength to cm
        l = wavelength.to("cm").value
        # evaluate all interpolators
        real_s = _get_real_sband(ct)
        real_x = _get_real_xband(ct)
        imaginary_s = _get_imaginary_sband(ct)
        imaginary_x = _get_imaginary_xband(ct)
        # get wavelength slopes
        delta_lambda = LAMBDA_S - LAMBDA_X
        real_slope = (real_s - real_x) / delta_lambda
        imaginary_slope = (imaginary_s - imaginary_x) / delta_lambda
        # extrapolate
        eps_prime_r = Quantity(
            real_slope * (l - LAMBDA_X) + real_x,
            u.dimensionless_unscaled,
        )
        eps_dprime_r = Quantity(
            imaginary_slope * (l - LAMBDA_X) + imaginary_x,
            u.dimensionless_unscaled,
        )
        # done
        return eps_prime_r, eps_dprime_r


# provide the equations to compute the polarization of the droplets
def get_h2so4_droplet_polarization_volfrac(
    cloud_mass_density: Quantity, eps_r_shell: Quantity
) -> Quantity:
    """
    Compute the droplet polarization and volume fraction assuming a
    shell-like structure, following eq. (10).

    Parameters
    ----------
    cloud_mass_density
        Cloud mass density profile [kg^3/m^3]
    eps_r_shell
        Complex relative permittivity of the shell [-]

    Returns
    -------
    Pnu
        Polarization per molar volume of the H2SO4 cloud droplets [-]
    vol_frac_droplets
        Volume fraction of the droplets [-]

    Note
    ----
    Here, the imaginary part of the relative atmospheric permittivity
    has the opposite sign as in :cite:t:`duan2010`.
    """
    # readability
    eps_c_plus_2s = EPS_DROPLETS_CORE + 2 * eps_r_shell
    eps_c_minus_s = EPS_DROPLETS_CORE - eps_r_shell
    q3 = RATIO_RADIUS_DROPLETS**3
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
    # calculate mass of shell and core
    d_shell = cloud_mass_density / (1 + RATIO_MASS_DROPLETS)
    d_core = RATIO_MASS_DROPLETS * d_shell
    vol_frac_droplets = (
        d_core / D_DROPLET_CORE + d_shell / D_DROPLET_SHELL
    ).decompose()
    return Pnu, vol_frac_droplets
