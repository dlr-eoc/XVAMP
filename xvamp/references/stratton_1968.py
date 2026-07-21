"""
Module that provides the background, reference atmospheric properties.
"""

# standard imports
from astropy.units import Quantity

# no data files to load


def get_refractivity(
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
