"""This file contains source functions to calculate scaling relations presented in Moss and Ross
(2011).

Reference: https://doi.org/10.1785/0120100248
"""

import numpy as np


def _calc_distrib_params_mag_ad(*, magnitude):
    """
    Calculate mu and sigma for the AD=f(M) relation in Moss & Ross (2011) Eqn 8.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    Returns
    -------
    Tuple[np.array, np.array]
        mu : Mean prediction in log10 units.
        sigma : Standard deviation in log10 units.

    Notes
    ------
    Mu and sigma are in log10 units
    """

    magnitude = np.atleast_1d(magnitude)

    a, b, sigma = -2.2192, 0.3244, 0.17
    mu = a + b * magnitude

    return mu, np.full(len(mu), sigma)


def _calc_distrib_params_mag_md(*, magnitude):
    """
    Calculate mu and sigma for the MD=f(M) relation in Moss & Ross (2011) Eqn 9.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    Returns
    -------
    Tuple[np.array, np.array]
        mu : Mean prediction in log10 units.
        sigma : Standard deviation in log10 units.

    Notes
    ------
    Mu and sigma are in log10 units
    """
    magnitude = np.atleast_1d(magnitude)

    a, b, sigma = -3.1971, 0.5102, 0.31
    mu = a + b * magnitude

    return mu, np.full(len(mu), sigma)


def _calc_distrib_params_d_ad(*, location):
    """
    Calculate alpha and beta per Eqn 7 based on location.

    Parameters
    ----------
    location : Union[float, np.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[float, float]
        alpha : Shape parameter for Gamma distribution.
        beta : Scale parameter for Gamma distribution.

    """

    location = np.atleast_1d(location)

    folded_location = np.minimum(location, 1 - location)

    alpha = np.exp(
        -30.4 * folded_location**3 + 19.9 * folded_location**2 - 2.29 * folded_location + 0.574
    )
    beta = np.exp(
        50.3 * folded_location**3 - 34.6 * folded_location**2 + 6.6 * folded_location - 1.05
    )

    return alpha, beta


def _calc_distrib_params_d_md(*, location):
    """
    Calculate alpha and beta per Eqn "7.5" based on location. (Eqn # number is missing in
    manuscript, but it is the MD formulation between Eqns 7 and 8 on page 1547.)

    Parameters
    ----------
    location : Union[float, np.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[float, float]
        alpha : Shape parameter for Beta distribution.
        beta : Shape parameter for Beta distribution.

    """

    location = np.atleast_1d(location)

    folded_location = np.minimum(location, 1 - location)

    a1, a2 = 0.901, 0.713
    b1, b2 = -1.86, 1.74

    alpha = a1 * folded_location + a2
    beta = b1 * folded_location + b2

    return alpha, beta
