"""This file contains source functions to calculate scaling relations presented in Wells and
Coppersmith (1994).

Reference: https://doi.org/10.1785/BSSA0840040974

# NOTE: Only a small subset of the models are implemented herein currently.
"""

import numpy as np


def _calc_distrib_params_mag_ad(*, magnitude, style="all"):
    """
    Calculate mu and sigma for the AD=f(M) relations in Wells & Coppersmith (1994) Table 2B.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    style : str, optional
        Style of faulting (case-insensitive). Default is "all". Valid options are "strike-slip",
        "reverse", "normal", or "all".

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
    style = style.lower()

    coeffs = {
        "all": (-4.8, 0.69, 0.36),
        "strike-slip": (-6.32, 0.90, 0.28),
        "reverse": (-0.74, 0.08, 0.38),
        "normal": (-4.45, 0.63, 0.33),
    }

    a, b, sigma = coeffs[style]
    mu = a + b * magnitude

    return mu, np.full(len(mu), sigma)


def _calc_distrib_params_mag_md(*, magnitude, style="all"):
    """
    Calculate mu and sigma for the MD=f(M) relations in Wells & Coppersmith (1994) Table 2B.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    style : str, optional
        Style of faulting (case-insensitive). Default is "all". Valid options are "strike-slip",
        "reverse", "normal", or "all".

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
    style = style.lower()

    coeffs = {
        "all": (-5.46, 0.82, 0.42),
        "strike-slip": (-7.03, 1.03, 0.34),
        "reverse": (-1.84, 0.29, 0.42),
        "normal": (-5.90, 0.89, 0.38),
    }

    a, b, sigma = coeffs[style]
    mu = a + b * magnitude

    return mu, np.full(len(mu), sigma)
