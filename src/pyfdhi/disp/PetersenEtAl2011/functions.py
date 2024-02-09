"""This file contains source functions to calcualte fault displacement using the Petersen et al.
(2011) model.

Reference: https://doi.org/10.1785/0120100035

# NOTE: Only the principal fault displacement models for direct (i.e., not normalized) predictions
are implemented herein currently.
"""

# Python imports
import numpy as np


def _calc_xstar(*, location):
    """
    Calculate elliptical position scaling ("xstar") given location.

    Parameters
    ----------
    location : Union[float, np.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Union[float, np.ndarray]
        Calculated xstar value.
    """

    return np.sqrt(1 - (1 / (0.5**2)) * np.power(location - 0.5, 2))


def _calc_distrib_params_elliptical(*, magnitude, location):
    """
    Calculate mu and sigma per Eqn 13 based on magnitude and location.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    location : Union[float, np.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[float, float]
        mu : Mean prediction.
        sigma : Total standard deviation.

    Notes
    ------
    Mu and sigma are in natural log units. Exp(mu) is in centimeters, not meters.
    """

    # Convert inputs to list-like numpy arrays
    magnitude, location = map(np.atleast_1d, (magnitude, location))

    a, b, c = 1.7927, 3.3041, -11.2192
    sigma = 1.1348

    xstar = _calc_xstar(location=location)
    mu = b * xstar + a * magnitude + c

    return mu, np.full(len(mu), sigma)


def _calc_distrib_params_quadratic(*, magnitude, location):
    """
    Calculate mu and sigma per Eqn 10 based on magnitude and location.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    location : Union[float, np.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[float, float]
        mu : Mean prediction.
        sigma : Total standard deviation.

    Notes
    ------
    Mu and sigma are in natural log units. Exp(mu) is in centimeters, not meters.
    """

    # Convert inputs to list-like numpy arrays
    magnitude, location = map(np.atleast_1d, (magnitude, location))

    folded_location = np.minimum(location, 1 - location)

    a, b, c, d = 1.7895, 14.4696, -20.1723, -10.54512
    sigma = 1.1346

    mu = a * magnitude + b * folded_location + c * np.power(folded_location, 2) + d

    return mu, np.full(len(mu), sigma)


def _calc_distrib_params_bilinear(*, magnitude, location):
    """
    Calculate mu and sigma per Eqns 7 & 8 based on magnitude and location.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    location : Union[float, np.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[float, float]
        mu : Mean prediction.
        sigma : Total standard deviation.

    Notes
    ------
    Mu and sigma are in natural log units. Exp(mu) is in centimeters, not meters.
    """

    # Convert inputs to list-like numpy arrays
    magnitude, location = map(np.atleast_1d, (magnitude, location))

    folded_location = np.minimum(location, 1 - location)

    a1, b, c1, a2, c2 = 1.7969, 8.5206, -10.2855, 1.7658, -7.8962
    sigma1, sigma2 = 1.2906, 0.9624

    l_L_prime = 1 / b * ((a2 - a1) * magnitude + (c2 - c1))

    mu = np.where(
        folded_location < l_L_prime,
        a1 * magnitude + b * folded_location + c1,
        a2 * magnitude + c2,
    )
    sigma = np.where(folded_location < l_L_prime, sigma1, sigma2)

    return mu, sigma
