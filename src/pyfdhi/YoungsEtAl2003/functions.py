"""This file contains source functions to calculate scaling relations presented in Youngs et al.
(2003).

Reference: https://doi.org/10.1193/1.1542891

# NOTE: The D/MD relationships are not implemented herein because the results in Figures 6 and 7a
in Youngs et al. (2003) cannot be reproduced from the formulations and coefficients in the
appendix of the paper.

"""

import numpy as np


def _calc_distrib_params_d_ad(*, location):
    """
    Calculate alpha and beta per Appendix based on location.

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

    a1, a2 = 1.628, -0.193
    b1, b2 = -0.476, 0.009

    alpha = np.exp(a1 * folded_location + a2)
    beta = np.exp(b1 * folded_location + b2)

    return alpha, beta
