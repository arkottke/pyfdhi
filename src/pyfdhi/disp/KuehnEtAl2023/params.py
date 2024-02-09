"""This file contains the helper functions to calculate the statistical distribution parameters
(i.e., mu and sigma).

# NOTE: The `_calculate_distribution_parameters` function is called in the main
functions in `run_model()` and `run_probex()`.

Reference: https://doi.org/10.1177/ToBeAssigned
"""

import numpy as np

# Module imports
import pyfdhi.disp.KuehnEtAl2023.model_config as model_config  # noqa: F401
from pyfdhi.disp.KuehnEtAl2023.data import POSTERIOR, POSTERIOR_MEAN
from pyfdhi.disp.KuehnEtAl2023.functions import func_nm, func_rv, func_ss


def _calculate_distribution_parameters(*, magnitude, location, style, mean_model):
    """
    A vectorized helper function to calculate predicted mean and standard deviation in transformed
    units and the Box-Cox transformation parameter.

    Parameters
    ----------
    magnitude : np.array
        Earthquake moment magnitude.

    location : np.array
        Normalized location along rupture length, range [0, 1.0].

    style : np.array
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    mean_model : bool
        If True, use mean coefficients. If False, use full coefficients.

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array]
        mu : Mean prediction in transformed units.
        sigma : Total standard deviation in transformed units.
        lam : Box-Cox transformation parameter.
        model_num : Model coefficient row number. Returns -1 for mean model.
    """

    if mean_model:
        # Calculate for all submodels
        # NOTE: it is actually faster to just do this instead of if/else, loops, etc.

        # Define coefficients (loaded with module imports)
        # NOTE: Coefficients are pandas dataframes; convert here to recarray for faster computations
        # NOTE: Check for appropriate style is handled in `run_model`
        mean_coeffs_ss = POSTERIOR_MEAN.get("strike-slip").to_records(index=False)
        mean_coeffs_rv = POSTERIOR_MEAN.get("reverse").to_records(index=False)
        mean_coeffs_nm = POSTERIOR_MEAN.get("normal").to_records(index=False)

        result_ss = func_ss(mean_coeffs_ss, magnitude, location)
        result_rv = func_rv(mean_coeffs_rv, magnitude, location)
        result_nm = func_nm(mean_coeffs_nm, magnitude, location)

        lam_ss = mean_coeffs_ss["lambda"]
        lam_rv = mean_coeffs_rv["lambda"]
        lam_nm = mean_coeffs_nm["lambda"]

        model_num_ss = mean_coeffs_ss["model_number"]
        model_num_rv = mean_coeffs_rv["model_number"]
        model_num_nm = mean_coeffs_nm["model_number"]

        # Conditions for np.select
        conditions = [
            style == "strike-slip",
            style == "reverse",
            style == "normal",
        ]

        # Choices for mu and sigma
        choices_mu = [result_ss[0], result_rv[0], result_nm[0]]
        choices_sigma = [result_ss[1], result_rv[1], result_nm[1]]
        choices_lam = [lam_ss, lam_rv, lam_nm]
        choices_model_num = [model_num_ss, model_num_rv, model_num_nm]

        # Use np.select to get the final mu, sigma, and lambda
        mu = np.select(conditions, choices_mu, default=np.nan)
        sigma = np.select(conditions, choices_sigma, default=np.nan)
        lam = np.select(conditions, choices_lam, default=np.nan)
        model_num = np.select(conditions, choices_model_num, default=np.nan)

        return mu, sigma, lam, model_num

    else:
        # NOTE: Check for appropriate style is handled in `run_model`
        function_map = {"strike-slip": func_ss, "reverse": func_rv, "normal": func_nm}

        # NOTE: use instead of style[0] as another way to check only one style in list; #TODO make this a try/except?
        s = "".join(style)
        model = function_map[s]

        # Define coefficients (loaded with module imports)
        # NOTE: Coefficients are pandas dataframes; convert here to recarray for faster computations
        coeffs = POSTERIOR.get(s).to_records(index=False)

        mu, sigma = model(coeffs, magnitude, location)
        lam = coeffs["lambda"]
        model_num = coeffs["model_number"]

        return mu, sigma, lam, model_num
