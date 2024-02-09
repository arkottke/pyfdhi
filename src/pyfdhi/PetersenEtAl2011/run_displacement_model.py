"""This file runs the PEA11 principal fault displacement model.
- Any number of scenarios are allowed (e.g., user can enter multiple magnitudes).
- The results are returned in a pandas DataFrame.
- Only the principal fault displacement models for direct (i.e., not normalized) predictions are
implemented herein currently.
- Command-line use is supported; try `python run_displacement_model.py --help`
- Module use is supported; try `from run_displacement_model import run_model`

Reference: https://doi.org/10.1785/0120100035
"""

# Python imports
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import product
from scipy import stats
from typing import Union, List

# Module imports
import pyfdhi.PetersenEtAl2011.model_config as model_config  # noqa: F401
from pyfdhi.PetersenEtAl2011.functions import (
    _calc_distrib_params_elliptical,
    _calc_distrib_params_quadratic,
    _calc_distrib_params_bilinear,
)


def _calc_distrib_params(*, magnitude, location, submodel):
    """
    A vectorized helper function to calculate predicted distribution parameters.

    Parameters
    ----------
    magnitude : numpy.ndarray
        Earthquake moment magnitude.

    location : numpy.ndarray
        Normalized location along rupture length, range [0, 1.0].

    submodel : str
        PEA11 shape model name. Valid options are "elliptical", "quadratic", or "bilinear".

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        mu : Mean prediction.
        sigma : Total standard deviation.

    Notes
    ------
    Mu and sigma are in natural log units. Exp(mu) is in centimeters, not meters.
    """

    # Calculate for all submodels
    # NOTE: it is actually faster to just do this instead of if/else, loops, etc.
    result_elliptical = _calc_distrib_params_elliptical(
        magnitude=magnitude, location=location
    )
    result_quadratic = _calc_distrib_params_quadratic(
        magnitude=magnitude, location=location
    )
    result_bilinear = _calc_distrib_params_bilinear(
        magnitude=magnitude, location=location
    )

    # Conditions for np.select
    conditions = [
        submodel == "elliptical",
        submodel == "quadratic",
        submodel == "bilinear",
    ]

    # Choices for mu and sigma
    choices_mu = [result_elliptical[0], result_quadratic[0], result_bilinear[0]]
    choices_sigma = [result_elliptical[1], result_quadratic[1], result_bilinear[1]]

    # Use np.select to get the final mu and sigma
    mu = np.select(conditions, choices_mu, default=np.nan)
    sigma = np.select(conditions, choices_sigma, default=np.nan)

    return mu, sigma


def run_model(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    location: Union[float, int, List[Union[float, int]], np.ndarray],
    percentile: Union[float, int, List[Union[float, int]], np.ndarray],
    submodel: Union[str, List[str], np.ndarray] = "elliptical",
    style: Union[str, List[str], np.ndarray] = "strike-slip",
    debug_bilinear_model: bool = False,
) -> pd.DataFrame:
    """
    Run PEA11 principal fault displacement model. All parameters must be passed as keyword
    arguments. Any number of scenarios (i.e., magnitude inputs, location inputs, etc.) are allowed.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    location : Union[float, list, numpy.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    percentile : Union[float, list, numpy.ndarray]
        Aleatory quantile value. Use -1 for mean.

    submodel : Union[str, list, numpy.ndarray], optional
        PEA11 shape model name (case-insensitive). Default is "elliptical". Valid options are
        "elliptical", "quadratic", or "bilinear".

    style : Union[str, list, numpy.ndarray], optional
        Style of faulting (case-insensitive). Default is "strike-slip".

    debug_bilinear_model : bool, optional
        If True, bilinear model will run for any percentile with a UserWarning. If False, bilinear
        model results will be dropped for every percentile except median. Default False.

        # NOTE: There is an issue with the bilinear model. The standard deviation changes across
        ... l/L' and Figure 5b in PEA11 cannot be reproduced.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Aleatory quantile value [from user input].
        - 'model_name': Profile shape model name [from user input].
        - 'mu': Natural log transform of mean displacement in cm.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'displ': Displacement in meters.

    Raises
    ------
    ValueError
        If invalid `submodel` is provided.

    Warns
    -----
    UserWarning
        If an unsupported `style` is provided. The user input will be over-ridden with "strike-slip".

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_displacement_model.py --magnitude 7 --location 0.5 --percentile 0.5 --submodel quadratic elliptical`
        Run `python run_displacement_model.py --help`

    #TODO
    ------
    Raise a ValueError for invalid location
    Raise a ValueError for invalid percentile.
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # Check for allowable styles, then over-ride
    if style not in ("strike-slip", "Strike-Slip"):
        warnings.warn(
            f"This model is only recommended for strike-slip faulting, but '{style}' was entered."
            f"User input will be over-ridden.",
            category=UserWarning,
        )
        style = "strike-slip"

    # Check if there are any invalid submodels
    submodel = [
        x.lower() for x in ([submodel] if isinstance(submodel, str) else submodel)
    ]
    supported_submodels = ["elliptical", "quadratic", "bilinear"]
    invalid_mask = ~np.isin(submodel, supported_submodels)

    if np.any(invalid_mask):
        invalid_submodels = np.asarray(submodel)[invalid_mask]
        raise ValueError(
            f"Invalid submodel names: {invalid_submodels}. Supported submodels are {supported_submodels}."
        )

    # Vectorize scenarios
    scenarios = product(
        [magnitude] if not isinstance(magnitude, (list, np.ndarray)) else magnitude,
        [location] if not isinstance(location, (list, np.ndarray)) else location,
        [percentile] if not isinstance(percentile, (list, np.ndarray)) else percentile,
        [submodel] if not isinstance(submodel, (list, np.ndarray)) else submodel,
    )
    magnitude, location, percentile, submodel = map(np.array, zip(*scenarios))

    # Calculate distribution parameters
    mu, sigma = _calc_distrib_params(
        magnitude=magnitude, location=location, submodel=submodel
    )

    # Calculate natural log of displacement (vectorized approach)
    if np.any(percentile == -1):
        # Compute the mean
        ln_displ_mean = mu + np.power(sigma, 2) / 2
    else:
        ln_displ_mean = np.nan

    # Compute the aleatory quantile
    ln_displ_normal = stats.norm.ppf(percentile, loc=mu, scale=sigma)

    # Use np.where for vectorization
    ln_displ = np.where(percentile == -1, ln_displ_mean, ln_displ_normal)

    # NOTE: exp(mu) is in centimeters, must be converted to  meters.
    D = np.exp(ln_displ) / 100

    # Create a DataFrame
    n = len(magnitude)
    results = (
        magnitude,
        location,
        np.full(n, style),
        percentile,
        np.full(n, submodel),
        mu,
        sigma,
        D,
    )

    cols_dict = {
        "magnitude": float,
        "location": float,
        "style": str,
        "percentile": float,
        "model_name": str,
        "mu": float,
        "sigma": float,
        "displ": float,
    }
    dataframe = pd.DataFrame(np.column_stack(results), columns=cols_dict.keys())
    dataframe = dataframe.astype(cols_dict)

    # FIXME: There is an issue with the bilinear model. The standard deviation changes across ...
    # ... l/L' and Figure 5b in PEA11 cannot be reproduced; issue UserWarning on this.
    if np.any(submodel == "bilinear") and np.any(percentile != 0.5):
        if debug_bilinear_model:
            warnings.warn(
                "#FIXME: Don't use the bilinear model for anything except median 50th percentile; current code does not reproduce Fig 5b in P11."
            )
        else:
            warnings.warn(
                "#FIXME: There is a problem with how the bilinear model is currently implemented for anything except median 50th percentile, so these have been dropped from your analysis."
            )
            dataframe = dataframe[
                ~(
                    dataframe["model_name"].str.contains("bilinear")
                    & (dataframe["percentile"] != 0.5)
                )
            ].reset_index(drop=True)

    return dataframe


def main():
    description_text = """Run PEA11 principal fault displacement model. Any number of scenarios are
    allowed (e.g., user can enter multiple magnitudes or locations).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Aleatory quantile value [from user input].
        - 'model_name': Profile shape model name [from user input].
        - 'mu': Natural log transform of mean displacement in cm.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'displ': Displacement in meters.
    """

    parser = argparse.ArgumentParser(
        description=description_text, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--magnitude",
        required=True,
        nargs="+",
        type=float,
        help="Earthquake moment magnitude.",
    )
    parser.add_argument(
        "-l",
        "--location",
        required=True,
        nargs="+",
        type=float,
        help="Normalized location along rupture length, range [0, 1.0].",
    )
    parser.add_argument(
        "-p",
        "--percentile",
        required=True,
        nargs="+",
        type=float,
        help=" Aleatory quantile value. Use -1 for mean.",
    )
    parser.add_argument(
        "-shape",
        "--submodel",
        default="elliptical",
        nargs="+",
        type=str.lower,
        choices=("elliptical", "quadratic", "bilinear"),
        help="PEA11 shape model name (case-insensitive). Default is 'elliptical'.",
    )
    parser.add_argument(
        "-s",
        "--style",
        default="strike-slip",
        nargs="+",
        type=str.lower,
        help="Style of faulting (case-insensitive). Default is 'strike-slip'; other styles not recommended.",
    )

    # FIXME: bilinear model debugger issue
    parser.add_argument(
        "--debug",
        dest="debug_bilinear_model",
        action="store_true",
        help="Return bilinear results that are erroneous for debugging purposes.",
        default=False,
    )

    args = parser.parse_args()

    magnitude = args.magnitude
    location = args.location
    percentile = args.percentile
    submodel = args.submodel
    style = args.style
    debug = args.debug_bilinear_model  # FIXME: bilinear model debugger issue

    try:
        results = run_model(
            magnitude=magnitude,
            location=location,
            percentile=percentile,
            submodel=submodel,
            style=style,
            debug_bilinear_model=debug,  # FIXME: bilinear model debugger issue
        )
        print(results)

        # Prompt to save results to CSV
        save_option = (
            input("Do you want to save the results to a CSV (yes/no)? ").strip().lower()
        )

        if save_option in ["y", "yes"]:
            file_path = input("Enter filepath to save results: ").strip()
            if file_path:
                # Create the directory if it doesn't exist
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                results.to_csv(file_path, index=False)
                print(f"Results saved to {file_path}")
            else:
                print("Invalid file path. Results not saved.")
        else:
            print("Results not saved.")

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
