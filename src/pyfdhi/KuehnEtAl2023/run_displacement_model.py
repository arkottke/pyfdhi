"""This file runs the KEA23 displacement model.
- The results are returned in a pandas DataFrame.
- Results for the location, its complement, and folded location are always returned.
- The mean model (i.e., mean coefficients) is run by default, but results for all coefficients can be computed.
- If full model is run (i.e., `mean_model=False`), then only one scenario is allowed.
- A scenario is defined as a magnitude-location-style-percentile combination.
- If mean model is run (i.e., `mean_model=True` or default), then any number of scenarios is allowed.
- Command-line use is supported; try `python run_displacement_model.py --help`
- Module use is supported; try `from run_displacement_model import run_model`

# NOTE: Several helper functions are defined herein, but the main function is `run_model()`.

Reference: https://doi.org/10.1177/ToBeAssigned
"""

# Python imports
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import product
from scipy import stats
from typing import Union, List

# Module imports
import pyfdhi.KuehnEtAl2023.model_config as model_config  # noqa: F401
from pyfdhi.KuehnEtAl2023.params import _calculate_distribution_parameters


def _calculate_Y(*, mu, sigma, lam, percentile):
    """
    A vectorized helper function to calculate predicted displacement in transformed units.

    Parameters
    ----------
    mu : np.array
        Mean prediction in transformed units.

    sigma : np.array
        Total standard deviation in transformed units.

    lam : np.array
        "Lambda" transformation parameter in Box-Cox transformation.

    percentile : np.array
        Aleatory quantile value. Use -1 for mean.

    Returns
    -------
    Y : np.array
        Predicted displacement in transformed units.
    """
    if np.any(percentile == -1):
        # Compute the mean
        # NOTE: Analytical solution from https://robjhyndman.com/hyndsight/backtransforming/
        D_mean = (np.power(lam * mu + 1, 1 / lam)) * (
            1 + (np.power(sigma, 2) * (1 - lam)) / (2 * np.power(lam * mu + 1, 2))
        )
        # NOTE: Analytical soluion is in meters, so convert back to Y transform for consistency
        Y_mean = (np.power(D_mean, lam) - 1) / lam
    else:
        Y_mean = np.nan

    # Compute the aleatory quantile
    Y_normal = stats.norm.ppf(percentile, loc=mu, scale=sigma)

    # Use np.where for vectorization
    Y = np.where(percentile == -1, Y_mean, Y_normal)

    return Y


def _calculate_displacement(*, predicted_Y, lam):
    """
    A vectorized helper function to calculate predicted displacement in meters.

    Parameters
    ----------
    predicted_Y : np.array
        Predicted displacement in transformed units.

    lam : np.array
        "Lambda" transformation parameter in Box-Cox transformation.

    Returns
    -------
    D : np.array
        Predicted displacement in meters.
    """

    D = np.power(predicted_Y * lam + 1, 1 / lam)

    # Handle values that are too small to calculate
    D = np.where(np.isnan(D), 0, D)

    return D


def run_model(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    location: Union[float, int, List[Union[float, int]], np.ndarray],
    style: Union[str, List[str], np.ndarray],
    percentile: Union[str, List[str], np.ndarray],
    mean_model: bool = True,
) -> pd.DataFrame:
    """
    Run KEA23 displacement model. All parameters must be passed as keyword arguments.
    A couple "gotchas":
        If full model is run (i.e., `mean_model=False`), then only one scenario is allowed.
        If mean model is run (i.e., `mean_model=True` or default), then any number of scenarios is allowed.
        A scenario is defined as a magnitude-location-style-percentile combination.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    location : Union[float, list, numpy.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    style : Union[str, list, numpy.ndarray]
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    percentile : Union[float, list, numpy.ndarray]
        Aleatory quantile value. Use -1 for mean.

    mean_model : bool, optional
        If True, use mean coefficients. If False, use full coefficients. Default True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Aleatory quantile value [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'lambda': Box-Cox transformation parameter.
        - 'mu_site': Mean transformed displacement for the location.
        - 'sigma_site': Standard deviation transformed displacement for the location.
        - 'mu_complement': Mean transformed displacement for the complementary location.
        - 'sigma_complement': Standard deviation transformed displacement for the complementary location.
        - 'Y_site': Transformed displacement for the location.
        - 'Y_complement': Transformed displacement for the complementary location.
        - 'Y_folded': Transformed displacement for the folded location.
        - 'displ_site': Displacement in meters for the location.
        - 'displ_complement': Displacement in meters for the complementary location.
        - 'displ_folded': Displacement in meters for the folded location.

    Raises
    ------
    ValueError
        If the provided `style` is not one of the supported styles.

    TypeError
        If more than one value is provided for `magnitude`, `location`, `style`, or `percentile` when `mean_model=False`.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_displacement_model.py --magnitude 7 --location 0.5 --style strike-slip --percentile 0.5 0.84`
        Run `python run_displacement_model.py --help`

    #TODO
    ------
    Raise a ValueError for invalid location
    Raise a ValueError for invalid percentile.
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # Check if there are any invalid styles
    style = [x.lower() for x in ([style] if isinstance(style, str) else style)]
    supported_styles = ["strike-slip", "reverse", "normal"]
    invalid_mask = ~np.isin(style, supported_styles)

    if np.any(invalid_mask):
        invalid_styles = np.asarray(style)[invalid_mask]
        raise ValueError(
            f"Unsupported style: {invalid_styles}. Supported styles are {supported_styles} (case-insensitive)."
        )

    # If full model, only one scenario is allowed
    # TODO: allow more than one scenario? need to refactor `functions.py`?
    if not mean_model:
        scenario_dict = {
            "magnitude": magnitude,
            "location": location,
            "percentile": percentile,
            "style": style,
        }
        for key, value in scenario_dict.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                if len(value) != 1:
                    raise TypeError(
                        f"Only one value is allowed for '{key}' when `mean_model=False`, but user entered '{value}', which is {len(value)} values."
                    )

    # Vectorize scenarios
    scenarios = product(
        [magnitude] if not isinstance(magnitude, (list, np.ndarray)) else magnitude,
        [location] if not isinstance(location, (list, np.ndarray)) else location,
        [percentile] if not isinstance(percentile, (list, np.ndarray)) else percentile,
        [style] if not isinstance(style, (list, np.ndarray)) else style,
    )
    magnitude, location, percentile, style = map(np.array, zip(*scenarios))

    # Get distribution parameters for site and complement
    mu_site, sigma_site, lam, model_number = _calculate_distribution_parameters(
        magnitude=magnitude, location=location, style=style, mean_model=mean_model
    )
    mu_complement, sigma_complement, _, _ = _calculate_distribution_parameters(
        magnitude=magnitude,
        location=1 - location,
        style=style,
        mean_model=mean_model,
    )

    # Calculate Y (transformed displacement)
    Y_site = _calculate_Y(mu=mu_site, sigma=sigma_site, lam=lam, percentile=percentile)
    Y_complement = _calculate_Y(
        mu=mu_complement, sigma=sigma_complement, lam=lam, percentile=percentile
    )
    Y_folded = np.mean([Y_site, Y_complement], axis=0)

    # Calculate displacement in meters
    displ_site = _calculate_displacement(predicted_Y=Y_site, lam=lam)
    displ_complement = _calculate_displacement(predicted_Y=Y_complement, lam=lam)
    displ_folded = _calculate_displacement(predicted_Y=Y_folded, lam=lam)

    # Create a DataFrame
    # NOTE: number of rows will be controlled by number of scenarios (if mean model) or number of coefficients (if full model)
    n = max(len(magnitude), len(mu_site))
    results = (
        np.full(n, magnitude),
        np.full(n, location),
        np.full(n, style),
        np.full(n, percentile),
        [int(x) for x in model_number],
        lam,
        mu_site,
        sigma_site,
        mu_complement,
        sigma_complement,
        Y_site,
        Y_complement,
        Y_folded,
        displ_site,
        displ_complement,
        displ_folded,
    )

    type_dict = {
        "magnitude": float,
        "location": float,
        "style": str,
        "percentile": float,
        "model_number": int,
        "lambda": float,
        "mu_site": float,
        "sigma_site": float,
        "mu_complement": float,
        "sigma_complement": float,
        "Y_site": float,
        "Y_complement": float,
        "Y_folded": float,
        "displ_site": float,
        "displ_complement": float,
        "displ_folded": float,
    }
    dataframe = pd.DataFrame(np.column_stack(results), columns=type_dict.keys())
    dataframe = dataframe.astype(type_dict)

    return dataframe


def main():
    description_text = """Run KEA23 displacement model. A couple "gotchas":
        If full model is run (i.e., `--no-mean_model`), then only one scenario is allowed.
        If mean model is run (i.e., `--mean_model` or default), then any number of scnearios is allowed.
        A scenario is defined as a magnitude-location-style-percentile combination.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Aleatory quantile value [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'lambda': Box-Cox transformation parameter.
        - 'mu_site': Mean transformed displacement for the location.
        - 'sigma_site': Standard deviation transformed displacement for the location.
        - 'mu_complement': Mean transformed displacement for the complementary location.
        - 'sigma_complement': Standard deviation transformed displacement for the complementary location.
        - 'Y_site': Transformed displacement for the location.
        - 'Y_complement': Transformed displacement for the complementary location.
        - 'Y_folded': Transformed displacement for the folded location.
        - 'displ_site': Displacement in meters for the location.
        - 'displ_complement': Displacement in meters for the complementary location.
        - 'displ_folded': Displacement in meters for the folded location.
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
        "-s",
        "--style",
        required=True,
        nargs="+",
        type=str.lower,
        choices=("strike-slip", "reverse", "normal"),
        help="Style of faulting (case-insensitive).",
    )
    parser.add_argument(
        "-p",
        "--percentile",
        required=True,
        nargs="+",
        type=float,
        help="Aleatory quantile value. Use -1 for mean.",
    )

    parser.add_argument(
        "--full-model",
        dest="mean_model",
        action="store_false",
        help="Use full model coefficients. Default uses mean model coefficients.",
        default=True,
    )

    args = parser.parse_args()

    magnitude = args.magnitude
    location = args.location
    style = args.style
    percentile = args.percentile
    mean_model = args.mean_model

    try:
        results = run_model(
            magnitude=magnitude,
            location=location,
            style=style,
            percentile=percentile,
            mean_model=mean_model,
        )
        print(results)

        # Prompt to save results to CSV
        save_option = input("Do you want to save the results to a CSV (yes/no)? ").strip().lower()

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
