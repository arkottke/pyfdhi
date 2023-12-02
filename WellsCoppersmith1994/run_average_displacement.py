"""This file runs the WC94 model to calculate the average displacement as a function of magnitude.
- Any number of scenarios are allowed (e.g., user can enter multiple magnitudes).
- The results are returned in a pandas DataFrame.
- Command-line use is supported; try `python run_average_displacement.py --help`
- Module use is supported; try `from run_average_displacement import run_ad`

Reference: https://doi.org/10.1785/BSSA0840040974
"""

# Python imports
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import product
from scipy import stats
from typing import Union, List

# Module imports
import model_config  # noqa: F401
from WellsCoppersmith1994.functions import _calc_distrib_params_mag_ad


def _calc_distrib_params(*, magnitude, style):
    """
    A vectorized helper function to calculate predicted distribution parameters.

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude.

    style : str, optional
        Style of faulting (case-insensitive). Default is 'all'. Valid options are 'strike-slip',
        'reverse', 'normal', or 'all'.

    Returns
    -------
    Tuple[np.array, np.array]
        mu : Mean prediction in log10 units.
        sigma : Standard deviation in log10 units.

    Notes
    ------
    Mu and sigma are in log10 units
    """

    # Calculate for all submodels
    # NOTE: it is actually faster to just do this instead of if/else, loops, etc.
    result_ss = _calc_distrib_params_mag_ad(magnitude=magnitude, style="strike-slip")
    result_rv = _calc_distrib_params_mag_ad(magnitude=magnitude, style="reverse")
    result_nm = _calc_distrib_params_mag_ad(magnitude=magnitude, style="normal")
    result_all = _calc_distrib_params_mag_ad(magnitude=magnitude, style="all")

    # Conditions for np.select
    conditions = [
        style == "strike-slip",
        style == "reverse",
        style == "normal",
        style == "all",
    ]

    # Choices for mu and sigma
    choices_mu = [result_ss[0], result_rv[0], result_nm[0], result_all[0]]
    choices_sigma = [result_ss[1], result_rv[1], result_nm[1], result_all[1]]

    # Use np.select to get the final mu and sigma
    mu = np.select(conditions, choices_mu, default=np.nan)
    sigma = np.select(conditions, choices_sigma, default=np.nan)

    return mu, sigma


def run_ad(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    percentile: Union[float, int, List[Union[float, int]], np.ndarray],
    style: str = "all",
) -> pd.DataFrame:
    """
    Run WC94 model to calculate the average displacement as a function of magnitude. All parameters
    must be passed as keyword arguments. Any number of scenarios (i.e., magnitude inputs, style
    inputs, etc.) are allowed.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    percentile : Union[float, list, numpy.ndarray], optional
        Aleatory quantile value. Default is 0.5. Use -1 for mean.

    style : Union[str, list, numpy.ndarray], optional
        Style of faulting (case-insensitive). Default is 'all'. Valid options are 'strike-slip',
        'reverse', 'normal', or 'all'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'mu': Log10 transform of mean displacement in m.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'avg_displ': Average displacement in meters.

    Raises
    ------
    ValueError
        If the provided `style` is not one of the supported styles.

    Warns
    -----
    UserWarning
        If reverse `style` is provided.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_average_displacement.py --magnitude 7 --style strike-slip all`
        Run `python run_average_displacement.py --help`

    #TODO
    ------
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # Check if there are any invalid styles
    style = [x.lower() for x in ([style] if isinstance(style, str) else style)]
    supported_styles = ["strike-slip", "reverse", "normal", "all"]
    invalid_mask = ~np.isin(style, supported_styles)

    if np.any(invalid_mask):
        invalid_styles = np.asarray(style)[invalid_mask]
        raise ValueError(
            f"Invalid style: {invalid_styles}. Supported submodels are {supported_styles}."
        )

    # Warn if reverse; relations not recommended
    if np.any(np.asarray(style) == "reverse"):
        msg = "Regressions for reverse-slip relationships are not significant at 95% probability level (per WC94). Use with caution."
        warnings.warn(msg)

    # Vectorize scenarios
    scenarios = product(
        [magnitude] if not isinstance(magnitude, (list, np.ndarray)) else magnitude,
        [percentile] if not isinstance(percentile, (list, np.ndarray)) else percentile,
        [style] if not isinstance(style, (list, np.ndarray)) else style,
    )
    magnitude, percentile, style = map(np.array, zip(*scenarios))

    # Calculate distribution parameters
    mu, sigma = _calc_distrib_params(magnitude=magnitude, style=style)

    # Calculate natural log of displacement (vectorized approach)
    if np.any(percentile == -1):
        # Compute the mean
        log10_displ_mean = mu + (np.log(10) / 2 * np.power(sigma, 2))
    else:
        log10_displ_mean = np.nan

    # Compute the aleatory quantile
    log10_displ_normal = stats.norm.ppf(percentile, loc=mu, scale=sigma)

    # Use np.where for vectorization
    log10_displ = np.where(percentile == -1, log10_displ_mean, log10_displ_normal)

    # Calculate displacement
    D = np.power(10, log10_displ)

    # Create a DataFrame
    n = len(magnitude)
    results = (
        magnitude,
        np.full(n, style),
        percentile,
        mu,
        sigma,
        D,
    )

    cols_dict = {
        "magnitude": float,
        "style": str,
        "percentile": float,
        "mu": float,
        "sigma": float,
        "avg_displ": float,
    }
    dataframe = pd.DataFrame(np.column_stack(results), columns=cols_dict.keys())
    dataframe = dataframe.astype(cols_dict)

    return dataframe


def main():
    description_text = """Run WC94 model to calculate the average displacement as a function of
    magnitude. Any number of scenarios are allowed (e.g., user can enter multiple magnitudes or
    styles).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Aleatory quantile value [from user input].
        - 'mu': Log10 transform of mean displacement in m.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'avg_displ': Average displacement in meters.
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
        "-p",
        "--percentile",
        default=0.5,
        nargs="+",
        type=float,
        help=" Aleatory quantile value. Default is 0.5. Use -1 for mean.",
    )
    parser.add_argument(
        "-s",
        "--style",
        default="all",
        nargs="+",
        type=str.lower,
        choices=("strike-slip", "reverse", "normal", "all"),
        help="Style of faulting (case-insensitive). Default is 'all'.",
    )

    args = parser.parse_args()

    magnitude = args.magnitude
    percentile = args.percentile
    style = args.style

    try:
        results = run_ad(magnitude=magnitude, percentile=percentile, style=style)
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
