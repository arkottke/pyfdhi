"""This file runs the MR11 model to calculate the max displacement as a function of magnitude.
- Any number of scenarios are allowed (e.g., user can enter multiple magnitudes).
- The results are returned in a pandas DataFrame.
- Command-line use is supported; try `python run_max_displacement.py --help`
- Module use is supported; try `from run_max_displacement import run_ad`

Reference: https://doi.org/10.1785/BSSA0840040974
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

# Add path for project
# FIXME: shouldn't need to do this!
PROJ_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJ_DIR))
del PROJ_DIR

# Module imports
import MossRoss2011.model_config as model_config  # noqa: F401
from MossRoss2011.functions import _calc_distrib_params_mag_md


def run_md(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    percentile: Union[float, int, List[Union[float, int]], np.ndarray] = 0.5,
    style: str = "reverse",
) -> pd.DataFrame:
    """
    Run MR11 model to calculate the maximum displacement as a function of magnitude. All parameters
    must be passed as keyword arguments. Any number of scenarios (i.e., magnitude inputs,
    percentile inputs, etc.) are allowed.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    percentile : Union[float, list, numpy.ndarray], optional
        Aleatory quantile value. Default is 0.5. Use -1 for mean.

    style : str, optional
        Style of faulting (case-insensitive). Default is 'reverse'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'mu': Log10 transform of mean displacement in m.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'avg_displ': Maximum displacement in meters.

    Warns
    -----
    UserWarning
        If an unsupported `style` is provided. The user input will be over-ridden with 'reverse'.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_max_displacement.py --magnitude 6.5 7`
        Run `python run_max_displacement.py --help`

    #TODO
    ------
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # Check for allowable styles, then over-ride
    if style not in ("reverse", "Reverse"):
        warnings.warn(
            f"This model is only recommended for strike-slip faulting, but '{style}' was entered."
            f"User input will be over-ridden.",
            category=UserWarning,
        )
        style = "reverse"

    # Vectorize scenarios
    scenarios = product(
        [magnitude] if not isinstance(magnitude, (list, np.ndarray)) else magnitude,
        [percentile] if not isinstance(percentile, (list, np.ndarray)) else percentile,
    )
    magnitude, percentile = map(np.array, zip(*scenarios))

    # Calculate distribution parameters
    mu, sigma = _calc_distrib_params_mag_md(magnitude=magnitude)

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
        "max_displ": float,
    }
    dataframe = pd.DataFrame(np.column_stack(results), columns=cols_dict.keys())
    dataframe = dataframe.astype(cols_dict)

    return dataframe


def main():
    description_text = """Run MR11 model to calculate the maximum displacement as a function of
    magnitude. Any number of scenarios are allowed (e.g., user can enter multiple magnitudes or
    percentiles).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'mu': Log10 transform of mean displacement in m.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'avg_displ': Maximum displacement in meters.
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
        default="reverse",
        nargs="+",
        type=str.lower,
        help="Style of faulting (case-insensitive). Default is 'reverse'; other styles not recommended.",
    )

    args = parser.parse_args()

    magnitude = args.magnitude
    percentile = args.percentile
    style = args.style

    try:
        results = run_md(magnitude=magnitude, percentile=percentile, style=style)
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
