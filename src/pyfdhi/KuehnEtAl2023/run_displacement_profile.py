"""This file runs the KEA23 displacement model to create a slip profile.
- The mean model (i.e., mean coefficients) is used.
- The results are returned in a pandas DataFrame.
- Results for left-peak, right-peak, and folded (symmetrical) profiles are always returned.
- Command-line use is supported; try `python run_displacement_profile.py --help`
- Module use is supported; try `from run_displacement_profile import run_profile`

# NOTE: This script just calls `run_displacement_model.py`

Reference: https://doi.org/10.1177/ToBeAssigned
"""


# Python imports
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Union, List

# Add path for project
# FIXME: shouldn't need to do this!
PROJ_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJ_DIR))
del PROJ_DIR

# Module imports
import KuehnEtAl2023.model_config as model_config  # noqa: F401
from KuehnEtAl2023.run_displacement_model import run_model


def run_profile(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    style: Union[str, List[str], np.ndarray],
    percentile: Union[float, int, List[Union[float, int]], np.ndarray],
    location_step: float = 0.05,
) -> pd.DataFrame:
    """
    Run KEA23 displacement model to create slip profile. All parameters must be passed as keyword
    arguments. The mean model (i.e., mean coefficients) is used. Any number of scenarios (i.e.,
    magnitudes, styles, percentiles) are allowed.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    style : Union[str, list, numpy.ndarray]
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    percentile : Union[float, list, numpy.ndarray]
        Aleatory quantile value. Use -1 for mean.

    location_step : float, optional
        Profile step interval in percentage. Default 0.05.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [generated from location_step].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'lambda': Box-Cox transformation parameter.
        - 'mu_left': Mean transformed displacement for the left-peak profile.
        - 'sigma_left': Standard deviation transformed displacement for the left-peak profile.
        - 'mu_right': Mean transformed displacement for the right-peak profile.
        - 'sigma_right': Standard deviation transformed displacement for the right-peak profile.
        - 'Y_left': Transformed displacement for the left-peak profile.
        - 'Y_right': Transformed displacement for the right-peak profile.
        - 'Y_folded': Transformed displacement for the folded (symmetrical) profile.
        - 'displ_left': Displacement in meters for the left-peak profile.
        - 'displ_right': Displacement in meters for the right-peak profile.
        - 'displ_folded': Displacement in meters for the folded (symmetrical) profile.

    Raises (inherited from `run_displacement_model.py`)
    ------
    ValueError
        If the provided `style` is not one of the supported styles.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_displacement_profile.py --magnitude 6 7 --style strike-slip --percentile 0.5 -step 0.01`
        Run `python run_displacement_profile.py --help`

    #TODO
    ------
    Raise a ValueError for invalid location_step size.
    Raise a ValueError for invalid percentile.
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # NOTE: Check for appropriate style is handled in `run_model`

    # Create profile location array
    locations = np.arange(0, 1 + location_step, location_step)

    dataframe = run_model(
        magnitude=magnitude,
        location=locations,
        style=style,
        percentile=percentile,
        mean_model=True,
    )

    return dataframe.sort_values(by=["magnitude", "style", "percentile", "location"]).reset_index(
        drop=True
    )


def main():
    description_text = """Run KEA23 displacement model to create slip profile. The mean model (i.e., mean coefficients) is used.Any number of scenarios (i.e., magnitudes, styles, percentiles) are allowed.

    Returns
    -------
    pandas.DataFrame
        A DataFpandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [generated from location_step].
        - 'style': Style of faulting [from user input].
        - 'percentile': Aleatory quantile value [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'lambda': Box-Cox transformation parameter.
        - 'mu_left': Mean transformed displacement for the left-peak profile.
        - 'sigma_left': Standard deviation transformed displacement for the left-peak profile.
        - 'mu_right': Mean transformed displacement for the right-peak profile.
        - 'sigma_right': Standard deviation transformed displacement for the right-peak profile.
        - 'Y_left': Transformed displacement for the left-peak profile.
        - 'Y_right': Transformed displacement for the right-peak profile.
        - 'Y_folded': Transformed displacement for the folded (symmetrical) profile.
        - 'displ_left': Displacement in meters for the left-peak profile.
        - 'displ_right': Displacement in meters for the right-peak profile.
        - 'displ_folded': Displacement in meters for the folded (symmetrical) profile.
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
        "-s",
        "--style",
        required=True,
        nargs="+",
        type=str.lower,
        choices=("strike-slip", "reverse", "normal"),
        help="Style of faulting (case-sensitive).",
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
        "-step",
        "--location_step",
        default=0.05,
        type=float,
        help="Profile step interval in percentage. Default 0.05.",
    )

    args = parser.parse_args()

    magnitude = args.magnitude
    style = args.style
    percentile = args.percentile
    location_step = args.location_step

    try:
        results = run_profile(
            magnitude=magnitude,
            style=style,
            percentile=percentile,
            location_step=location_step,
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
