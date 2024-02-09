"""This file runs the MR11 principal fault displacement model to create a slip profile.
- Any number of scenarios are allowed (e.g., user can enter multiple magnitudes).
- The results are returned in a pandas DataFrame.
- Results with full aleatory variability and with location-only aleatory variability are always returned.
- The results with full aleatory variability are computed by convolving distributions for normalized
displacement (based on magnitude) and normalization ratio (based on location) using the Monte Carlo sampling
method described in Moss and Ross (2011).
- Only the principal fault displacement models are implemented herein currently.
- Command-line use is supported; try `python run_displacement_profile.py --help`
- Module use is supported; try `from run_displacement_profile import run_profile`

# NOTE: This script just loops over locations in `run_displacement_model.py`

Reference: https://doi.org/10.1785/0120100248
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
import MossRoss2011.model_config as model_config  # noqa: F401
from MossRoss2011.run_displacement_model import run_model


def run_profile(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    percentile: Union[float, int, List[Union[float, int]], np.ndarray],
    submodel: Union[str, List[str], np.ndarray],
    style: Union[str, List[str], np.ndarray] = "reverse",
    location_step: float = 0.05,
) -> pd.DataFrame:
    """
    Run MR11 principal fault displacement model to create slip profile. All parameters must be
    passed as keyword arguments. Any number of scenarios (i.e., magnitude inputs, percentile
    inputs, etc.) are allowed.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    percentile : Union[float, list, numpy.ndarray]
        Aleatory quantile value. Use -1 for mean.

    submodel : Union[str, list, numpy.ndarray]
        MR11 normalization model name (case-insensitive). Valid options are "d_ad" or "d_md".

    style : Union[str, list, numpy.ndarray], optional
        Style of faulting (case-insensitive). Default is "reverse".

    location_step : float, optional
        Profile step interval in percentage. Default 0.05.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [generated from location_step].
        - 'style': Style of faulting [from user input].
        - 'percentile': Aleatory quantile value [from user input].
        - 'model_name': Normalization ratio model name [from user input].
        - 'mu': Log10 transform of mean average or maximum displacement in m.
        - 'sigma': Standard deviation of average or maximum displacement in same units as `mu`.
        - 'alpha': Shape parameter for Gamma distribution (D/AD) or Beta distribution (D/MD).
        - 'beta': Scale parameter for Gamma distribution (D/AD) or shape parameter for Beta distribution (D/MD).
        - 'xd': Median predicted displacement for AD or MD.
        - 'd_xd': Normalization ratio D/AD or D/MD for aleatory quantile.
        - 'displ_without_aleatory': Displacement in meters without aleatory variability on magntiude.
        - 'displ_with_aleatory': Displacement in meters with full aleatory variability.

    Raises (inherited from `run_displacement_model.py`)
    ------
    ValueError
        If invalid `submodel` is provided.

    Warns  (inherited from `run_displacement_model.py`)
    -----
    UserWarning
        If an unsupported `style` is provided. The user input will be over-ridden with "reverse".

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_displacement_profile.py --magnitude 7 7.5 --percentile 0.5 -model d_md -step 0.01`
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
        percentile=percentile,
        submodel=submodel,
        style=style,
    )

    return dataframe.sort_values(
        by=["magnitude", "model_name", "percentile", "location"]
    ).reset_index(drop=True)


def main():
    description_text = """Run MR11 principal fault displacement model to create slip profile. Any
    number of scenarios are allowed (e.g., user can enter multiple magnitudes or submodels).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [generated from location_step].
        - 'style': Style of faulting [from user input].
        - 'percentile': Aleatory quantile value [from user input].
        - 'model_name': Normalization ratio model name [from user input].
        - 'mu': Log10 transform of mean average or maximum displacement in m.
        - 'sigma': Standard deviation of average or maximum displacement in same units as `mu`.
        - 'alpha': Shape parameter for Gamma distribution (D/AD) or Beta distribution (D/MD).
        - 'beta': Scale parameter for Gamma distribution (D/AD) or shape parameter for Beta distribution (D/MD).
        - 'xd': Median predicted displacement for AD or MD.
        - 'd_xd': Normalization ratio D/AD or D/MD for aleatory quantile.
        - 'displ_without_aleatory': Displacement in meters without aleatory variability on magntiude.
        - 'displ_with_aleatory': Displacement in meters with full aleatory variability.
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
        required=True,
        nargs="+",
        type=float,
        help="Aleatory quantile value. Use -1 for mean.",
    )
    parser.add_argument(
        "-model",
        "--submodel",
        nargs="+",
        type=str.lower,
        choices=("d_ad", "d_md"),
        help="MR11 normalization model name (case-insensitive).",
    )
    parser.add_argument(
        "-s",
        "--style",
        default="reverse",
        nargs="+",
        type=str.lower,
        help="Style of faulting (case-insensitive). Default is 'reverse'; other styles not recommended.",
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
    percentile = args.percentile
    submodel = args.submodel
    style = args.style
    location_step = args.location_step

    try:
        results = run_profile(
            magnitude=magnitude,
            percentile=percentile,
            submodel=submodel,
            style=style,
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
