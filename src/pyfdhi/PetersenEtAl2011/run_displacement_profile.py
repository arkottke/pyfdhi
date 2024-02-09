"""This file runs the PEA11 principal fault displacement model to create a slip profile.
- Any number of scenarios are allowed (e.g., user can enter multiple magnitudes).
- The results are returned in a pandas DataFrame.
- Only the principal fault displacement models for direct (i.e., not normalized) predictions are
implemented herein currently.
- Command-line use is supported; try `python run_displacement_profile.py --help`
- Module use is supported; try `from run_displacement_profile import run_profile`

# NOTE: This script just loops over locations in `run_displacement_model.py`

Reference: https://doi.org/10.1785/0120100035
"""

# Python imports
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Union, List

# Module imports
import pyfdhi.PetersenEtAl2011.model_config as model_config  # noqa: F401
from pyfdhi.PetersenEtAl2011.run_displacement_model import run_model


def run_profile(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    percentile: Union[float, int, List[Union[float, int]], np.ndarray],
    submodel: Union[str, List[str], np.ndarray] = "elliptical",
    style: Union[str, List[str], np.ndarray] = "strike-slip",
    location_step: float = 0.05,
    debug_bilinear_model: bool = False,
) -> pd.DataFrame:
    """
    Run PEA11 principal fault displacement model to create slip profile. All parameters must be
    passed as keyword arguments. Any number of scenarios (i.e., magnitude inputs, percentile
    inputs, etc.) are allowed.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    percentile : Union[float, list, numpy.ndarray]
        Aleatory quantile value. Use -1 for mean.

    submodel : Union[str, list, numpy.ndarray], optional
        PEA11 shape model name  (case-insensitive). Default is "elliptical". Valid options are
        "elliptical", "quadratic", or "bilinear".

    style : Union[str, list, numpy.ndarray], optional
        Style of faulting (case-insensitive). Default is "strike-slip".

    location_step : float, optional
        Profile step interval in percentage. Default 0.05.

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
        - 'location':  Normalized location along rupture length [generated from location_step].
        - 'style': Style of faulting [from user input].
        - 'percentile': Aleatory quantile value [from user input].
        - 'model_name': Profile shape model name [from user input].
        - 'mu': Natural log transform of mean displacement in cm.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'displ': Displacement in meters.

    Raises (inherited from `run_displacement_model.py`)
    ------
    ValueError
        If invalid `submodel` is provided.

    Warns  (inherited from `run_displacement_model.py`)
    -----
    UserWarning
        If an unsupported `style` is provided. The user input will be over-ridden with "strike-slip".

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_displacement_profile.py --magnitude 7 7.5 --percentile 0.5 -shape bilinear -step 0.01`
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
        debug_bilinear_model=debug_bilinear_model,  # FIXME: bilinear model debugger issue
    )

    return dataframe.sort_values(
        by=["magnitude", "model_name", "percentile", "location"]
    ).reset_index(drop=True)


def main():
    description_text = """Run PEA11 principal fault displacement model to create slip profile. Any
    number of scenarios are allowed (e.g., user can enter multiple magnitudes or submodels).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [generated from location_step].
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
        "-p",
        "--percentile",
        required=True,
        nargs="+",
        type=float,
        help="Aleatory quantile value. Use -1 for mean.",
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
    parser.add_argument(
        "-step",
        "--location_step",
        default=0.05,
        type=float,
        help="Profile step interval in percentage. Default 0.05.",
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
    percentile = args.percentile
    submodel = args.submodel
    style = args.style
    location_step = args.location_step
    debug = args.debug_bilinear_model  # FIXME: bilinear model debugger issue

    try:
        results = run_profile(
            magnitude=magnitude,
            percentile=percentile,
            submodel=submodel,
            style=style,
            location_step=location_step,
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
