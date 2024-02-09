"""This file runs the KEA23 displacement model to calculate the average displacement that is implied by the model prediction.
- The model-implied Average Displacement is calculated as the area under the mean slip profile.
- The mean model (i.e., mean coefficients) is used.
- The results are returned in a pandas DataFrame.
- Command-line use is supported; try `python run_average_displacement.py --help`
- Module use is supported; try `from run_average_displacement import run_ad`

# NOTE: This script just calls `run_displacement_profile.py` which in turn calls `run_displacement_model.py`

Reference: https://doi.org/10.1177/ToBeAssigned
"""

# Python imports
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Union, List

# Module imports
import pyfdhi.KuehnEtAl2023.model_config as model_config  # noqa: F401
from pyfdhi.KuehnEtAl2023.run_displacement_profile import run_profile


def run_ad(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    style: Union[str, List[str], np.ndarray],
) -> pd.DataFrame:
    """
    Run KEA23 displacement model to calculate the average displacement that is implied by the model
    prediction. All parameters must be passed as keyword arguments. The mean model (i.e., mean
    coefficients) is used. Any number of scenarios (i.e., magnitudes, styles) are allowed.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    style : Union[str, list, numpy.ndarray]
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'style': Style of faulting [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'avg_displ': Average displacement in meters.

    Raises (inherited from `run_displacement_model.py`)
    ------
    ValueError
        If the provided `style` is not one of the supported styles.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_average_displacement.py --magnitude 5 6 7 --style reverse normal`
        Run `python run_average_displacement.py --help`

    #TODO
    ------
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # NOTE: Check for appropriate style is handled upstream

    # Calculate mean slip profile
    # NOTE: `percentile=-1` is used for mean
    # NOTE: `location_step=0.01` is used to create well-descritized profile for intergration
    results = run_profile(magnitude=magnitude, style=style, percentile=-1, location_step=0.01)

    # Group by magnitude and style
    grouped = results.groupby(["magnitude", "model_number", "style"])

    # Calculate area under the mean slip profile; this is the Average Displacement (AD)
    # NOTE: use dictionary comprehension, it is probably slightly faster than apply lambda
    areas = {
        (mag, model, style): np.trapz(group["displ_site"], group["location"])
        for (mag, model, style), group in grouped
    }

    # Create output dataframe
    magnitudes, model_numbers, styles, area_values = zip(
        *[(mag, model, style, area) for (mag, model, style), area in areas.items()]
    )

    values = (
        list(magnitudes),
        list(styles),
        list(model_numbers),
        list(area_values),
    )

    type_dict = {
        "magnitude": float,
        "style": str,
        "model_number": str,
        "avg_displ": float,
    }
    dataframe = pd.DataFrame(np.column_stack(values), columns=type_dict.keys())
    dataframe = dataframe.astype(type_dict)

    return dataframe


def main():
    description_text = """Run KEA23 displacement model to calculate the average displacement that is implied by the model prediction. The mean model (i.e., mean coefficients) is used. Any number of scenarios (i.e., magnitudes, styles) are allowed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'style': Style of faulting [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
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
        "-s",
        "--style",
        required=True,
        nargs="+",
        type=str.lower,
        choices=("strike-slip", "reverse", "normal"),
        help="Style of faulting (case-sensitive).",
    )

    args = parser.parse_args()

    magnitude = args.magnitude
    style = args.style
    try:
        results = run_ad(magnitude=magnitude, style=style)
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
