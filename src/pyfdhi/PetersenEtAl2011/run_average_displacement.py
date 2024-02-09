"""This file runs the PEA11 principal fault displacement model to calculate the average
displacement that is implied by the model prediction.
- Any number of scenarios are allowed (e.g., user can enter multiple magnitudes).
- The model-implied Average Displacement is calculated as the area under the mean slip profile.
- The results are returned in a pandas DataFrame.
- Only the principal fault displacement models for direct (i.e., not normalized) predictions are
implemented herein currently.
- Command-line use is supported; try `python run_average_displacement.py --help`
- Module use is supported; try `from run_average_displacement import run_ad`

# NOTE: This script just calls `run_displacement_profile.py` which in turn calls `run_displacement_model.py`

Reference: https://doi.org/10.1785/0120100035

"""

# Python imports
import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

# Module imports
import pyfdhi.PetersenEtAl2011.model_config as model_config  # noqa: F401
from pyfdhi.PetersenEtAl2011.run_displacement_profile import run_profile


def run_ad(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    submodel: Union[str, List[str], np.ndarray] = "elliptical",
    style: Union[str, List[str], np.ndarray] = "strike-slip",
    debug_bilinear_model: bool = False,
) -> pd.DataFrame:
    """
    Run PEA11 principal fault displacement model to calculate the average displacement that is
    implied by the model prediction. All parameters must be passed as keyword arguments. Any number
    of scenarios (i.e., magnitude inputs, submodel inputs, etc.) are allowed.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    submodel : Union[str, list, numpy.ndarray], optional
        PEA11 shape model name  (case-insensitive). Default is "elliptical". Valid options are
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
        - 'style': Style of faulting [from user input].
        - 'model_name': Profile shape model name [from user input].
        - 'avg_displ': Average displacement in meters.

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
        Run (e.g.) `python run_average_displacement.py --magnitude 7 7.5 --submodel quadratic`
        Run `python run_average_displacement.py --help`

    #TODO
    ------
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # NOTE: Check for appropriate style is handled upstream

    # Calculate mean slip profile
    # NOTE: `percentile=-1` is used for mean
    # NOTE: `location_step=0.01` is used to create well-descritized profile for intergration
    results = run_profile(
        magnitude=magnitude,
        percentile=-1,
        submodel=submodel,
        style=style,
        location_step=0.01,
        debug_bilinear_model=debug_bilinear_model,  # FIXME: bilinear model debugger issue
    )

    # Group by magnitude and submodel shape
    grouped = results.groupby(["magnitude", "model_name", "style"])

    # Calculate area under the mean slip profile; this is the Average Displacement (AD)
    # NOTE: use dictionary comprehension, it is probably slightly faster than apply lambda
    areas = {
        (mag, model, style): np.trapz(group["displ"], group["location"])
        for (mag, model, style), group in grouped
    }

    # Create output dataframe
    magnitudes, model_names, styles, area_values = zip(
        *[(mag, model, style, area) for (mag, model, style), area in areas.items()]
    )

    values = (
        list(magnitudes),
        list(model_names),
        list(styles),
        list(area_values),
    )

    type_dict = {
        "magnitude": float,
        "model_name": str,
        "style": str,
        "avg_displ": float,
    }
    dataframe = pd.DataFrame(np.column_stack(values), columns=type_dict.keys())
    dataframe = dataframe.astype(type_dict)

    return dataframe


def main():
    description_text = """Run PEA11 principal fault displacement model to calculate the average displacement that is
    implied by the model prediction. Any number of scenarios (i.e., magnitude inputs, submodel inputs, etc.) are allowed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'style': Style of faulting [from user input].
        - 'model_name': Profile shape model name [from user input].
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
    submodel = args.submodel
    style = args.style
    debug = args.debug_bilinear_model  # FIXME: bilinear model debugger issue

    try:
        results = run_ad(
            magnitude=magnitude,
            submodel=submodel,
            style=style,
            debug_bilinear_model=debug,  # FIXME: bilinear model debugger issue
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
