"""This file runs hazard (probability of exceedance) for the KEA23 displacement model.
- The results are returned in a pandas DataFrame.
- Results for the location, its complement, and folded location are always returned.
- The mean model (i.e., mean coefficients) is run by default, but results for all coefficients can be computed.
- If full model is run (i.e., `mean_model=False`), then only one scenario is allowed.
- A scenario is defined as a magnitude-location-style combination.
- If mean model is run (i.e., `mean_model=True` or default), then any number of scenarios is allowed.
- Command-line use is supported; try `python run_probability_exceedance.py --help`
- Module use is supported; try `from run_probability_exceedance import run_probex`

Reference: https://doi.org/10.1177/ToBeAssigned
"""

# Python imports
import argparse
from itertools import product
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from scipy import stats

# Module imports
import pyfdhi.disp.KuehnEtAl2023.model_config as model_config  # noqa: F401
from pyfdhi.disp.KuehnEtAl2023.params import _calculate_distribution_parameters


def run_probex(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    location: Union[float, int, List[Union[float, int]], np.ndarray],
    style: Union[str, List[str], np.ndarray],
    displacement: Union[float, int, List[Union[float, int]], np.ndarray],
    mean_model: bool = True,
) -> pd.DataFrame:
    """
    Calculate hazard, in the form of probability of exceedance, using the KEA23 displacement model.
    All parameters must be passed as keyword arguments.
    A couple "gotchas":
        If full model is run (i.e., `mean_model=False`), then only one scenario is allowed.
        If mean model is run (i.e., `mean_model=True` or default), then any number of scenarios is allowed.
        A scenario is defined as a magnitude-location-style combination.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    location : Union[float, list, numpy.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    style : Union[str, list, numpy.ndarray]
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    displacement : Union[float, list, numpy.ndarray]
        Test values of displacement in meters.

    mean_model : bool, optional
        If True, use mean coefficients. If False, use full coefficients. Default True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
        - 'style': Style of faulting [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'lambda': Box-Cox transformation parameter.
        - 'mu_site': Mean transformed displacement for the location.
        - 'sigma_site': Standard deviation transformed displacement for the location.
        - 'mu_complement': Mean transformed displacement for the complementary location.
        - 'sigma_complement': Standard deviation transformed displacement for the complementary location.
        - 'transformed_displacement': Transformed displacement test value.
        - 'displacement': Displacement test value in meters [from user input].
        - 'probex_site': Probability of exceedance for the location.
        - 'probex_complement': Probability of exceedance for the complementary location.
        - 'probex_folded': Equally-weighted probability of exceedance for site and complement.

    Raises
    ------
    ValueError
        If the provided `style` is not one of the supported styles.

    TypeError
        If more than one value is provided for `magnitude`, `location`, or `style` when `mean_model=False`.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `run_probability_exceedance.py --magnitude 7 --location 0.5 --style strike-slip --displacement 0.01 0.03 0.1 0.3 1 3 10 30`
        Run `python rrun_probability_exceedance.py --help`

    #TODO
    ------
    Raise a ValueError for invalid location
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
        [style] if not isinstance(style, (list, np.ndarray)) else style,
    )
    magnitude, location, style = map(np.array, zip(*scenarios))

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

    # Arrange scenarios and displacements for element-wise calculations
    # TODO: will need to change this approach if more than one scenario is allowed for full model
    if not mean_model:
        n_models = len(model_number)
        magnitude = np.repeat(magnitude, n_models)
        location = np.repeat(location, n_models)
        style = np.repeat(style, n_models)

    n_scenarios = len(magnitude)
    n_displacements = len(np.atleast_1d(displacement))

    _arrays = [
        magnitude,
        location,
        style,
        mu_site,
        sigma_site,
        lam,
        model_number,
        mu_complement,
        sigma_complement,
    ]

    _repeated_arrays = [np.repeat(arr, n_displacements) for arr in _arrays]

    # Unpack
    (
        magnitude,
        location,
        style,
        mu_site,
        sigma_site,
        lam,
        model_number,
        mu_complement,
        sigma_complement,
    ) = _repeated_arrays

    displacement = np.tile(displacement, n_scenarios)

    # Calculate probability of exceedance
    transformed_displ = (displacement**lam - 1) / lam
    probex_site = 1 - stats.norm.cdf(x=transformed_displ, loc=mu_site, scale=sigma_site)
    probex_complement = 1 - stats.norm.cdf(
        x=transformed_displ, loc=mu_complement, scale=sigma_complement
    )
    probex_folded = np.mean((probex_site, probex_complement), axis=0)

    # Create a DataFrame
    results = (
        magnitude,
        location,
        style,
        [int(x) for x in model_number],
        lam,
        mu_site,
        sigma_site,
        mu_complement,
        sigma_complement,
        transformed_displ,
        displacement,
        probex_site,
        probex_complement,
        probex_folded,
    )

    type_dict = {
        "magnitude": float,
        "location": float,
        "style": str,
        "model_number": int,
        "lambda": float,
        "mu_site": float,
        "sigma_site": float,
        "mu_complement": float,
        "sigma_complement": float,
        "transformed_displacement": float,
        "displacement": float,
        "probex_site": float,
        "probex_complement": float,
        "probex_folded": float,
    }
    dataframe = pd.DataFrame(np.column_stack(results), columns=type_dict.keys())
    dataframe = dataframe.astype(type_dict)

    return dataframe.sort_values(
        by=["model_number", "magnitude", "style", "location", "displacement"]
    ).reset_index(drop=True)


def main():
    description_text = """Calculate hazard, in the form of probability of exceedance, using the KEA23 displacement model. A couple "gotchas":
        If full model is run (i.e., `mean_model=False`), then only one scenario is allowed.
        If mean model is run (i.e., `mean_model=True` or default), then any number of scenarios is allowed.
        A scenario is defined as a magnitude-location-style combination.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
        - 'style': Style of faulting [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'lambda': Box-Cox transformation parameter.
        - 'mu_site': Mean transformed displacement for the location.
        - 'sigma_site': Standard deviation transformed displacement for the location.
        - 'mu_complement': Mean transformed displacement for the complementary location.
        - 'sigma_complement': Standard deviation transformed displacement for the complementary location.
        - 'transformed_displacement': Transformed displacement test value.
        - 'displacement': Displacement test value in meters [from user input].
        - 'probex_site': Probability of exceedance for the location.
        - 'probex_complement': Probability of exceedance for the complementary location.
        - 'probex_folded': Equally-weighted probability of exceedance for site and complement.
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
        "-d",
        "--displacement",
        required=True,
        nargs="+",
        type=float,
        help="Test values of displacement in meters.",
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
    displacement = args.displacement
    mean_model = args.mean_model

    try:
        results = run_probex(
            magnitude=magnitude,
            location=location,
            style=style,
            displacement=displacement,
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
