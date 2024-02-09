"""This file runs the MR11 principal fault displacement model.
- Any number of scenarios are allowed (e.g., user can enter multiple magnitudes).
- The results are returned in a pandas DataFrame.
- Results with full aleatory variability and with location-only aleatory variability are always returned.
- The results with full aleatory variability are computed by convolving distributions for normalized
displacement (based on magnitude) and normalization ratio (based on location) using the Monte Carlo sampling
method described in Moss and Ross (2011).
- Only the principal fault displacement models are implemented herein currently.
- Command-line use is supported; try `python run_displacement_model.py --help`
- Module use is supported; try `from run_displacement_model import run_model`

# NOTE: Several helper functions are defined herein, but the main function is `run_model()`.

Reference: https://doi.org/10.1785/0120100248
"""

# Python imports
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List

# Add path for project
# FIXME: shouldn't need to do this!
PROJ_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJ_DIR))
del PROJ_DIR

# Module imports
import MossRoss2011.model_config as model_config  # noqa: F401

# Set numpy seed and number of samples
SEED = model_config.NP_SEED
N = model_config.N_SAMPLES


from MossRoss2011.functions import (
    _calc_distrib_params_mag_ad,
    _calc_distrib_params_mag_md,
    _calc_distrib_params_d_ad,
    _calc_distrib_params_d_md,
)


def _calc_distribution_params_and_samples(*, magnitude, location, percentile, submodel):
    """
    A vectorized helper function to do the following:
        (1) Calculate predicted distribution parameters for normalized displacement based on magnitude.
        (2) Calculate predicted distribution parameters for normalization ratio based on location.
        (3) Perform Monte Carlo sampling on each distribution (N=500000).
        (4) Broadcast results for all combinations of magnitude, location, and percentile.
        (5) Return results in a dictionary.

    Parameters
    ----------
    magnitude : numpy.ndarray
        Earthquake moment magnitude.

    location : numpy.ndarray
        Normalized location along rupture length, range [0, 1.0].

    percentile : numpy.ndarray
        Aleatory quantile value. Use -1 for mean.

    submodel : str
        MR11 normalization model name (case-insensitive). Valid options are "d_ad" or "d_md". Only
        one valid is permitted.

    Returns
    -------
    dict
        A dictionary containing the scenarios, predicted distribution parameters, and distribution
        samples. The keys are values are:
        - 'magnitude' : numpy.ndarray
            - Earthquake moment magnitude.
        - 'location' : numpy.ndarray
            - Normalized location along rupture length.
        - 'percentile' : numpy.ndarray
            - Aleatory quantile value.
        - 'model_name' : numpy.ndarray
            - Normalization ratio model name.
        - 'mu' : numpy.ndarray
            - Log10 transform of mean average or maximum displacement in m.
        - 'sigma' : numpy.ndarray
            - Standard deviation of average or maximum displacement in same units as `mu`.
        - 'alpha' : numpy.ndarray
            - Shape parameter for Gamma distribution (D/AD) or Beta distribution (D/MD).
        - 'beta' : numpy.ndarray
            - Scale parameter for Gamma distribution (D/AD) or shape parameter for Beta distribution (D/MD).
        - 'xd_samples' : numpy.ndarray
            - Samples from the distribution for normalization variable AD or MD (i.e., based on `mu` and `sigma`).
        - 'd_xd_samples' : numpy.ndarray
            - Samples from the distribution for normalization ratio D/AD or D/MD (i.e., based on `alpha` and `beta`).

    Raises
    ------
    TypeError
        If more than one value is provided for `submodel`.

    Notes
    ------
    N=500,000 was chosen because it is still reasonably fast and produces smooth slip profiles.
    """

    # Check for only one submodel
    # NOTE: Check for appropriate submodel name is handled in `run_model`
    if not isinstance(submodel, (str)):
        raise TypeError(
            f"Expected a string, got '{submodel}', which is a {type(submodel).__name__}."
            f"(In other words, only one value is allowed; check you have not entered a list or array.)"
        )

    # NOTE: Only one sample set is generated for a given magnitude or location; the sample is ...
    # ...broadcasted when the magnitude or location is repeated.

    # Calculate distribution parameters
    if submodel == "d_ad":
        mu, sigma = _calc_distrib_params_mag_ad(magnitude=magnitude)
        alpha, beta = _calc_distrib_params_d_ad(location=location)
        alpha, beta = alpha[:, np.newaxis], beta[:, np.newaxis]
        sampling_func = stats.gamma.rvs
        kwargs = {
            "a": alpha,
            "loc": 0,
            "scale": beta,
            "size": (location.size, N),
        }

    elif submodel == "d_md":
        mu, sigma = _calc_distrib_params_mag_md(magnitude=magnitude)
        alpha, beta = _calc_distrib_params_d_md(location=location)
        alpha, beta = alpha[:, np.newaxis], beta[:, np.newaxis]
        sampling_func = stats.beta.rvs
        kwargs = {
            "a": alpha,
            "b": beta,
            "size": (location.size, N),
        }

    # Calculate samples for normalization variable AD or MD
    mu, sigma = mu[:, np.newaxis], sigma[:, np.newaxis]
    mag_array_shape = (magnitude.size, N)
    np.random.seed(SEED)  # this needs to be reset before each RVS
    xd_samples_log = stats.norm.rvs(loc=mu, scale=sigma, size=mag_array_shape)
    xd_samples = np.power(10, xd_samples_log)

    # Calcuate samples for normalization ratio D_AD or D_MD
    np.random.seed(SEED)  # this needs to be reset before each RVS
    d_xd_samples = sampling_func(**kwargs)

    # Create index arrays for broadcasting
    mag_idx, loc_idx, perc_idx = np.indices((magnitude.size, location.size, percentile.size))
    mag_idx, loc_idx, perc_idx = [x.flatten() for x in (mag_idx, loc_idx, perc_idx)]

    # Broadcasting based on indicies; return as dictionary
    _results_dict = {
        "magnitude": magnitude[mag_idx],
        "location": location[loc_idx],
        "percentile": percentile[perc_idx],
        "model_name": np.repeat(submodel, len(mag_idx)),
        "mu": mu[mag_idx].flatten(),
        "sigma": sigma[mag_idx].flatten(),
        "alpha": alpha[loc_idx].flatten(),
        "beta": beta[loc_idx].flatten(),
        "xd_samples": xd_samples[mag_idx],
        "d_xd_samples": d_xd_samples[loc_idx],
    }

    return _results_dict


def _calc_displacement_from_sample(*, percentile, convolded_sample):
    """
    A vectorized helper function to calculate displacement with full aleatory variability from the
    distribution of convolved samples, as described in Moss and Ross (2011).

    Parameters
    ----------
    percentile : numpy.ndarray
        Aleatory quantile value. Use -1 for mean.

    convolded_sample : numpy.ndarray
        Convolved distribution of normalized displacement samples (based on magnitude) and
        normalization ratio samples (based on location).

    Returns
    ----------
    displacements : numpy.ndarray
        Displacement for percentile in meters.
    """

    # Define conditions for np.select filter
    conditions = [percentile != -1, percentile == -1]

    # Compute the aleatory quantile from the convolved sample
    percentile_displacement = np.array(
        [
            np.percentile(row, 100 * percentile[idx]) if percentile[idx] != -1 else np.nan
            for idx, row in enumerate(convolded_sample)
        ]
    )

    # Compute the mean of the convolved sample
    mean_displacement = np.array(
        [
            np.mean(row) if percentile[idx] == -1 else np.nan
            for idx, row in enumerate(convolded_sample)
        ]
    )

    choices = [percentile_displacement, mean_displacement]

    # Use np.select to create the final displacement array
    displacements = np.select(conditions, choices)

    return displacements


def _calc_displacement_from_scenario(*, mu, sigma, alpha, beta, percentile, submodel):
    """
    A vectorized helper function to calculate normalized displacement (D/AD or D/MD), normalization
    value (AD or MD), and displacement amplitude without aleatory variability on magnitude (e.g.,
    D = AD(P_50) * D/AD(P_p)).

    Parameters
    ----------
    mu : numpy.ndarray
        Mean prediction for AD or MD in log10 units.

    sigma : numpy.ndarray
        Standard deviation for AD or MD.

    alpha : numpy.ndarray
        Shape parameter for Gamma distribution (D/AD) or Beta distribution (D/MD).

    beta : numpy.ndarray
        Scale parameter for Gamma distribution (D/AD) or shape parameter for Beta distribution (D/MD).

    percentile : numpy.ndarray
        Aleatory quantile value. Use -1 for mean.

    submodel : str
        MR11 normalization model name (case-insensitive). Valid options are "d_ad" or "d_md". Only
        one valid is permitted.

    Returns
    -------
    dict
        A dictionary containing the scenarios, predicted displacements. The keys are values are:
        - 'xd' : numpy.ndarray
            - Median predicted displacement for AD or MD.
        - 'd_xd' : numpy.ndarray
            - Normalization ratio D/AD or D/MD for aleatory quantile.
        - 'displ_without_aleatory' : numpy.ndarray
            - NDisplacement in meters without aleatory variability on magntiude.

    Raises
    ------
    TypeError
        If more than one value is provided for `submodel`.
    """

    # Check for only one submodel
    # NOTE: Check for appropriate submodel name is handled in `run_model`
    if not isinstance(submodel, (str)):
        raise TypeError(
            f"Expected a string, got '{submodel}', which is a {type(submodel).__name__}."
            f"(In other words, only one value is allowed; check you have not entered a list or array.)"
        )

    if submodel == "d_ad":
        # Compute the mean
        if np.any(percentile == -1):
            d_xd_mean = alpha * beta
        else:
            d_xd_mean = np.nan

        # Compute the aleatory quantile
        d_xd_ptile = stats.gamma.ppf(q=percentile, a=alpha, loc=0, scale=beta)

    elif submodel == "d_md":
        # Compute the mean
        if np.any(percentile == -1):
            d_xd_mean = alpha / (alpha + beta)
        else:
            d_xd_mean = np.nan

        # Compute the aleatory quantile
        d_xd_ptile = stats.beta.ppf(q=percentile, a=alpha, b=beta, loc=0)

    # Use np.where for vectorization
    d_xd = np.where(percentile == -1, d_xd_mean, d_xd_ptile)

    # Calculate displacements
    xd = np.power(10, mu)
    displacements = xd * d_xd

    # Return as dictionary
    _results_dict = {
        "xd": xd,
        "d_xd": d_xd,
        "displ_without_aleatory": displacements,
    }

    return _results_dict


def _model_runner_helper(*, magnitude, location, percentile, style, submodel):
    """
    A helper function to calculate predicted distribution parameters, perform Monte Carlo sampling
    on the distributions, convolve the distributions, and calculate displacements with and without
    aleatory variability on magnitude for all combinations of magntiude, location, and percentile.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    location : Union[float, list, numpy.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    percentile : Union[float, list, numpy.ndarray]
        Aleatory quantile value. Use -1 for mean.

    style : str
        Style of faulting (case-insensitive). Only one valid is permitted.

    submodel : str
        MR11 normalization model name (case-insensitive). Valid options are "d_ad" or "d_md". Only
        one valid is permitted.

    Returns
    -------
    dict
        A dictionary containing the scenarios, predicted distribution parameters, and distribution
        samples. The keys are values are:
        - 'magnitude' : numpy.ndarray
            - Earthquake moment magnitude.
        - 'location' : numpy.ndarray
            - Normalized location along rupture length.
        - 'style' : numpy.ndarray
            - Style of faulting.
        - 'percentile' : numpy.ndarray
            - Aleatory quantile value.
        - 'model_name' : numpy.ndarray
            - Normalization ratio model name.
        - 'mu' : numpy.ndarray
            - Log10 transform of mean average or maximum displacement in m.
        - 'sigma' : numpy.ndarray
            - Standard deviation of average or maximum displacement in same units as `mu`.
        - 'alpha' : numpy.ndarray
            - Shape parameter for Gamma distribution (D/AD) or Beta distribution (D/MD).
        - 'beta' : numpy.ndarray
            - Scale parameter for Gamma distribution (D/AD) or shape parameter for Beta distribution (D/MD).
        - 'xd' : numpy.ndarray
            - Median predicted displacement for AD or MD.
        - 'd_xd' : numpy.ndarray
            - Normalization ratio D/AD or D/MD for aleatory quantile..
        - 'displ_without_aleatory' : numpy.ndarray
            - Displacement in meters without aleatory variability on magntiude.
        - 'displ_with_aleatory' : numpy.ndarray
            - Displacement in meters with full aleatory variability.

    Raises
    ------
    TypeError
        If more than one value is provided for `submodel` or `style`.
    """

    # Check for only one submodel and style
    # NOTE: Check for appropriate submodel name is handled in `run_model`
    for variable in [style, submodel]:
        if not isinstance(variable, (str)):
            raise TypeError(
                f"Expected a string, got '{variable}', which is a {type(variable).__name__}."
                f"(In other words, only one value is allowed; check you have not entered a list or array.)"
            )

    # Compute Monte Carlo samples in a dictionary of broadcasted magnitude/location/percentile scenarios
    full_results_dict = _calc_distribution_params_and_samples(
        magnitude=magnitude, location=location, percentile=percentile, submodel=submodel
    )

    # NOTE: Assumes any invalid user input is over-ridden in `run_model`
    full_results_dict["style"] = np.repeat(style, len(full_results_dict["magnitude"]))

    # Convolve samples
    samples = full_results_dict["xd_samples"] * full_results_dict["d_xd_samples"]

    # Calculate displacement with full aleatory variability
    displ_with_aleatory = _calc_displacement_from_sample(
        percentile=full_results_dict["percentile"], convolded_sample=samples
    )
    full_results_dict["displ_with_aleatory"] = displ_with_aleatory

    # Calculate displacement without magnitude aleatory variability
    kwargs_keys = ["mu", "sigma", "alpha", "beta", "percentile"]
    kwargs = {key: full_results_dict[key] for key in kwargs_keys}
    kwargs["submodel"] = submodel
    simple_results_dict = _calc_displacement_from_scenario(**kwargs)

    # Combine dictionaries, remove sample arrays
    final_dict = {**full_results_dict, **simple_results_dict}
    final_dict = {
        key: final_dict[key] for key in final_dict if key not in {"xd_samples", "d_xd_samples"}
    }

    return final_dict


def run_model(
    *,
    magnitude: Union[float, int, List[Union[float, int]], np.ndarray],
    location: Union[float, int, List[Union[float, int]], np.ndarray],
    percentile: Union[float, int, List[Union[float, int]], np.ndarray],
    submodel: Union[str, List[str], np.ndarray],
    style: Union[str, List[str], np.ndarray] = "reverse",
) -> pd.DataFrame:
    """
    Run MR11 principal fault displacement model. All parameters must be passed as keyword
    arguments. Any number of scenarios (i.e., magnitude inputs, location inputs, etc.) are allowed.

    Parameters
    ----------
    magnitude : Union[float, list, numpy.ndarray]
        Earthquake moment magnitude.

    location : Union[float, list, numpy.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    percentile : Union[float, list, numpy.ndarray]
        Aleatory quantile value. Use -1 for mean.

    submodel : Union[str, list, numpy.ndarray]
        MR11 normalization model name (case-insensitive). Valid options are "d_ad" or "d_md".

    style : Union[str, list, numpy.ndarray], optional
        Style of faulting (case-insensitive). Default is "reverse".

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
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

    Raises
    ------
    ValueError
        If invalid `submodel` is provided.

    Warns
    -----
    UserWarning
        If an unsupported `style` is provided. The user input will be over-ridden with "reverse".

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_displacement_model.py --magnitude 7 --location 0.5 --percentile 0.5 --submodel d_ad`
        Run `python run_displacement_model.py --help`

    #TODO
    ------
    Raise a ValueError for invalid location
    Raise a ValueError for invalid percentile.
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # Check for allowable styles, then over-ride
    if style not in ("reverse", "Reverse"):
        warnings.warn(
            f"This model is only recommended for reverse faulting, but '{style}' was entered."
            f"User input will be over-ridden.",
            category=UserWarning,
        )
        style = "reverse"

    # Check if there are any invalid submodels
    submodel = [x.lower() for x in ([submodel] if isinstance(submodel, str) else submodel)]
    supported_submodels = ["d_ad", "d_md"]
    invalid_mask = ~np.isin(submodel, supported_submodels)

    if np.any(invalid_mask):
        invalid_submodels = np.asarray(submodel)[invalid_mask]
        raise ValueError(
            f"Invalid submodel names: {invalid_submodels}. Supported submodels are {supported_submodels}."
        )

    # Convert inputs to list-like numpy arrays
    magnitude, location, percentile, submodel = map(
        np.atleast_1d, (magnitude, location, percentile, submodel)
    )

    # Calculate and organize results
    col_order = [
        "magnitude",
        "location",
        "style",
        "percentile",
        "model_name",
        "mu",
        "sigma",
        "alpha",
        "beta",
        "xd",
        "d_xd",
        "displ_without_aleatory",
        "displ_with_aleatory",
    ]
    ad_dict = {key: np.array([]) for key in col_order}
    md_dict = {key: np.array([]) for key in col_order}

    if "d_ad" in submodel:
        ad_dict = _model_runner_helper(
            magnitude=magnitude,
            location=location,
            percentile=percentile,
            style=style,
            submodel="d_ad",
        )

    if "d_md" in submodel:
        md_dict = _model_runner_helper(
            magnitude=magnitude,
            location=location,
            percentile=percentile,
            style=style,
            submodel="d_md",
        )

    results_dict = {key: np.concatenate([ad_dict[key], md_dict[key]]) for key in ad_dict}
    dataframe = pd.DataFrame(results_dict)
    dataframe = dataframe[col_order]

    return dataframe.sort_values(
        by=["model_name", "magnitude", "location", "percentile"]
    ).reset_index(drop=True)


def main():
    description_text = """Run MR11 principal fault displacement model. Any number of scenarios are
    allowed (e.g., user can enter multiple magnitudes or locations).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
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

    args = parser.parse_args()

    magnitude = args.magnitude
    location = args.location
    percentile = args.percentile
    submodel = args.submodel
    style = args.style

    try:
        results = run_model(
            magnitude=magnitude,
            location=location,
            percentile=percentile,
            submodel=submodel,
            style=style,
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
