"""This script loads the model coefficients for the KEA23 displacement model.
- The coefficient files vary based on style of faulting.
- The coefficients are returned in a pandas dataframe.

# NOTE: This script is called in the main function in `run_model()`.

Reference: https://doi.org/10.1177/ToBeAssigned
"""

# Python imports
from pathlib import Path
from typing import Union

import pandas as pd

# Filepath for model coefficients
DIR_DATA = Path(__file__).parents[1] / "data" / "KuehnEtAl2023"

# Filenames for model coefficients
FILENAMES = {
    "strike-slip": "coefficients_posterior_SS_powtr.csv",
    "reverse": "coefficients_posterior_REV_powtr.csv",
    "normal": "coefficients_posterior_NM_powtr.csv",
}


def _load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load model coefficients.

    Parameters
    ----------
    filepath : Union[str, pathlib.Path]
        The path to the CSV file containting the model coefficients.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the model coefficients.

    """

    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    # FIXME: Various issues with recarray, use pandas for now
    # data = np.genfromtxt(filepath, delimiter=",", names=True, encoding="UTF-8-sig")
    # return data.view(np.recarray)
    return pd.read_csv(filepath).rename(columns={"Unnamed: 0": "model_number"})


def _calculate_mean_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to calculate mean model coefficients.

    Parameters
    ----------
    coefficients : pd.DataFrame
        A pandas DataFrame containing model coefficients.

    Returns
    -------
    coeffs : pd.DataFrame
        A pandas DataFrame containing mean model coefficients.
    """

    coeffs = coefficients.mean(axis=0).to_frame().transpose()
    coeffs.loc[0, "model_number"] = -1  # Define model id as -1 for mean coeffs
    coeffs["model_number"] = coeffs["model_number"].astype(int)

    return coeffs


# Import model coefficients
POSTERIOR_SS = _load_data(DIR_DATA / FILENAMES["strike-slip"])
POSTERIOR_RV = _load_data(DIR_DATA / FILENAMES["reverse"])
POSTERIOR_NM = _load_data(DIR_DATA / FILENAMES["normal"])

# Create style-data dictionary
POSTERIOR = {
    "strike-slip": POSTERIOR_SS,
    "reverse": POSTERIOR_RV,
    "normal": POSTERIOR_NM,
}

# Calculate mean model coefficients
POSTERIOR_SS_MEAN = _calculate_mean_coefficients(POSTERIOR_SS)
POSTERIOR_RV_MEAN = _calculate_mean_coefficients(POSTERIOR_RV)
POSTERIOR_NM_MEAN = _calculate_mean_coefficients(POSTERIOR_NM)

# Create style-data dictionary
POSTERIOR_MEAN = {
    "strike-slip": POSTERIOR_SS_MEAN,
    "reverse": POSTERIOR_RV_MEAN,
    "normal": POSTERIOR_NM_MEAN,
}
