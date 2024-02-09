"""This file contains tests for the errors or warnings that should be raised with the user
functions for the Youngs et al. (2003) models.
"""

# Python imports
import sys
from pathlib import Path

import pytest

# Module imports
from pyfdhi.YoungsEtAl2003.run_displacement_model import run_model


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_input_style():
    mag, loc, ptile, model = [6.5, 7], 0.25, 0.5, "d_ad"

    # Test with appropriate styles, case-insensitive; no exception should be raised
    sof = ["Normal", "NORMAL"]
    run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model, style=sof)

    # Test with invalid style
    sof = "Strike-Slip"
    with pytest.warns(UserWarning):
        run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model, style=sof)

    # Test with invalid style
    sof = ["normal", "Reverse"]
    with pytest.warns(UserWarning):
        run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model, style=sof)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_input_submodel():
    mag, loc, ptile, model = 7, [0.1, 0.25], [0.5, 0.84], "d_ad"

    # Test with appropriate submodels, case-insensitive; no exception should be raised
    run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model)

    # Test with invalid submodel
    model = "d_md"
    with pytest.warns(UserWarning):
        run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model)
