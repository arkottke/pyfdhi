"""This file contains tests for the errors or warnings that should be raised with the user
functions for the Kuehn et al. (2023) model.
"""

# Python imports
import pytest

# Module imports
from pyfdhi.disp.KuehnEtAl2023.run_displacement_model import run_model


def test_input_mean_model_style():
    mag, loc, ptile = [6.5, 7], 0.25, 0.5

    # Test mean model with appropriate style, case-insensitive; no exception should be raised
    sof = ["Strike-Slip", "strike-slip", "normal"]
    run_model(magnitude=mag, location=loc, style=sof, percentile=ptile)

    # Test with invalid style
    sof = ["Strike-Slip", "meow"]
    with pytest.raises(ValueError):
        run_model(magnitude=mag, location=loc, style=sof, percentile=ptile)


def test_input_full_model():
    mag, loc, sof, ptile = 7.2, 0.25, "Reverse", 0.5

    # Test with one scenario for full model; no exception should be raised
    run_model(magnitude=mag, location=loc, style=sof, percentile=ptile, mean_model=False)

    # Tests with multiple scenarios for full model (not allowed)
    mag, loc, sof, ptile = [6, 7.2], 0.25, "Reverse", 0.5
    with pytest.raises(TypeError):
        run_model(magnitude=mag, location=loc, style=sof, percentile=ptile, mean_model=False)

    mag, loc, sof, ptile = 6, [0.1, 0.25], "Reverse", 0.5
    with pytest.raises(TypeError):
        run_model(magnitude=mag, location=loc, style=sof, percentile=ptile, mean_model=False)

    mag, loc, sof, ptile = 6, 0.1, ["Reverse", "normal"], 0.5
    with pytest.raises(TypeError):
        run_model(magnitude=mag, location=loc, style=sof, percentile=ptile, mean_model=False)

    mag, loc, sof, ptile = 6, 0.1, "Reverse", [0.5, 0.84]
    with pytest.raises(TypeError):
        run_model(magnitude=mag, location=loc, style=sof, percentile=ptile, mean_model=False)
