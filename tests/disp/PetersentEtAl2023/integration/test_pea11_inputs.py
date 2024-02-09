"""This file contains tests for the errors or warnings that should be raised with the user
functions for the Petersen et al. (2011_ models.
"""

# Python imports
import pytest

# Module imports
from pyfdhi.disp.PetersenEtAl2011.run_displacement_model import run_model


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_input_style():
    mag, loc, ptile = [6.5, 7], 0.25, 0.5

    # Test with appropriate style, case-insensitive; no exception should be raised
    sof = ["Strike-Slip", "strike-slip"]
    run_model(magnitude=mag, location=loc, percentile=ptile, style=sof)

    # Test with style that is not recommended
    sof = "normal"
    with pytest.warns(UserWarning):
        run_model(magnitude=mag, location=loc, percentile=ptile, style=sof)


def test_input_submodel():
    mag, loc, ptile = [6.5, 7], 0.25, 0.5

    # Test with appropriate submodel, case-insensitive; no exception should be raised
    model = ["Elliptical", "quadratic"]
    run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model)

    # Test with style that is not recommended
    model = "meow"
    with pytest.raises(ValueError):
        run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model)
