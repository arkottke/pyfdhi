"""This file contains tests for the errors or warnings that should be raised with the user
functions for the Petersen et al. (2011_ models.
"""

# Python imports
import sys
from pathlib import Path

import pytest

# Add path for module
# FIXME: shouldn't need this with a package install (`__init__` should suffice)
PROJ_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJ_DIR))

# Module imports
from PetersenEtAl2011.run_displacement_model import run_model


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
