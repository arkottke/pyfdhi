"""This file contains tests for the errors or warnings that should be raised with the user
functions for the Moss and Ross (2011) models.
"""

# Python imports
import pytest

# Module imports
from pyfdhi.MossRoss2011.run_average_displacement import run_ad
from pyfdhi.MossRoss2011.run_displacement_model import run_model
from pyfdhi.MossRoss2011.run_max_displacement import run_md


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_input_style():
    mag, loc, ptile, model = [6.5, 7], 0.25, 0.5, "d_md"

    # Test with appropriate styles, case-insensitive; no exception should be raised
    sof = ["Reverse", "reverse"]
    run_ad(magnitude=mag, percentile=ptile, style=sof)
    run_md(magnitude=mag, percentile=ptile, style=sof)
    run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model, style=sof)

    # Test with invalid style
    sof = "Strike-Slip"
    with pytest.warns(UserWarning):
        run_ad(magnitude=mag, style=sof)
        run_md(magnitude=mag, style=sof)
        run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model, style=sof)

    # Test with invalid style
    sof = ["normal", "Reverse"]
    with pytest.warns(UserWarning):
        run_ad(magnitude=mag, style=sof)
        run_md(magnitude=mag, style=sof)
        run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model, style=sof)


def test_input_submodel():
    mag, loc, ptile, model = 7, [0.1, 0.25], [0.5, 0.84], ["d_md", "d_ad"]

    # Test with appropriate submodels, case-insensitive; no exception should be raised
    run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model)

    # Test with invalid submodel
    model = "meow"
    with pytest.raises(ValueError):
        run_model(magnitude=mag, location=loc, percentile=ptile, submodel=model)
