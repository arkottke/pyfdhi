"""This file contains tests for the user functions used to calculate principal fault displacement
using the Moss and Ross (2011) model. The basis for the test answers were provided by Dr. Robb Moss
in August 2021 and Dr. Arash Zandieh in August 2022; and the results were computed by Alex
Sarmiento in October 2022.
"""

# Python imports
import sys
from pathlib import Path

import numpy as np
import pytest

# Add path for module
# FIXME: shouldn't need this with a package install (`__init__` should suffice?!)
PROJ_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJ_DIR))

# Module imports
from MossRoss2011.run_displacement_model import run_model

# Test setup; higher tolerance required due to differences between sampling distributions (code) and integrating PDF (XLS)
RTOL = 4e-2

# Add path for expected outputs
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


# Load the expected outputs, run tests
@pytest.fixture
def results_data():
    ffp = SCRIPT_DIR / "expected_output" / "moss_ross_displacements.csv"
    dtype = ["U20"] + [float] * 11
    return np.genfromtxt(ffp, delimiter=",", skip_header=1, dtype=dtype)


def test_run_model(results_data):
    for row in results_data:
        # Inputs
        model, magnitude, location, percentile, *expected_outputs = row

        # Expected values
        expected_keys = [
            "mu",
            "sigma",
            "alpha",
            "beta",
            "xd",
            "d_xd",
            "displ_without_aleatory",
            "displ_with_aleatory",
        ]
        expected_values = dict(zip(expected_keys, expected_outputs))

        # Computed values
        results = run_model(
            magnitude=magnitude,
            location=location,
            percentile=percentile,
            submodel=model,
        )

        # Testing
        for key in expected_keys:
            np.testing.assert_allclose(
                np.asarray(expected_values[key]).flatten(),
                np.asarray(results[key]).flatten(),
                rtol=RTOL,
                err_msg=f"Magnitude {magnitude}, location {location}, percentile {percentile}, model {model}, Expected: {expected_values[key]}, Computed: {results[key]}",
            )
