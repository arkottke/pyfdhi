"""This file contains tests for the user functions used to calculate principal fault displacement
using the Petersen et al. (2011) model. The basis for the test answers (i.e., mu, sigma) were
provided by Dr. Rui Chen in July 2021 and the quantiles were computed by Alex Sarmiento in October
2022.
"""

# Python imports
import sys
from pathlib import Path

import numpy as np
import pytest

# Module imports
from pyfdhi.disp.PetersenEtAl2011.run_displacement_model import run_model

# Test setup
RTOL = 1e-2

# Add path for expected outputs
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


# Load the expected outputs, run tests
@pytest.fixture
def results_data():
    ffp = SCRIPT_DIR / "expected_output" / "petersen_et_al_2011_displacements.csv"
    dtype = [float, float, float, "U20", float]
    return np.genfromtxt(ffp, delimiter=",", skip_header=1, dtype=dtype)


def test_run_model(results_data):
    for row in results_data:
        # Inputs
        magnitude, location, percentile, shape, *expected_outputs = row

        # Expected values
        expected_keys = ["displ"]
        expected_values = dict(zip(expected_keys, expected_outputs))

        # Computed values
        results = run_model(
            magnitude=magnitude,
            location=location,
            percentile=percentile,
            submodel=shape,
        )

        # Testing
        for key in expected_keys:
            np.testing.assert_allclose(
                expected_values[key],
                results[key],
                rtol=RTOL,
                err_msg=f"magnitude {magnitude}, location {location}, shape {shape}, percentile {percentile}, Expected: {expected_values[key]}, Computed: {results[key]}",
            )

        # Inputs
        # magnitude = row[0]
        # location = row[1]
        # percentile = row[2]
        # shape = row[4]

        # Expected
        # displ_expect = row[3]

        # Computed
        # results = run_model(
        # magnitude=magnitude,
        # location=location,
        # percentile=percentile,
        # submodel=shape,
        # )
        # displ_calc = results["displ"]

        # Tests
        # np.testing.assert_allclose(
        # displ_expect,
        # displ_calc,
        # rtol=RTOL,
        # err_msg=f"Mag {magnitude}, loc {location}, percentile {percentile}, submodel {shape},Expected: {displ_expect}, Computed: {displ_calc}",
        # )
