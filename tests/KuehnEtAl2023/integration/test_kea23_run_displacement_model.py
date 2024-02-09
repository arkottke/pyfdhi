"""This file contains tests for Kuehn et al. (2023) fault displacement model. The results were
computed by Alex Sarmiento and checked by Dr. Nico Kuehn in November 2023.
"""

# Python imports
import sys
from pathlib import Path

import numpy as np
import pytest


# Module imports
from pyfdhi.KuehnEtAl2023.run_displacement_model import run_model


# Test setup
RTOL = 1e-2

# Add path for expected outputs
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


# Load the expected outputs, run tests
@pytest.fixture
def results_data():
    ffp = SCRIPT_DIR / "expected_output" / "displacement_mean-model.csv"
    dtype = [float, float, float, "U20"] + [float] * 10
    return np.genfromtxt(ffp, delimiter=",", skip_header=1, dtype=dtype)


def test_run_model(results_data):
    for row in results_data:
        # Inputs
        magnitude, location, percentile, style, *expected_outputs = row

        # Expected values
        expected_keys = [
            "mu_site",
            "sigma_site",
            "mu_complement",
            "sigma_complement",
            "Y_site",
            "Y_complement",
            "Y_folded",
            "displ_site",
            "displ_complement",
            "displ_folded",
        ]
        expected_values = dict(zip(expected_keys, expected_outputs))

        # Computed values
        results = run_model(
            magnitude=magnitude,
            location=location,
            style=style,
            percentile=percentile,
        )

        # Testing
        for key in expected_keys:
            np.testing.assert_allclose(
                expected_values[key],
                results[key],
                rtol=RTOL,
                err_msg=f"Mag {magnitude}, u-star {location}, style {style}, percentile {percentile}, Expected: {expected_values[key]}, Computed: {results[key]}",
            )
