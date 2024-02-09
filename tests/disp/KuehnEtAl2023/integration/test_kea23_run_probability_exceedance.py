"""This file contains tests for probability of exceedance calculations for the Kuehn et al. (2023)
fault displacement model. The results were computed by Alex Sarmiento and checked by Dr. Nico Kuehn
in November 2023.
"""

# Python imports
import sys
from pathlib import Path

import numpy as np
import pytest

# Module imports
from pyfdhi.disp.KuehnEtAl2023.run_probability_exceedance import run_probex

# Test setup
RTOL = 1e-2

# Add path for expected outputs
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


# Load the expected outputs, run tests
@pytest.fixture
def results_data():
    ffp = SCRIPT_DIR / "expected_output" / "probexceed_mean-model.csv"
    dtype = [float, float, "U20"] + [float] * 4
    return np.genfromtxt(ffp, delimiter=",", skip_header=1, dtype=dtype)


def test_run_model(results_data):
    for row in results_data:
        # Inputs
        magnitude, location, style, displacement, *expected_outputs = row

        # Expected values
        expected_keys = [
            "probex_site",
            "probex_complement",
            "probex_folded",
        ]
        expected_values = dict(zip(expected_keys, expected_outputs))

        # Computed values
        results = run_probex(
            magnitude=magnitude,
            location=location,
            style=style,
            displacement=displacement,
        )

        # Testing
        for key in expected_keys:
            np.testing.assert_allclose(
                expected_values[key],
                results[key],
                rtol=RTOL,
                err_msg=f"Mag {magnitude}, u-star {location}, style {style}, Expected: {expected_values[key]}, Computed: {results[key]}",
            )
