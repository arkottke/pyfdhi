"""This file contains tests for the internal functions used to calculate aggregated fault
displacement using the Kuehn et al. (2023) model. The results were computed by Alex Sarmiento and
checked by Dr. Nico Kuehn in November 2023.
"""

# Python imports
import sys
from pathlib import Path

import numpy as np
import pytest

# Module imports
from pyfdhi.disp.KuehnEtAl2023.data import POSTERIOR_MEAN
from pyfdhi.disp.KuehnEtAl2023.functions import (
    func_mode,
    func_mu,
    func_sd_mode_bilinear,
    func_sd_u,
    func_ss,
)

# Test setup
RTOL = 1e-2
STYLE = "strike-slip"
FILE = "strike-slip_mean-model.csv"

# Add path for expected outputs
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


# Load the expected outputs, run tests
@pytest.fixture
def results_data():
    ffp = SCRIPT_DIR / "expected_output" / FILE
    return np.genfromtxt(ffp, delimiter=",", skip_header=1, dtype=None)


def test_strike_slip_mean_model(results_data):
    coeffs = POSTERIOR_MEAN.get(STYLE)
    for row in results_data:
        # Inputs
        magnitude, location = row[:2]

        # Expected values
        expected_values = {
            "mode": row[2],
            "mu": row[3],
            "sd_mode": row[4],
            "sd_u": row[5],
            "median": row[3],  # Same as mu_expect
            "sd_tot": row[6],
        }

        # Computed values
        computed_values = {
            "mode": func_mode(coeffs, magnitude),
            "mu": func_mu(coeffs, magnitude, location),
            "sd_mode": func_sd_mode_bilinear(coeffs, magnitude),
            "sd_u": func_sd_u(coeffs, location),
        }
        computed_values["median"], computed_values["sd_tot"] = func_ss(coeffs, magnitude, location)

        # Tests
        for key, expected in expected_values.items():
            computed = computed_values[key]
            np.testing.assert_allclose(
                expected,
                computed,
                rtol=RTOL,
                err_msg=f"Mag {magnitude}, u-star {location}, {key}, Expected: {expected}, Computed: {computed}",
            )
