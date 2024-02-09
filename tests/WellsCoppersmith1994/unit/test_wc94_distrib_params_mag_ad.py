# Python imports
from pathlib import Path

import numpy as np
import pytest

# Module imports
from pyfdhi.WellsCoppersmith1994 import functions as wc94

# Test setup
FUNCTION = wc94._calc_distrib_params_mag_ad
FILE = "wells_coppersmith_params_mag_ad.csv"
RTOL = 1e-2


@pytest.mark.parametrize("filename", [FILE])
def test_calc(load_data_as_recarray):
    for row in load_data_as_recarray:
        # Inputs
        magnitude = row[0]
        style = row[1]

        # Expected
        mu_expect = row[2]
        sigma_expect = row[3]

        # Computed
        results = FUNCTION(magnitude=magnitude, style=style)
        mu_calc = results[0]
        sigma_calc = results[1]

        # Tests
        np.testing.assert_allclose(
            mu_expect,
            mu_calc,
            rtol=RTOL,
            err_msg=f"Mag {magnitude}, style {style}, Expected mu (log10): {mu_expect}, Computed: {mu_calc}",
        )

        np.testing.assert_allclose(
            sigma_expect,
            sigma_calc,
            rtol=RTOL,
            err_msg=f"Mag {magnitude}, style {style}, Expected sigma (log10): {sigma_expect}, Computed: {sigma_calc}",
        )
