# Python imports
import numpy as np
import pytest

# Module imports
from pyfdhi.disp.MossRoss2011.functions import _calc_distrib_params_mag_ad

# Test setup
FUNCTION = _calc_distrib_params_mag_ad
FILE = "moss_ross_params_mag_ad.csv"
RTOL = 1e-2


@pytest.mark.parametrize("filename", [FILE])
def test_calc(load_data_as_recarray):
    # Define inputs and expected outputs
    magnitudes = load_data_as_recarray["magnitude"]
    mu_expect = load_data_as_recarray["mu"]
    sigma_expect = load_data_as_recarray["sigma"]

    # Perform calculations
    results = FUNCTION(magnitude=magnitudes)
    mu_calc, sigma_calc = results

    # Comparing exepcted and calculated
    np.testing.assert_allclose(mu_expect, mu_calc, rtol=RTOL, err_msg="Discrepancy in mu values")
    np.testing.assert_allclose(
        sigma_expect, sigma_calc, rtol=RTOL, err_msg="Discrepancy in sigma values"
    )
