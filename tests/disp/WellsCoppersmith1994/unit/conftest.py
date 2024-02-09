# Import python libraries
from pathlib import Path

import numpy as np
import pytest


# Define fixtures
@pytest.fixture
def load_data_as_recarray(filename):
    """Load expected values."""
    filepath = Path(__file__).parent / "expected_output" / filename
    dtype = [float, "U20", float, float]
    return np.genfromtxt(filepath, delimiter=",", skip_header=1, dtype=dtype)
