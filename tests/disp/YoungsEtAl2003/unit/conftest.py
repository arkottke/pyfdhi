# Import python libraries
from pathlib import Path

import numpy as np
import pytest


# Define fixtures
@pytest.fixture
def load_data_as_recarray(filename):
    """Load expected values."""
    filepath = Path(__file__).parent / "expected_output" / filename
    return np.genfromtxt(filepath, delimiter=",", names=True, encoding="UTF-8-sig", dtype=None)
