from importlib.metadata import version

import pandas as pd

__author__ = "Alex Sarmiento"
__copyright__ = "Copyright 2024 Alex Sarmiento"
__license__ = "MIT"
__title__ = "pyFDHI"
# __version__ = "0.0.1"
__version__ = version("pyFDHI")


# Console output formatting
pd.set_option("display.max_columns", 800)
pd.set_option("display.width", 800)

# Monte Carlo sample size
# NOTE: N=500,000 was chosen because it is still reasonably fast and produces smooth slip profiles.
N_SAMPLES = 500000

# FIXME: Why set this?
# Numpy seed
NP_SEED = 123
