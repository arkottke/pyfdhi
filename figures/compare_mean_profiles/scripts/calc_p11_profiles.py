""" This script calculates a slip profile using the PEA11 models. """

# Python imports
import numpy as np
import pandas as pd  # noqa: F401

# Module imports
import proj_setup
from PetersenEtAl2011.run_displacement_profile import run_profile

# Calculations
df = run_profile(
    magnitude=proj_setup.MAG,
    percentile=-1,
    submodel="elliptical",
    location_step=proj_setup.STEP,
)

area = np.trapz(df["displ"], df["location"])
df["normalized_displ"] = df["displ"] / area

# print(area)
# check = np.trapz(df["normalized_d_xd"], df["location"])
# print(check)

# Save
fout = proj_setup.RESULTS_DIR / "pea11_elliptical_profile.csv"
df.to_csv(fout, index=False)
