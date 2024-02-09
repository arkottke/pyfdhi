""" This script calculates a slip profile using the MR11 models. """

# Python imports
import numpy as np
import pandas as pd  # noqa: F401

# Module imports
import proj_setup
from MossRoss2011.run_displacement_profile import run_profile

# Calculations
df = run_profile(
    magnitude=proj_setup.MAG,
    percentile=-1,
    submodel="d_ad",
    location_step=proj_setup.STEP,
)

area = np.trapz(df["d_xd"], df["location"])  # area is not exactly 1 for mean D/AD profile
df["normalized_d_xd"] = df["d_xd"] / area

# print(area)
# check = np.trapz(df["normalized_d_xd"], df["location"])
# print(check)

# Save
fout = proj_setup.RESULTS_DIR / "mr11_d_ad_profile.csv"
df.to_csv(fout, index=False)
