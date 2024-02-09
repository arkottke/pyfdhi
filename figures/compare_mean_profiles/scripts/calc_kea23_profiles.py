""" This script calculates a slip profile using the KEA23 model. """

# Python imports
import numpy as np
import pandas as pd  # noqa: F401

# Module imports
import proj_setup
from KuehnEtAl2023.run_displacement_profile import run_profile

# Calculations
df_ss = run_profile(
    magnitude=proj_setup.MAG,
    style="strike-slip",
    percentile=-1,
    location_step=proj_setup.STEP,
)
df_rv = run_profile(
    magnitude=proj_setup.MAG,
    style="reverse",
    percentile=-1,
    location_step=proj_setup.STEP,
)
df_nm = run_profile(
    magnitude=proj_setup.MAG,
    style="normal",
    percentile=-1,
    location_step=proj_setup.STEP,
)

y_vals = ["displ_site", "displ_complement", "displ_folded"]
dataframes = [df_ss, df_rv, df_nm]

for _y in y_vals:
    for _df in dataframes:
        area = np.trapz(_df[_y], _df["location"])
        col = f"normalized_{_y}"
        _df[col] = _df[_y] / area

        # check = np.trapz(_df[col], _df["location"])
        # print(area, check)


# Save
filenames = ["kea23_ss_profile.csv", "kea23_rv_profile.csv", "kea23_nm_profile.csv"]
for file, df in zip(filenames, dataframes):
    fout = proj_setup.RESULTS_DIR / file
    df.to_csv(fout, index=False)
