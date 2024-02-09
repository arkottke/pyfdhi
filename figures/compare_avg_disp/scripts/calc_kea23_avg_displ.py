""" This script calculates the average displacement that is implied by the KEA23 model prediction. """

# Python imports
import pandas as pd  # noqa: F401

# Module imports
import proj_setup
from KuehnEtAl2023.run_average_displacement import run_ad

# Calculations
df_ss = run_ad(magnitude=proj_setup.MAGS, style="strike-slip")
df_rv = run_ad(magnitude=proj_setup.MAGS, style="reverse")
df_nm = run_ad(magnitude=proj_setup.MAGS, style="normal")

# Save
dataframes = [df_ss, df_rv, df_nm]
filenames = ["kea23_ss_ad.csv", "kea23_rv_ad.csv", "kea23_nm_ad.csv"]

for file, df in zip(filenames, dataframes):
    fout = proj_setup.RESULTS_DIR / file
    df.to_csv(fout, index=False)
