""" This script calculates the average displacement using the relations in WC94. """

# Python imports
import pandas as pd  # noqa: F401

# Module imports
import proj_setup
from WellsCoppersmith1994.run_average_displacement import run_ad

# Calculations
df_ss = run_ad(magnitude=proj_setup.MAGS, style="strike-slip")
df_rv = run_ad(magnitude=proj_setup.MAGS, style="reverse")
df_nm = run_ad(magnitude=proj_setup.MAGS, style="normal")
df_all = run_ad(magnitude=proj_setup.MAGS, style="all")

# Save
dataframes = [df_ss, df_rv, df_nm, df_all]
filenames = ["wc94_ss_ad.csv", "wc94_rv_ad.csv", "wc94_nm_ad.csv", "wc94_all_ad.csv"]

for file, df in zip(filenames, dataframes):
    fout = proj_setup.RESULTS_DIR / file
    df.to_csv(fout, index=False)
