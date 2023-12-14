""" """

# Python imports
import matplotlib.pyplot as plt
import pandas as pd

# Module imports
import proj_setup
import plot_functions
import plot_style  # noqa: F401

# Data imports
RESULTS_FILES = {
    "KEA23": "kea23_rv_ad.csv",
    "WC94-all": "wc94_all_ad.csv",
    # "WC94-rv": "wc94_rv_ad.csv",
    "MR11": "mr11_rv_ad.csv",
}
STYLE = "Reverse"

dataframes = {}
for key, filename in RESULTS_FILES.items():
    df = pd.read_csv(proj_setup.RESULTS_DIR / filename)
    dataframes[key] = df

# print(dataframes["KEA23"])

# Plotting
fig, ax = plot_functions.plot_mag_scaling(dataframes)
ax.set_title(STYLE)
ax.set_ylabel("Average Displacement (m)")
ax.set(ylim=[0.01, 30])
fout = f"{STYLE}_mag_scaling_ad.png"
fout = proj_setup.PLOTS_DIR / fout
plt.savefig(fout, bbox_inches="tight")
plt.close(fig)
