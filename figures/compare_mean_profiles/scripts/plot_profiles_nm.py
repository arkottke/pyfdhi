""" """

# Python imports
import matplotlib.pyplot as plt
import pandas as pd
import plot_definitions as myplot
import plot_functions
import plot_style  # noqa: F401

# Module imports
import proj_setup

# Data imports
RESULTS_FILES = {
    "KEA23": "kea23_nm_profile.csv",
    "YEA03": "yea03_d_ad_profile.csv",
}

df_kea = pd.read_csv(proj_setup.RESULTS_DIR / RESULTS_FILES["KEA23"])
df_yea = pd.read_csv(proj_setup.RESULTS_DIR / RESULTS_FILES["YEA03"])

STYLE = "Normal"

# Plotting

# This sets up plot style and plots KEA results
fig, ax = plot_functions.plot_profiles_rev(df_kea)

# Plot style-specific results
model = "YEA03"
ax.plot(
    df_yea["location"],
    df_yea["normalized_d_xd"],
    c=myplot.model_colors.get(model),
    ls=myplot.model_linestyles.get(model),
    label=myplot.model_labels.get(model),
    zorder=10,
)

ax.set_title(STYLE)
ax.legend(loc="lower center")

# Save
fout = f"{STYLE}_profiles.png"
fout = proj_setup.PLOTS_DIR / fout
plt.savefig(fout, bbox_inches="tight")
plt.close(fig)
