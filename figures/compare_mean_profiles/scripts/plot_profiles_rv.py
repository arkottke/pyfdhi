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
    "KEA23": "kea23_rv_profile.csv",
    "MR11": "mr11_d_ad_profile.csv",
}

df_kea = pd.read_csv(proj_setup.RESULTS_DIR / RESULTS_FILES["KEA23"])
df_mr = pd.read_csv(proj_setup.RESULTS_DIR / RESULTS_FILES["MR11"])

STYLE = "Reverse"

# Plotting

# This sets up plot style and plots KEA results
fig, ax = plot_functions.plot_profiles_rev(df_kea)

# Plot style-specific results
model = "MR11"
ax.plot(
    df_mr["location"],
    df_mr["normalized_d_xd"],
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
