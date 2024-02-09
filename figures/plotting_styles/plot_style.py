"""This file defines the plot style for plots used in the report."""

import matplotlib.pyplot as plt

plt.style.use("default")

# FIGURE
plt.rcParams["figure.figsize"] = (3.2, 2.5)
plt.rcParams["figure.dpi"] = 600

# LINES
plt.rcParams["lines.linewidth"] = 1.1
plt.rcParams["lines.markersize"] = 3

# LEGEND
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["legend.labelspacing"] = 0.2
plt.rcParams["legend.borderpad"] = 0.2
plt.rcParams["legend.handlelength"] = 2.5

# FONTS
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
    "Lucida Grande",
    "Verdana",
    "Geneva",
    "Lucid",
    "Avant Garde",
    "sans-serif",
]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.size"] = 10

# AXES
plt.rcParams["axes.grid"] = "True"
plt.rcParams["axes.grid.axis"] = "both"
plt.rcParams["axes.grid.which"] = "major"
plt.rcParams["grid.linewidth"] = "0.6"
plt.rcParams["grid.color"] = "#BABABA"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# LATEX
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}"
