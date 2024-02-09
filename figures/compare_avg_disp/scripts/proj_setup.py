import sys
from pathlib import Path

import numpy as np

# Get script directories
CWD = Path(sys.argv[0]).absolute().parent
ROOT = CWD.parent

# Add path for modules
# FIXME: shouldn't need this with a package install (`__init__` should suffice)
PROJ_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJ_DIR))
sys.path.extend(
    str(PROJ_DIR / model)
    for model in ["KuehnEtAl2023", "WellsCoppersmith1994", "MossRoss2011", "PetersenEtAl2011"]
)

# Add path for custom plotting
PLOT_DIR = PROJ_DIR / "figures" / "plotting_styles"
sys.path.append(str(PLOT_DIR))
del PROJ_DIR, PLOT_DIR

# Define output paths
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

# Define cases
MAGS = np.arange(5, 8.6, 0.1).tolist()
