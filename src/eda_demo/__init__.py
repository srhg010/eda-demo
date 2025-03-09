"""
EDA Demo Package

This package provides tools for exploratory data analysis across various datasets.
It centralizes common imports and paths for all modules in the package.
"""

# Standard library imports
import json
import re
import types
from itertools import combinations
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple, Union

# Third-party imports
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import sqlalchemy
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import EngFormatter
from nltk.stem.porter import PorterStemmer
from numpy.typing import NDArray
from pandas import DataFrame, Series
from pandas.compat.pickle_compat import patch_pickle
from scipy import stats
from scipy.optimize import minimize
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

# Define project paths
try:
    # Base paths
    # PROJECT_ROOT = Path(__file__).parent.resolve()
    PROJECT_ROOT = Path().resolve().parent.parent

    # Try to locate datasets directory relative to project
    # This attempts several common locations
    possible_dataset_paths = [
        Path("~/Documents/Projects/Python/datasets").expanduser(),
        Path("../datasets").resolve(),
        Path("../../datasets").resolve(),
    ]

    # Use the first valid path
    for path in possible_dataset_paths:
        if path.exists():
            DATASETS_PATH = path
            break
    else:
        # If none found, use a default
        DATASETS_PATH = Path("../datasets").resolve()
        print(f"Warning: Using default datasets path: {DATASETS_PATH}")
        print("If your data is elsewhere, edit the path in __init__.py")

    # Define specific dataset paths
    PURPLEAIR_PATH = DATASETS_PATH / "purpleair_study"
    PACLEANED_PATH = PURPLEAIR_PATH / "cleaned_purpleair_aqs"
    CLEANED24_PATH = PACLEANED_PATH / "Full24hrdataset.csv"
    AQSSITES_PATH = PURPLEAIR_PATH / "list_of_aqs_sites.csv"
    PASENSORS_PATH = PURPLEAIR_PATH / "list_of_purpleair_sensors.json"
    SACRAMSENSOR_PATH = PURPLEAIR_PATH / "aqs_06-067-0010.csv"
    AMTSTESTINGADIR_PATH = PURPLEAIR_PATH / "purpleair_AMTS"
    PA_CSVS = sorted(AMTSTESTINGADIR_PATH.glob("*.csv"))
    TEXT_PATH = DATASETS_PATH / "stateoftheunion1790-2025.txt"
    DONKEYS_PATH = DATASETS_PATH / "donkeys.csv"
    SFH_PATH = DATASETS_PATH / "sfhousing.csv"

    # Style path
    PACOTY_PATH = PROJECT_ROOT / "pacoty.mplstyle"
    if PACOTY_PATH.exists():
        mpl.style.use(PACOTY_PATH)
except Exception as e:
    print(f"Warning: Error setting up paths: {e}")
    print("Some functionality may be limited.")

# Configure matplotlib
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["hist.bins"] = "doane"
mpl.rcParams["figure.facecolor"] = mpl.rcParams["axes.facecolor"]

# Store the original histogram function
original_hist = Axes.hist


# Create a wrapper function that modifies the default rwidth
def hist_with_gaps(self, x, bins=None, **kwargs):
    """
    Modified histogram function that adds gaps between bars by default.

    Args:
        self: The Axes instance
        x: The data to plot
        bins: Number of bins or bin edges
        **kwargs: Additional keyword arguments for hist

    Returns:
        The return value from the original histogram function
    """
    # Set default rwidth if not provided
    if "rwidth" not in kwargs:
        kwargs["rwidth"] = 0.85  # Create 15% gaps between bars

    # Call the original histogram function with our modified parameters
    return original_hist(self, x, bins=bins, **kwargs)


# Replace the original hist method with our modified version
Axes.hist = hist_with_gaps


# Utility function for saving figures
def save_figure(fig: Figure, filename: str, dpi: int = 300) -> Path:
    """
    Save a matplotlib figure with standard settings.

    Args:
        fig: The matplotlib figure to save
        filename: Name of the file (without path)
        dpi: Resolution in dots per inch

    Returns:
        Path to the saved figure
    """
    # Ensure filename has extension
    if not any(filename.endswith(ext) for ext in [".png", ".jpg", ".pdf", ".svg"]):
        filename = f"{filename}.png"

    # Create figures dir if it doesn't exist
    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Save the figure
    save_path = figures_dir / filename
    fig.savefig(save_path, dpi=dpi)
    print(f"Figure saved to {save_path}")
    return save_path


# Package version
__version__ = "0.1.0"
