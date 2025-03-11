import json
import re
import types
from itertools import combinations
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple, Union

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

try:
    # PROJECT_ROOT = Path(__file__).parent.resolve()
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    possible_dataset_paths = [
        Path("~/Documents/Projects/Python/datasets").expanduser(),
        Path("../datasets").resolve(),
        Path("../../datasets").resolve(),
    ]

    for path in possible_dataset_paths:
        if path.exists():
            DATASETS_PATH = path
            break
    else:
        DATASETS_PATH = Path("../datasets").resolve()
        print(f"{DATASETS_PATH=}")
        print("trol")

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

    PACOTY_PATH = PROJECT_ROOT / "pacoty.mplstyle"
    if PACOTY_PATH.exists():
        mpl.style.use(PACOTY_PATH)
except Exception as e:
    print(f"Advertenthia: {e}")
    print("Nyaaa!!! >.<")

mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["hist.bins"] = "doane"
mpl.rcParams["figure.facecolor"] = mpl.rcParams["axes.facecolor"]

# begin guerrilla patch
original_hist = Axes.hist


def hist_with_gaps(self, x, bins=None, **kwargs):
    if "rwidth" not in kwargs:
        kwargs["rwidth"] = 0.85
    return original_hist(self, x, bins=bins, **kwargs)


Axes.hist = hist_with_gaps
# end guerrilla patch

__version__ = "0.1.0"
