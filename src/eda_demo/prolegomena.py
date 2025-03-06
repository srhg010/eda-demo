"""No sé hacer boxplots, entonces este lugar es para aprender.
Referencia: https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html"""

from typing import Final
from eda_demo.sf_housing import sfh

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy._core.numerictypes import float64
from numpy.typing import ArrayLike, NDArray
import matplotlib as mpl

from matplotlib.patches import Polygon

plt.style.use("seaborn-v0_8-darkgrid")
mpl.rcParams["figure.facecolor"] = "e6e6e6"
mpl.rcParams["axes.facecolor"] = "e6e6e6"

# sostener el azar en mis manos
np.random.seed(19680801)

# generar valores
spread: NDArray[float64] = np.random.rand(50) * 100
center: NDArray = np.ones(25) * 50
flier_high: NDArray[float64] = np.random.rand(10) * 100 + 100
flier_low: NDArray[float64] = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low))

N: Final[int] = 500
norm: Final[NDArray[float64]] = np.random.normal(1, 1, N)
logn: Final[NDArray[float64]] = np.random.lognormal(1, 1, N)
expo: Final[NDArray[float64]] = np.random.exponential(1, N)
gumb: Final[NDArray[float64]] = np.random.gumbel(6, 4, N)
tria: Final[NDArray[float64]] = np.random.triangular(2, 9, 11, N)
df_distributions = pd.DataFrame(
    {"norm": norm, "logn": logn, "expo": expo, "gumb": gumb, "tria": tria}
)
fig: Figure
axs: NDArray
fig, axs = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
sym = "+"

# sym = r"卐"
# básico
axs[0, 0].set_title("Sequence (list) of 1D arrays (NDArray).")
axs[0, 0].boxplot([norm, logn], sym=sym)
axs[0, 1].set_title("DataFrame of 5 columns")
axs[0, 1].boxplot(df_distributions, sym=sym)
axs[0, 1].set_xticklabels(df_distributions.columns, rotation=45)
axs[0, 2].set_title("Slice of 2 columns of previous DataFrame")
axs[0, 2].boxplot(df_distributions[["norm", "logn"]], sym=sym)
axs[1, 0].set_title("Gumbel(6, 4)")
axs[1, 0].boxplot(gumb, sym=sym)
axs[1, 1].set_title("Triangular(2, 9, 11)")
axs[1, 1].boxplot(tria, sym=sym)
axs[1, 2].set_title("N/A")

plt.show()
