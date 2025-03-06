from pathlib import Path
from typing import Any, Final

from scipy import stats
from pandas.compat.pickle_compat import patch_pickle
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

plt.style.use("seaborn-v0_8-darkgrid")
mpl.rcParams["figure.facecolor"] = "e6e6e6"
mpl.rcParams["axes.facecolor"] = "e6e6e6"

# paths
datasets_path: Final[Path] = Path("/home/zerserob/Documents/Projects/Python/datasets")
sfh_path: Final[Path] = datasets_path / "sfhousing.csv"


# csv -> DataFrame
def _parse_dates_and_years(df: pd.DataFrame) -> pd.DataFrame:
    date_format = "%Y-%m-%d"
    column = "date"
    dates = pd.to_datetime(df[column], format=date_format)
    return df.assign(timestamp=dates)


def _df_1() -> pd.DataFrame:
    usecols = pd.Index(
        ["city", "zip", "street", "price", "br", "lsqft", "bsqft", "date"]
    )
    sfh_df: pd.DataFrame = pd.read_csv(
        sfh_path, on_bad_lines="skip", usecols=usecols, low_memory=False
    )
    sfh_df["price"] = pd.to_numeric(sfh_df.loc[:, "price"], errors="coerce")
    sfh_df = sfh_df.dropna(subset="price")
    sfh_df = sfh_df.pipe(_parse_dates_and_years)
    # ya no se necesita esta columna, pues se creó otra
    # con un tipo de datos óptimo
    sfh_df = sfh_df.drop(["date"], axis=1)
    return sfh_df


# Understanding Price
# detener la notación científica
pd.set_option("display.float_format", "{:.2f}".format)


def inspect_percentile(
    df: pd.DataFrame, *, col: str, mode: str, dropna: bool = False
) -> pd.DataFrame:
    percs_dict = {
        "regular": [0, 25, 50, 75, 100],
        "upper_zoom": [95, 97, 98, 99, 99.5, 99.9],
        "lower_zoom": [0.5, 1, 1.5, 2, 2.5, 3],
    }
    percs = percs_dict[mode]
    percs_index = pd.Index(percs)
    percs_values: NDArray
    if not dropna:
        percs_values = np.percentile(df[col], percs, method="lower")
    else:
        percs_values = np.percentile(df[col].dropna(), percs, method="lower")
    increment = np.append(0, np.diff(percs_values))
    return pd.DataFrame(
        {
            col: dict(zip(percs_index, percs_values)),
            "increment": dict(zip(percs_index, increment)),
        }
    )


def _figure_1(df: pd.DataFrame) -> tuple[Figure, list[Axes]]:
    under_4m = df[df["price"] < 4_000_000].copy()
    under_4m["log_price"] = np.log10(under_4m["price"])
    fig: Figure
    axes: list[Axes]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained", sharey=True)
    axes[0].hist(under_4m["price"], bins=50, label="Una venta en ese rango de precio")
    axes[0].legend()
    axes[0].set_title("Precio como tal")
    axes[0].set_ylabel("Conteo")
    axes[0].set_xlabel("(USD)")
    formatter = EngFormatter(places=1, sep="")
    axes[0].xaxis.set_major_formatter(formatter)
    axes[1].hist(
        under_4m["log_price"], bins=50, label="Una venta en ese rango transformado"
    )
    axes[1].legend()
    axes[1].set_title("Precio con transformación log10")
    axes[1].set_xlabel("(log10 USD)")
    fig.suptitle("Distribución de los precios de venta\ncon una partición de bins=50")
    return fig, axes


def figure_1() -> tuple[Figure, list[Axes]]:
    """Muestra dos histogramas yuxtapuestos de los precios de venta. En
    el izquierdo se muestran los precios "como tal" y en el derecho se
    muestran con una transformación log10."""
    df_1 = _df_1()
    fig, axes = _figure_1(df_1)
    return fig, axes


# se concentra el análisis para casa de menos de cuatro millones y menos de doce mil
# pies cuadrados
def _subset(df: pd.DataFrame) -> pd.DataFrame:
    bm_price = df["price"] < 4_000_000
    bm_bsqft = df["bsqft"] < 12_000
    bm_timestamp = df["timestamp"].dt.year == 2004

    return df.loc[bm_price & bm_bsqft & bm_timestamp]


def _df_2() -> pd.DataFrame:
    """Añade la columna `log_bsqft`."""
    df_1 = _df_1()
    df_2 = df_1.pipe(_subset).assign(log_bsqft=np.log10(df_1["bsqft"]))
    return df_2


def _figure_2(df: pd.DataFrame) -> tuple[Figure, list[Axes]]:
    fig: Figure
    axes: list[Axes]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained", sharey=True)
    axes[0].hist(df["bsqft"], bins=50, label="bins=50")
    axes[0].legend()
    axes[0].set_title(r"Tamaño de construcción en (ft$^{2}$)")
    axes[0].set_xlabel(r"ft$^{2}$")
    axes[0].set_ylabel("Conteo")
    formatter = EngFormatter(places=1, sep="")
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[1].hist(df["log_bsqft"], bins=50, label="bins=50")
    axes[1].set_xlabel(r"log10 ft$^{2}$")
    axes[1].legend()
    axes[1].set_title(r"Tamaño de construcción en (log10 ft$^{2}$)")
    fig.suptitle("Histogramas del tamaño de construcción")
    return fig, axes


def figure_2() -> tuple[Figure, list[Axes]]:
    df_2 = _df_2()
    fig, axes = _figure_2(df_2)
    return fig, axes


def _df_3() -> pd.DataFrame:
    """Añade la columna `log_lsqft`."""
    df_2 = _df_2()
    df_3 = df_2.assign(log_lsqft=np.log10(df_2["lsqft"]))
    return df_3


def _figure_3(df: pd.DataFrame) -> tuple[Figure, list[Axes]]:
    fig: Figure
    axes: list[Axes]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained")
    axes[0].scatter(
        x=df["bsqft"],
        y=df["lsqft"],
        s=0.4,
        alpha=0.5,
    )
    formatter = EngFormatter(places=1, sep="")
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].set_xlabel(r"Tamaño de construcción (ft$^{2}$)")
    axes[0].set_ylabel(r"Tamaño del terreno (ft$^{2}$)")
    axes[0].set_title("Normal")
    axes[1].scatter(
        x=df["log_bsqft"],
        y=df["log_lsqft"],
        s=0.4,
        alpha=0.5,
    )
    axes[1].set_xlabel(r"Tamaño de construcción (log10 ft$^{2}$)")
    axes[1].set_ylabel(r"Tamaño del terreno (log10 ft$^{2}$)")
    axes[1].set_title("Transformación log10")
    fig.suptitle("Diagrama de dispersion: x=construcción vs y=terreno")
    return fig, axes


def figure_3() -> tuple[Figure, list[Axes]]:
    df_3 = _df_3()
    fig, axes = _figure_3(df_3)
    return fig, axes


def _df_4() -> pd.DataFrame:
    df_3 = _df_3()
    df_4 = df_3.copy()
    return df_4


def _figure_4(df: pd.DataFrame) -> tuple[Figure, Axes]:
    br_cat = df["br"].value_counts().reset_index()
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(7 * 1.4, 7), layout="constrained")
    ax.bar(x=br_cat["br"], height=br_cat["count"])
    formatter = EngFormatter(places=1, sep="")
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("Número de cuartos")
    ax.set_ylabel("Conteo")
    fig.suptitle(
        "Gráfica de barras del número de casas vendidas\ndivididas por el número de cuartos"
    )
    return fig, ax


def figure_4() -> tuple[Figure, Axes]:
    df_4 = _df_4()
    fig, ax = _figure_4(df_4)
    return fig, ax


# número de cuartos -> característica ordinal (orden)
def _df_5() -> pd.DataFrame:
    df_4 = _df_4()
    df_5 = df_4.copy()
    eight_up = df_5.loc[df_5["br"] >= 8, "br"].unique()
    df_5["new_br"] = df_5["br"].replace(eight_up, 8)
    return df_5


def _figure_5(df: pd.DataFrame) -> tuple[Figure, Axes]:
    br_cat = df["new_br"].value_counts().reset_index()
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(7 * 1.4, 7), layout="constrained")
    ax.bar(
        x=br_cat["new_br"],
        height=br_cat["count"],
        tick_label=[
            "3",
            "2",
            "4",
            "5",
            "1",
            "6",
            "8+",
            "7",
        ],  # el orden está raro
    )
    formatter = EngFormatter(places=1, sep="")
    ax.yaxis.set_major_formatter(formatter)
    ax.set_title("Las casas de 8 cuartos o más se juntan en\nuna sola categoría: 8+")
    fig.suptitle(
        "Gráfica de barras del número de casas vendidas\ndivididas por el número de cuartos"
    )
    return fig, ax


def figure_5() -> tuple[Figure, Axes]:
    df_5 = _df_5()
    fig, ax = _figure_5(df_5)
    return fig, ax


# estas dos funciones son lo mismo que ya habíamos hecho, pero
# se guardan para meterlo en un pipeline
def _log_vals(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        log_price=np.log10(df["price"]),
        log_bsqft=np.log10(df["bsqft"]),
        log_lsqft=np.log10(df["lsqft"]),
    )


def _clip_br(df: pd.DataFrame) -> pd.DataFrame:
    eight_up = df.loc[df["br"] >= 8, "br"].unique()
    new_br = df["br"].replace(eight_up, 8)
    return df.assign(new_br=new_br)


def _df_6() -> pd.DataFrame:
    df_5 = _df_5()
    df_6 = df_5.pipe(_subset).pipe(_log_vals).pipe(_clip_br)
    return df_6


# relación entre el número de habitaciones y el precio
def _figure_6(df: pd.DataFrame) -> tuple[Figure, Axes]:
    """
         Q1-1.5IQR   Q1   median  Q3   Q3+1.5IQR
                      |-----:-----|
      o      |--------|     :     |--------|    o  o
                      |-----:-----|
    flier             <----------->            fliers
                           IQR
    """
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(7 * 1.4, 7), layout="constrained")
    boxplot_data = []
    boxplot_ticks = []
    for k, g in df.groupby("new_br"):
        boxplot_data.append(g["price"].to_numpy())
        boxplot_ticks.append(k)
    boxplot_ticks[-1] = "8+"
    boxprops = {
        "flierprops": {
            "alpha": 0.3,
            "marker": ".",
            "markerfacecolor": "black",
            "markeredgecolor": "none",
            "markersize": 4,
        }
    }
    ax.boxplot(x=boxplot_data, tick_labels=boxplot_ticks, patch_artist=True, **boxprops)
    ax.set_ylabel("Precio (USD)")
    ax.set_xlabel("Número de cuartos")
    formatter = EngFormatter(places=1, sep="")
    ax.yaxis.set_major_formatter(formatter)
    fig.suptitle("Boxplot de la distribución del precio por número de cuartos")
    return fig, ax


def figure_6() -> tuple[Figure, Axes]:
    """Muestra boxplots del precio de las casas divididos por grupos pertenecientes
    al número de cuartos."""
    df_6 = _df_6()
    fig, ax = _figure_6(df_6)
    return fig, ax


def _df_7() -> pd.DataFrame:
    df_6 = _df_6()
    df_7 = df_6.copy()
    return df_7


# def _figure_7(
#     data: pd.DataFrame,
#     x_col: str,
#     y_col: str,
#     log_y: bool = False,
#     width: float = 8.0 * 1.4,
#     height: float = 8.0,
#     x_label: str | None = None,
#     y_label: str | None = None,
# ) -> tuple[Figure, Axes]:
#     figsize = (width, height)
#     fig, ax = plt.subplots(figsize=figsize, layout="constrained")
#     new_br = data["new_br"]

#     grouped_data = [
#         data[data[x_col] == val].loc[:, y_col].dropna().values
#         for val in sorted(data[x_col].unique())
#     ]

#     ax.boxplot(grouped_data)
#     ax.set_xticklabels(sorted(data[x_col].unique()))
#     if log_y:
#         ax.set_yscale("log")
#     ax.set_xlabel(x_label if x_label else x_col)
#     ax.set_ylabel(y_label if y_label else y_col)
#     return fig, ax


# def figure_7() -> None:
#     """Esto es escoria."""
#     df_7 = _df_7()
#     x_col = "bsqft"
#     y_col = "price"
#     fig, ax = _figure_7(df_7, x_col, y_col)
#     plt.show()
#     return None


def _df_8() -> pd.DataFrame:
    df_7 = _df_7()
    df_8 = df_7.assign(
        ppsf=df_7["price"] / df_7["bsqft"], log_ppsf=lambda df: np.log10(df["ppsf"])
    )
    return df_8


def _figure_8(df: pd.DataFrame) -> tuple[Figure, tuple[Any, Any]]:
    color1 = sns.color_palette()[1]
    color2 = sns.color_palette()[2]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))

    sns.scatterplot(data=df, x="log_bsqft", y="log_price", alpha=0.1, s=20, ax=ax1)
    xs = np.linspace(2.5, 4, 100)
    curve = lowess(
        df["log_price"], df["log_bsqft"], frac=1 / 10, xvals=xs, return_sorted=False
    )
    ax1.plot(xs, curve, color=color1, linewidth=3)
    plt.setp(
        ax1,
        xlabel=r"Tamaño de construcción (log10 ft$^{2}$)",
        ylabel="Precio (log10 USD)",
    )

    ppsf = df.assign(
        ppsf=df["price"] / df["bsqft"], log_ppsf=lambda df: np.log10(df["ppsf"])
    )
    sns.scatterplot(
        data=ppsf,
        x="bsqft",
        y="log_ppsf",
        hue="br",
        legend=False,
        alpha=0.1,
        s=20,
        ax=ax2,
    )
    xs = np.linspace(200, 6_000, 100)
    curve = lowess(
        ppsf["log_ppsf"], ppsf["bsqft"], frac=1 / 10, xvals=xs, return_sorted=False
    )
    ax2.plot(xs, curve, color=color2, linewidth=3)
    plt.setp(
        ax2,
        xlabel=r"Tamaño de construcción (ft$^{2}$)",
        ylabel=r"Precio por ft$^{2}$ (log10 $\frac{\text{USD}}{\text{ft}^{2}}$)",
    )
    formatter = EngFormatter(places=1, sep="")
    ax2.xaxis.set_major_formatter(formatter)
    fig.suptitle("Diagramas de dispersion con curva de regresión ponderada localmente")

    return fig, (ax1, ax2)


def figure_8() -> tuple[Figure, tuple[Any, Any]]:
    df_8 = _df_8()
    fig, (ax1, ax2) = _figure_8(df_8)
    return fig, (ax1, ax2)


def _compute_ppsf(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        ppsf=(df["price"] / df["bsqft"]), log_ppsf=lambda df: np.log10(df["ppsf"])
    )


# Fixing Location
def _make_lamorinda(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(
        {
            "city": {
                "Lafayette": "Lamorinda",
                "Moraga": "Lamorinda",
                "Orinda": "Lamorinda",
            }
        }
    )


def _df_9() -> pd.DataFrame:
    df_1 = _df_1()
    df_9 = (
        df_1.pipe(_subset)
        .pipe(_log_vals)
        .pipe(_clip_br)
        .pipe(_compute_ppsf)
        .pipe(_make_lamorinda)
    )
    return df_9


# distribution of sale price for these cities
def _figure_9(df: pd.DataFrame) -> tuple[Figure, Axes]:
    cities = [
        "Richmond",
        "El Cerrito",
        "Albany",
        "Berkeley",
        "Walnut Creek",
        "Lamorinda",
        "Piedmont",
    ]
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6 * 1.8, 6), layout="constrained")
    query: pd.DataFrame = df.query("city in @cities")

    boxplot_data = []
    for city in cities:
        city_data = query[query["city"] == city]
        price_values = city_data["price"].values
        boxplot_data.append(price_values)
    boxprops = {
        "flierprops": {
            "alpha": 0.3,
            "marker": ".",
            "markerfacecolor": "black",
            "markeredgecolor": "none",
            "markersize": 4,
        }
    }

    ax.boxplot(boxplot_data, patch_artist=True, **boxprops)
    formatter = EngFormatter(places=1, sep="")
    ax.set_xticklabels(cities)
    ax.set_yscale("log")
    ax.set_ylabel("Precio de venta (USD)")
    ax.yaxis.set_major_formatter(formatter)
    fig.suptitle("Boxplot de las distribuciones del\nprecio de venta por ciudades")
    return fig, ax


def figure_9() -> tuple[Figure, Axes]:
    df_9 = _df_9()
    fig, ax = _figure_9(df_9)
    return fig, ax


def _df_10() -> pd.DataFrame:
    df_9 = _df_9()
    df_10 = df_9.copy()
    return df_10


def _figure_10(df: pd.DataFrame) -> tuple[Figure, Any]:
    four_cities = [
        "Berkeley",
        "Lamorinda",
        "Piedmont",
        "Richmond",
    ]
    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(8 * 1.125, 8),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    query = df.query("city in @four_cities")
    fig.suptitle(r"Price per $\text{ft}^{2}$")
    i_axes = [[0, 0], [1, 0], [0, 1], [1, 1]]
    for (i_axx, i_axy), (name, group) in zip(i_axes, query.groupby("city")):
        ax[i_axx, i_axy].scatter(x=group["bsqft"], y=group["log_ppsf"], alpha=0.2)
        slope, intercept, _, _, _ = stats.linregress(group["bsqft"], group["log_ppsf"])
        xs = np.linspace(group["bsqft"].min(), group["bsqft"].max(), 100)
        ax[i_axx, i_axy].plot(
            xs, slope * xs + intercept, c="k", linestyle="dashed", linewidth=0.5
        )
        ax[i_axx, i_axy].set_title(name)
    formatter = EngFormatter(places=1, sep="")
    ax[1, 0].xaxis.set_major_formatter(formatter)
    ax[1, 0].set_xlabel(r"Tamaño de construcción (ft$^{2}$)")
    ax[1, 0].set_ylabel(r"Precio por ft$^{2}$ (log10 $\frac{USD}{\text{ft}^{2}}$)")
    ax[1, 1].xaxis.set_major_formatter(formatter)
    ax[1, 1].set_xlabel(r"Tamaño de construcción (ft$^{2}$)")
    ax[0, 0].set_ylabel(r"Precio por ft$^{2}$ (log10 $\frac{USD}{\text{ft}^{2}}$)")
    fig.suptitle(
        "Gráfico de dispersión del tamaño de construcción contra\nel precio por pie cuadrado en escala log10"
    )

    return fig, ax


def figure_10() -> tuple[Figure, Any]:
    df_10 = _df_10()
    fig, ax = _figure_10(df_10)
    return fig, ax


# figs = {
#     "figure_1": figure_1,
#     "figure_2": figure_2,
#     "figure_3": figure_3,
#     "figure_4": figure_4,
#     "figure_5": figure_5,
#     "figure_6": figure_6,
#     "figure_8": figure_8,
#     "figure_9": figure_9,
#     "figure_10": figure_10,
# }

# figs_dirpath = Path(".").resolve() / "sf_figures"
# figs_dirpath.mkdir(exist_ok=True)
# fl = list(figs.keys())

# for i, func_name in enumerate(fl):
#     fig, _ = figs[func_name]()
#     save_path = figs_dirpath.name + func_name + ".png"
#     fig.savefig(save_path, dpi=300)
#     print(f"\nSe guardó la imagen {func_name}.\n", end="\u2a69" * 43)
#     print(f"Quedan {len(fl)- 1 - i}")

# print("Se guardaron todas las imágenes.")
