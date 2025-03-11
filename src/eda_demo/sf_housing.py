from eda_demo import (
    Path,
    Any,
    Final,
    stats,
    patch_pickle,
    lowess,
    sns,
    mpl,
    plt,
    EngFormatter,
    np,
    pd,
    Axes,
    Figure,
    NDArray,
    # paths
    SFH_PATH,
)


# csv -> DataFrame
def _parse_dates_and_years(df: pd.DataFrame) -> pd.DataFrame:
    date_format = "%Y-%m-%d"
    column = "date"
    dates = pd.to_datetime(df[column], format=date_format)
    return df.assign(timestamp=dates)


def _df_01() -> pd.DataFrame:
    usecols = pd.Index(
        ["city", "zip", "street", "price", "br", "lsqft", "bsqft", "date"]
    )
    sfh_df: pd.DataFrame = pd.read_csv(
        SFH_PATH, on_bad_lines="skip", usecols=usecols, low_memory=False
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


def _figure_01(df: pd.DataFrame) -> tuple[Figure, list[Axes]]:
    under_04m = df[df["price"] < 4_000_000].copy()
    under_04m["log_price"] = np.log10(under_04m["price"])
    fig: Figure
    axes: list[Axes]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained", sharey=True)
    axes[0].hist(under_04m["price"], bins=50, label="Una venta en ese rango de precio")
    axes[0].legend()
    axes[0].set_title("Precio como tal")
    axes[0].set_ylabel("Conteo")
    axes[0].set_xlabel("(USD)")
    formatter = EngFormatter(places=1, sep="")
    axes[0].xaxis.set_major_formatter(formatter)
    axes[1].hist(
        under_04m["log_price"], bins=50, label="Una venta en ese rango transformado"
    )
    axes[1].legend()
    axes[1].set_title("Precio con transformación log10")
    axes[1].set_xlabel("(log10 USD)")
    fig.suptitle("Distribución de los precios de venta\ncon una partición de bins=50")
    return fig, axes


def figure_01() -> tuple[Figure, list[Axes]]:
    """Muestra dos histogramas yuxtapuestos de los precios de venta. En
    el izquierdo se muestran los precios "como tal" y en el derecho se
    muestran con una transformación log10."""
    df_01 = _df_01()
    fig, axes = _figure_01(df_01)
    return fig, axes


# se concentra el análisis para casa de menos de cuatro millones y menos de doce mil
# pies cuadrados
def _subset(df: pd.DataFrame) -> pd.DataFrame:
    bm_price = df["price"] < 4_000_000
    bm_bsqft = df["bsqft"] < 12_000
    bm_timestamp = df["timestamp"].dt.year == 2004

    return df.loc[bm_price & bm_bsqft & bm_timestamp]


def _df_02() -> pd.DataFrame:
    """Añade la columna `log_bsqft`."""
    df_01 = _df_01()
    df_02 = df_01.pipe(_subset).assign(log_bsqft=np.log10(df_01["bsqft"]))
    return df_02


def _figure_02(df: pd.DataFrame) -> tuple[Figure, list[Axes]]:
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


def figure_02() -> tuple[Figure, list[Axes]]:
    df_02 = _df_02()
    fig, axes = _figure_02(df_02)
    return fig, axes


def _df_03() -> pd.DataFrame:
    """Añade la columna `log_lsqft`."""
    df_02 = _df_02()
    df_03 = df_02.assign(log_lsqft=np.log10(df_02["lsqft"]))
    return df_03


def _figure_03(df: pd.DataFrame) -> tuple[Figure, list[Axes]]:
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


def figure_03() -> tuple[Figure, list[Axes]]:
    df_03 = _df_03()
    fig, axes = _figure_03(df_03)
    return fig, axes


def _df_04() -> pd.DataFrame:
    df_03 = _df_03()
    df_04 = df_03.copy()
    return df_04


def _figure_04(df: pd.DataFrame) -> tuple[Figure, Axes]:
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


def figure_04() -> tuple[Figure, Axes]:
    df_04 = _df_04()
    fig, ax = _figure_04(df_04)
    return fig, ax


# número de cuartos -> característica ordinal (orden)
def _df_05() -> pd.DataFrame:
    df_04 = _df_04()
    df_05 = df_04.copy()
    eight_up = df_05.loc[df_05["br"] >= 8, "br"].unique()
    df_05["new_br"] = df_05["br"].replace(eight_up, 8)
    return df_05


def _figure_05(df: pd.DataFrame) -> tuple[Figure, Axes]:
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


def figure_05() -> tuple[Figure, Axes]:
    df_05 = _df_05()
    fig, ax = _figure_05(df_05)
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


def _df_06() -> pd.DataFrame:
    df_05 = _df_05()
    df_06 = df_05.pipe(_subset).pipe(_log_vals).pipe(_clip_br)
    return df_06


# relación entre el número de habitaciones y el precio
def _figure_06(df: pd.DataFrame) -> tuple[Figure, Axes]:
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
    ax.boxplot(x=boxplot_data, tick_labels=boxplot_ticks, patch_artist=True, **boxprops)  # type: ignore
    ax.set_ylabel("Precio (USD)")
    ax.set_xlabel("Número de cuartos")
    formatter = EngFormatter(places=1, sep="")
    ax.yaxis.set_major_formatter(formatter)
    fig.suptitle("Boxplot de la distribución del precio por número de cuartos")
    return fig, ax


def figure_06() -> tuple[Figure, Axes]:
    """Muestra boxplots del precio de las casas divididos por grupos pertenecientes
    al número de cuartos."""
    df_06 = _df_06()
    fig, ax = _figure_06(df_06)
    return fig, ax


def _df_07() -> pd.DataFrame:
    df_06 = _df_06()
    df_07 = df_06.copy()
    return df_07


def _df_08() -> pd.DataFrame:
    df_07 = _df_07()
    df_08 = df_07.assign(
        ppsf=df_07["price"] / df_07["bsqft"], log_ppsf=lambda df: np.log10(df["ppsf"])
    )
    return df_08


def _figure_08(df: pd.DataFrame) -> tuple[Figure, tuple[Any, Any]]:
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


def figure_08() -> tuple[Figure, tuple[Any, Any]]:
    df_08 = _df_08()
    fig, (ax1, ax2) = _figure_08(df_08)
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


def _df_09() -> pd.DataFrame:
    df_01 = _df_01()
    df_09 = (
        df_01.pipe(_subset)
        .pipe(_log_vals)
        .pipe(_clip_br)
        .pipe(_compute_ppsf)
        .pipe(_make_lamorinda)
    )
    return df_09


# distribution of sale price for these cities
def _figure_09(df: pd.DataFrame) -> tuple[Figure, Axes]:
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
        price_values = city_data.loc[:, "price"].values
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

    ax.boxplot(boxplot_data, patch_artist=True, **boxprops)  # type: ignore
    formatter = EngFormatter(places=1, sep="")
    ax.set_xticklabels(cities)
    ax.set_yscale("log")
    ax.set_ylabel("Precio de venta (USD)")
    ax.yaxis.set_major_formatter(formatter)
    fig.suptitle("Boxplot de las distribuciones del\nprecio de venta por ciudades")
    return fig, ax


def figure_09() -> tuple[Figure, Axes]:
    df_09 = _df_09()
    fig, ax = _figure_09(df_09)
    return fig, ax


def _df_10() -> pd.DataFrame:
    df_09 = _df_09()
    df_10 = df_09.copy()
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
#     "figure_02": figure_02,
#     "figure_03": figure_03,
#     "figure_04": figure_04,
#     "figure_05": figure_05,
#     "figure_06": figure_06,
#     "figure_08": figure_08,
#     "figure_09": figure_09,
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


def main():
    figs = {
        "figure_01": figure_01,
        "figure_02": figure_02,
        "figure_03": figure_03,
        "figure_04": figure_04,
        "figure_05": figure_05,
        "figure_06": figure_06,
        # "figure_07": figure_07,
        "figure_08": figure_08,
        "figure_09": figure_09,
        "figure_10": figure_10,
    }

    figs_dirpath = Path(__file__).parent / "sf_housing_figures"
    figs_dirpath.mkdir(exist_ok=True)
    fl = list(figs.keys())

    for i, func_name in enumerate(fl):
        try:
            fig, _ = figs[func_name]()
            fig.suptitle(
                f"Exploración del mercado de viviendas de SF\nGráfica {func_name.split("_")[1]} de 10"
            )
            print(f"Generando gráfica {func_name}")

            # plt.show()

            save_path = figs_dirpath / f"{func_name}.png"
            fig.savefig(save_path, dpi=300, format="png")
            plt.close()
            print(
                f"\nSe guardó la imagen {func_name}.\n", end="\u2a69" * (1 + i) + "\n"
            )
            print(f"Quedan {len(fl)- 1 - i}")
        except NameError:
            print(f"Todavía no está lista {func_name}")

    print("Se guardaron todas las imágenes.")


if __name__ == "__main__":
    main()
