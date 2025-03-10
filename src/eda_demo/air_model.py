from eda_demo import (
    AQSSITES_PATH,
    CLEANED24_PATH,
    PA_CSVS,
    PASENSORS_PATH,
    SACRAMSENSOR_PATH,
    Any,
    Axes,
    DataFrame,
    Figure,
    LinearRegression,
    NDArray,
    Path,
    Sequence,
    Series,
    json,
    lowess,
    mdates,
    mpl,
    np,
    pd,
    plt,
    sqlalchemy,
    stats,
)

# plt.style.use("seaborn-v0_8-darkgrid")
# mpl.rcParams["figure.facecolor"] = "e6e6e6"
# mpl.rcParams["axes.facecolor"] = "e6e6e6"

# paths -> dataframes
aqs_sites_full = pd.read_csv(AQSSITES_PATH)


id_counts = aqs_sites_full["AQS_Site_ID"].value_counts()
dup_site = aqs_sites_full.query("AQS_Site_ID == '19-163-0015'")
some_cols = [
    "POC",
    "Monitor_Start_Date",
    "Last_Sample_Date",
    "Sample_Collection_Method",
]
columns_descriptions = {
    "Column": "Description",
    "date": " Date of the observation",
    "id": " A unique label for a site, formatted as the US state abbreviation with a number (we performed data cleaning for site ID CA1)",
    "region": " The name of the region, which corresponds to a group of sites (the CA1 site is located in the West region)",
    "pm25aqs": "The PM2.5 measurement from the AQS sensor",
    "pm25pa": "The PM2.5 measurement from the PurpleAir sensor",
    "temp": " Temperature, in Celsius",
    "rh": " Relative humidity, ranging from 0% to 100%",
    "dew": " The dew point (a higher dew point means more moisture is in the air)",
}


def _full_df() -> pd.DataFrame:
    # así es como me gusta escribir el proceso de:
    #  1. leer un csv
    #  2. usar sólo ciertas columnas
    #  3. renombrar las columnas
    usecols = np.array(
        ["Date", "ID", "region", "PM25FM", "PM25cf1", "TempC", "RH", "Dewpoint"]
    )
    renamer_map = {
        "Date": "date",
        "ID": "id",
        "region": "region",
        "PM25FM": "pm25aqs",
        "PM25cf1": "pm25pa",
        "TempC": "temp",
        "RH": "rh",
        "Dewpoint": "dew",
    }
    full_df = (
        pd.read_csv(CLEANED24_PATH, usecols=usecols, parse_dates=["Date"])
        .rename(columns=renamer_map)
        .dropna()
    )
    return full_df


# Agrupamos los datos con base en una cierta granularidad:
# un instrumento de cada sitio de AQS; sin pérdida de generalidad
# eligiendo el primero de cada sitio.
def _rollup_dup_sites(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("AQS_Site_ID").first().reset_index()


# emparejar AQS con PurpleAir
def _cols_aqs(df: pd.DataFrame) -> pd.DataFrame:
    # columnas deseadas
    sub_cols: list[str] = ["AQS_Site_ID", "Latitude", "Longitude"]
    # se seleccionan
    subset: pd.DataFrame = df.loc[:, sub_cols]
    # se renombran
    subset.columns = ["site_id", "lat", "lon"]
    return subset


def _cols_pa(df: pd.DataFrame) -> pd.DataFrame:
    sub_cols = ["ID", "Label", "Lat", "Lon"]
    subset: pd.DataFrame = df.loc[:, sub_cols]
    subset.columns = ["id", "label", "lat", "lon"]
    return subset


def _rollup_dates(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("date_local").first().reset_index()


# parsing dates
def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_format = "%Y-%m-%d"
    timestamps = pd.to_datetime(df["date_local"], format=date_format)
    return df.assign(date_local=timestamps)


def _drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    sub_cols = ["date_local", "arithmetic_mean"]
    subset: pd.DataFrame = df.loc[:, sub_cols]
    return subset.rename(columns={"arithmetic_mean": "pm25"})


def _drop_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    sub_cols = ["created_at", "PM2.5_CF1_ug/m3", "Temperature_F", "Humidity_%"]
    df_sub: pd.DataFrame = df.loc[:, sub_cols]
    df_sub.columns = ["timestamp", "PM25cf1", "TempF", "RH"]
    return df_sub


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    date_format = "%Y-%m-%d %X %Z"
    times = pd.to_datetime(df["timestamp"], format=date_format)
    return df.assign(timestamp=times).set_index("timestamp")


def _convert_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Cambiar el uso horario de PurpleAir a US/Pacific"""
    return df.tz_convert("US/Pacific")


def _drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame | Any:
    return df[~df.index.duplicated()]


def _compute_daily_avgs(df: pd.DataFrame) -> pd.DataFrame:
    # esto ya no se usó
    def _has_enough_readings(one_day: pd.Series) -> bool:
        # 24 hour avgs
        needed_measurements_80s = 0.9 * 1080
        needed_measurements_120s = 0.9 * 720
        cutoff_date = pd.Timestamp("2019-05-30", tz="US/Pacific")
        [n] = one_day
        date = one_day.name
        return (
            n >= needed_measurements_80s
            if date <= cutoff_date
            else n >= needed_measurements_120s
        )

    should_keep = (
        df.resample("D")["PM25cf1"]
        .size()
        .to_frame()
        .apply(_has_enough_readings, axis="columns")
    )
    return df.resample("D").mean().loc[should_keep]


def _aqs_full() -> pd.DataFrame:
    return pd.read_csv(SACRAMSENSOR_PATH)


def wrangling_prolegomena():
    """modelo preliminar; no se usa"""
    # ahora trabajamos con PurpleAir
    m1: float
    m2: float
    b: float
    coefs: np.ndarray
    aqs_sites = aqs_sites_full.pipe(_rollup_dup_sites).pipe(_cols_aqs)
    with open(PASENSORS_PATH) as f:
        pa_json = json.load(f)
    pa_sites_full = pd.DataFrame(pa_json["data"], columns=pa_json["fields"])
    pa_sites = pa_sites_full.pipe(_cols_pa)
    # vecindad
    magic_meters_per_lat = 111_111
    offset_in_m = 25
    offset_in_lat = offset_in_m / magic_meters_per_lat
    median_latitude = aqs_sites["lat"].median()
    magic_meters_per_lon = 111_111 * np.cos(np.radians(median_latitude))
    offset_in_lon = offset_in_m / magic_meters_per_lon
    # hacemos una base de datos en SQL
    db = sqlalchemy.create_engine("sqlite://")
    # se crean dos tablas
    aqs_sites.to_sql(name="aqs", con=db, index=False)
    pa_sites.to_sql(name="pa", con=db, index=False)
    query = f"""
    SELECT
    aqs.site_id AS aqs_id,
    pa.id AS pa_id,
    pa.label AS pa_label,
    aqs.lat AS aqs_lat,
    aqs.lon AS aqs_lon,
    pa.lat AS pa_lat,
    pa.lon AS pa_lon
    FROM aqs JOIN pa
    ON pa.lat - {offset_in_lat} <= aqs.lat
    AND aqs.lat <= pa.lat + {offset_in_lat}
    AND pa.lon - {offset_in_lon} <= aqs.lon
    AND aqs.lon <= pa.lon + {offset_in_lon}
    """
    matched = pd.read_sql(query, db)
    aqs_full = _aqs_full()
    full_df = _full_df()
    aqs_date_counts = aqs_full["date_local"].value_counts()
    aqs = aqs_full.pipe(_rollup_dates).pipe(_drop_cols).pipe(_parse_dates)
    date_range: pd.Timedelta = aqs["date_local"].max() - aqs["date_local"].min()
    # Particulate Matter (PM) - micrograms/m^3 - rango: (0,2.5)

    # PA = b + mAQS + error
    # True air quality = -(b/m) + (1/m)PA + error

    # una variable
    AQS, PA = full_df[["pm25aqs"]], full_df[["pm25pa"]]
    model = LinearRegression().fit(AQS, PA)
    m, b = model.coef_[0], model.intercept_
    print(f"True air quality estimate = {-b/m} + {1/m}PA")

    # dos variables
    AQS_RH, PA = full_df[["pm25aqs", "rh"]], full_df["pm25pa"]
    model_h = LinearRegression().fit(AQS_RH, PA)
    coefs = model_h.coef_
    [m1, m2], b = coefs, model_h.intercept_
    return None


# desde aquí se empieza a trabajar con el csv final


def _pa_full() -> pd.DataFrame:
    return pd.read_csv(PA_CSVS[0])


def _df_01() -> pd.DataFrame:
    """aqs con tres pipes:
    rollup_dates, drop_cols, parse_dates"""
    aqs_full = _aqs_full()
    df_01 = aqs_full.pipe(_rollup_dates).pipe(_drop_cols).pipe(_parse_dates)
    return df_01


def _figure_01(
    data: Any,
    date_col="date_local",
    pm25_col="pm25",
    figsize=(10, 6),
) -> tuple[Figure, Axes]:
    """Gráfica de dispersión PM 2.5"""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.scatter(data[date_col], data[pm25_col])
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Promedio diario: AQS PM2.5")
    fig.suptitle("Calibración de sensores de PM2.5")
    return fig, ax


def figure_01() -> tuple[Figure, Axes]:
    df_01 = _df_01()
    fig, ax = _figure_01(df_01)
    return fig, ax


def _df_02() -> pd.DataFrame:
    pa_full = _pa_full()
    pa = pa_full.pipe(_drop_and_rename_cols).pipe(_parse_timestamps).pipe(_convert_tz)
    df_02 = (
        pa.resample("d")
        .size()
        .to_frame()
        .rename({0: "records_per_day"}, axis="columns")
    )
    return df_02


def _figure_02(
    data: Any, y_col: str = "records_per_day", figsize: tuple[int, int] = (10, 6)
) -> tuple[Figure, Axes]:
    """Registros diarios."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.plot(data.index, data[y_col])
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Registros por día")
    fig.suptitle("Calibración de sensores de PM2.5")
    return fig, ax


def figure_02() -> tuple[Figure, Axes]:
    df_02 = _df_02()
    fig, ax = _figure_02(df_02)
    return fig, ax


def _df_03() -> pd.DataFrame:
    pa_full = _pa_full()
    pa = (
        pa_full.pipe(_drop_and_rename_cols)
        .pipe(_parse_timestamps)
        .pipe(_convert_tz)
        .pipe(_drop_duplicate_rows)
    )
    df_03 = (
        pa.resample("D")
        .size()
        .to_frame()
        .rename({0: "records_per_day"}, axis="columns")
    )
    return df_03


def _figure_03(
    data: Any,
    y_col: str = "records_per_day",
    vline_date: str = "2019-05-30",
    annotation_date: str = "2019-07-24",
    upper_limit: float = 1080,
    lower_limit: float = 720,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure, Axes]:
    """Registros diarios con anotaciones."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data[y_col])
    ax.axhline(y=upper_limit, linestyle=":", linewidth=3, alpha=0.6)
    ax.axhline(y=lower_limit, linestyle=":", linewidth=3, alpha=0.6)
    ax.axvline(
        x=float(mdates.datestr2num(vline_date)), linestyle="--", linewidth=3, alpha=0.6
    )
    annotation_x = float(mdates.datestr2num(d=annotation_date))
    ax.annotate(
        str(int(upper_limit)),
        xy=(annotation_x, upper_limit),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
    )
    ax.annotate(
        str(int(lower_limit)),
        xy=(annotation_x, lower_limit),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Registros por día")
    fig.suptitle("Calibración de sensores de PM2.5")
    return fig, ax


def figure_03() -> tuple[Figure, Axes]:
    df_03 = _df_03()
    fig, ax = _figure_03(df_03)
    return fig, ax


def _nc4_ts_nc4() -> tuple[pd.DataFrame, pd.DataFrame]:
    """nc4ts_nc4"""
    full_df = _full_df()
    nc4 = full_df.loc[full_df["id"] == "NC4"]
    ts_nc4 = (
        nc4.set_index("date").resample("W")["pm25aqs", "pm25pa"].mean().reset_index()
    )
    return nc4, ts_nc4


def _df_04() -> pd.DataFrame:
    _, df_04 = _nc4_ts_nc4()
    return df_04


def _figure_04(df: pd.DataFrame) -> tuple[Figure, Axes]:
    x_srs = "date"
    y_srs = "pm25aqs"
    yalt_srs = "pm25pa"
    y_limit = [-2, 32]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.plot(df[x_srs], df[y_srs], label=y_srs, linewidth=2)
    ax.plot(
        df[x_srs],
        df[yalt_srs],
        color="black",
        linestyle="dotted",
        linewidth=2,
        label=yalt_srs,
    )
    for i in range(0, 35, 5):
        ax.axhline(i, c="k", alpha=0.1)
    # ax.axhline(0, c="k", alpha=0.2)
    # ax.axhline(5, c="k", alpha=0.2)
    ax.set_ylim(*y_limit)
    ax.set_xlabel("Fechas")
    ax.set_ylabel("Promedios semanales de PM2.5")
    fig.suptitle("Comparación de promedios semanales de PM2.5\nentre PurpleAir y AQS")
    return fig, ax


def figure_04() -> tuple[Figure, Axes]:
    df_04 = _df_04()
    fig, ax = _figure_04(df_04)
    return fig, ax


def _df_05() -> pd.DataFrame:
    df_05, _ = _nc4_ts_nc4()
    return df_05


def _figure_05(df: pd.DataFrame) -> tuple[Figure, list[Axes]]:
    fig: Figure
    axs: list[Axes]
    x_srs = "pm25aqs"
    xalt_srs = "pm25pa"

    fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
    axs[0].hist(df[x_srs], bins="doane", density=True)
    axs[0].set_title("Distribución de mediciones de AQS")
    axs[1].hist(df[xalt_srs], bins="doane", density=True, alpha=0.8)
    axs[1].set_title("Distribución de mediciones de PurpleAir")

    return fig, axs


def figure_05() -> tuple[Figure, list[Axes]]:
    df_05 = _df_05()
    fig, axs = _figure_05(df_05)
    return fig, axs


def _df_06() -> pd.DataFrame:
    """nc4"""
    full_df = _full_df()
    df_06 = full_df.loc[full_df["id"] == "NC4"]
    return df_06


def _figure_06(
    data: pd.DataFrame,
    aqs_col: str = "pm25aqs",
    pa_col: str = "pm25pa",
    figsize: tuple[int, int] = (10, 6),
    line_coords: tuple[tuple[float, float], tuple[float, float]] = ((2, 1), (13, 25)),
) -> tuple[Figure, Axes]:
    """Comparación de cuantiles"""
    percs: np.ndarray = np.arange(1, 100, 1)
    aqs_qs: np.ndarray = np.percentile(data[aqs_col], percs, method="lower")
    pa_qs: np.ndarray = np.percentile(data[pa_col], percs, method="lower")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.scatter(aqs_qs, pa_qs)
    (x1, y1), (x2, y2) = line_coords
    ax.plot(
        [x1, x2],
        [y1, y2],
        linestyle="--",
        linewidth=4,
        color="tab:orange",
        label="pendiente de 2.2",
    )
    ax.legend()
    ax.set_xlabel("Cuantiles de AQS")
    ax.set_ylabel("Cuantiles de PurpleAir")
    ax.set_title("Comparación de cuantiles")
    return fig, ax


def figure_06() -> tuple[Figure, Axes]:
    df_06 = _df_06()
    fig, ax = _figure_06(df_06)
    return fig, ax


def _df_07() -> pd.DataFrame:
    full_df = _full_df()
    nc4 = full_df.loc[full_df["id"] == "NC4"]
    df_07 = nc4["pm25pa"] - nc4["pm25aqs"]
    return df_07


def _figure_07(df: pd.DataFrame) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(6 * 1.4, 6), layout="constrained")
    ax.hist(df, bins="doane")
    ax.set_title("Distribución de las diferencias de medición entre PA y AQS")
    ax.set_ylabel("Conteo")
    ax.set_xlabel(r"Diferencia entre las lecturas de los sensores de PA y AQS")
    return fig, ax


def figure_07() -> tuple[Figure, Axes]:
    df_07 = _df_07()
    fig, ax = _figure_07(df_07)
    return fig, ax


def _df_10() -> pd.DataFrame:
    quid_cols = ["Date", "ID", "region", "PM25FM", "PM25cf1"]
    quo_cols = ["date", "id", "region", "pm25aqs", "pm25pa"]
    cols_hash = dict(zip(quid_cols, quo_cols))
    full: pd.DataFrame = pd.read_csv(
        CLEANED24_PATH, usecols=np.array(quid_cols)
    ).rename(columns=cols_hash)
    df_10: pd.DataFrame = full.loc[(full["id"] == "GA1"), :]
    return df_10


def _figure_10(
    data: DataFrame,
    x_col: str = "pm25aqs",
    y_col: str = "pm25pa",
    figsize: tuple[float, float] = (12.5, 8.5),
) -> tuple[Figure, Axes]:
    """Dispersión con regresión lineal trivial"""
    slope, intercept, *(_) = stats.linregress(data[x_col], data[y_col])
    line = slope * data[x_col] + intercept
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.scatter(data[x_col], data[y_col], alpha=0.5)
    ax.plot(data[x_col], line, color="darkorange", linewidth=2)
    ax.set_xlabel("AQS PM2.5")
    ax.set_ylabel("PurpleAir PM2.5")
    return fig, ax


def figure_10() -> tuple[Figure, Axes]:
    df_10 = _df_10()
    fig, ax = _figure_10(df_10)
    return fig, ax


def _df_11() -> pd.DataFrame:
    """fit"""

    def calculate_best_fitting_line(
        df: pd.DataFrame, *, srs: list[str]
    ) -> tuple[np.float64, np.float64]:
        srs_1: pd.Series = df.loc[:, srs[0]]
        srs_2: pd.Series = df.loc[:, srs[1]]

        def theta_1(x: pd.Series, y: pd.Series) -> np.float64:
            r = x.corr(y)
            return r * y.std() / x.std()

        def theta_0(x: pd.Series, y: pd.Series) -> np.float64:
            return y.mean() - theta_1(x, y) * x.mean()

        t0 = theta_0(srs_1, srs_2)
        t1 = theta_1(srs_1, srs_2)
        return t0, t1

    def examine_errors(
        df: pd.DataFrame, t0: np.float64, t1: np.float64
    ) -> pd.DataFrame:
        prediction = t0 + t1 * df["pm25aqs"]
        error = df["pm25pa"] - prediction
        return pd.DataFrame(dict(prediction=prediction, error=error))

    df_10 = _df_10()
    t0, t1 = calculate_best_fitting_line(df_10, srs=["pm25aqs", "pm25pa"])
    df_11 = examine_errors(df_10, t0, t1)
    return df_11


def _figure_11(
    data: DataFrame,
    x_col: str = "prediction",
    y_col: str = "error",
    figsize: tuple[float, float] = (10.5, 7.5),
) -> tuple[Figure, Axes]:
    """Dispersión de error"""
    fig, ax = plt.subplots(
        figsize=figsize,
    )
    ax.scatter(data.loc[:, x_col], data.loc[:, y_col], alpha=0.5)
    ax.axhline(0.0, linestyle="dashed", alpha=0.7)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Error")
    return fig, ax


def figure_11() -> tuple[Figure, Axes]:
    df_11 = _df_11()
    fig, ax = _figure_11(df_11)
    return fig, ax


def _df_12() -> pd.DataFrame:
    """GA; dates, bad dates, scatter"""

    quid_cols = ["Date", "ID", "region", "PM25FM", "PM25cf1", "RH"]
    quo_cols = ["date", "id", "region", "pm25aqs", "pm25pa", "rh"]
    cols_hash = dict(zip(quid_cols, quo_cols))
    full: pd.DataFrame = (
        pd.read_csv(CLEANED24_PATH, usecols=np.array(quid_cols), parse_dates=["Date"])
        .dropna()
        .rename(columns=cols_hash)
    )
    bad_dates = pd.to_datetime(["2019-08-21", "2019-08-22", "2019-09-24"])
    GA: pd.DataFrame = full.loc[
        (full.loc[:, "id"] == "GA1") & (~full.loc[:, "date"].isin(bad_dates)), :
    ]

    def calculate_best_fitting_line(
        df: pd.DataFrame, *, srs: list[str]
    ) -> tuple[np.float64, np.float64]:
        srs_1: pd.Series = df.loc[:, srs[0]]
        srs_2: pd.Series = df.loc[:, srs[1]]

        def theta_1(x: pd.Series, y: pd.Series) -> np.float64:
            r = x.corr(y)
            return r * y.std() / x.std()

        def theta_0(x: pd.Series, y: pd.Series) -> np.float64:
            return y.mean() - theta_1(x, y) * x.mean()

        t0 = theta_0(srs_1, srs_2)
        t1 = theta_1(srs_1, srs_2)
        return t0, t1

    def examine_errors(
        df: pd.DataFrame, t0: np.float64, t1: np.float64
    ) -> pd.DataFrame:
        prediction = t0 + t1 * df["pm25aqs"]
        error = df["pm25pa"] - prediction
        return pd.DataFrame(dict(prediction=prediction, error=error))

    df_10 = _df_10()
    t0, t1 = calculate_best_fitting_line(df_10, srs=["pm25aqs", "pm25pa"])
    pred_errs = examine_errors(df_10, t0, t1)
    return pd.DataFrame({"date": GA["date"], "error": pred_errs["error"]})


def _figure_12(
    df: pd.DataFrame,
    figsize=(10.5, 7.5),
) -> tuple[Figure, Axes]:
    """Dispersión de fechas"""
    x_srs: Series = df.loc[:, "date"]
    y_srs: Series = df.loc[:, "error"]
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.scatter(x=x_srs, y=y_srs, alpha=0.5)
    ax.axhline(0.0, linestyle="dashed", alpha=0.7)
    ax.set_ylabel("Error")
    ax.set_xlabel("Fecha")
    return fig, ax


def figure_12() -> tuple[Figure, Axes]:
    df_12 = _df_12()
    fig, ax = _figure_12(df_12)
    return fig, ax


def _df_13() -> pd.DataFrame:
    """GA; dates, bad dates, scatter"""
    quid_cols = ["Date", "ID", "region", "PM25FM", "PM25cf1", "RH"]
    quo_cols = ["date", "id", "region", "pm25aqs", "pm25pa", "rh"]
    cols_hash = dict(zip(quid_cols, quo_cols))
    full: pd.DataFrame = (
        pd.read_csv(CLEANED24_PATH, usecols=np.array(quid_cols), parse_dates=["Date"])
        .dropna()
        .rename(columns=cols_hash)
    )
    bad_dates = pd.to_datetime(["2019-08-21", "2019-08-22", "2019-09-24"])
    df_13: pd.DataFrame = full.loc[
        (full.loc[:, "id"] == "GA1") & (~full.loc[:, "date"].isin(bad_dates)), :
    ]
    return df_13


# facet scatter plot
def _figure_13(
    data: DataFrame,
    rh_bins: list[float] = [43, 50, 55, 60, 78],
    rh_labels: list[str] = ["<50", "50-55", "55-60", ">60"],
    figsize: tuple[float, float] = (10.5, 8.5),
) -> tuple[Figure, Axes]:
    """Paneles de dispersión con humedad relativa como variable"""
    x_col = "pm25aqs"
    y_col = "pm25pa"
    rh_col = "rh"
    rh_categories = pd.cut(data[rh_col], bins=rh_bins, labels=rh_labels)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(hspace=0.3)
    axs_flat = axs.flatten()
    for ax, (category, group) in zip(axs_flat, data.groupby(rh_categories)):
        ax.scatter(group[x_col], group[y_col], alpha=0.5)
        ax.set_title(f"RH {category}")
        ax.set_xlabel("AQS PM2.5")
        ax.set_ylabel("PurpleAir PM2.5")
        ax.set_xlim(data[x_col].min(), data[x_col].max())
        ax.set_ylim(data[y_col].min(), data[y_col].max())
    return fig, axs


def figure_13() -> tuple[Figure, Axes]:
    df_13 = _df_13()
    fig, axs = _figure_13(df_13)
    return fig, axs


def _df_14() -> pd.DataFrame:
    df_13 = _df_13()
    df_14 = df_13.copy()
    return df_14


def _figure_14(
    data: DataFrame,
    columns: Sequence[str] = ["pm25pa", "pm25aqs", "rh"],
    labels: dict[str, str] = {
        "pm25aqs": "AQS",
        "pm25pa": "PurpleAir",
        "rh": "Humidity",
    },
    figsize: tuple[float, float] = (10.5, 8.5),
) -> tuple[Figure, Axes]:
    """Dispersión de matriz de correlación"""
    fig, axs = plt.subplots(len(columns), len(columns), figsize=figsize)
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i != j:
                axs[i, j].scatter(
                    data[col2],
                    data[col1],
                    alpha=0.5,
                    s=20,
                )
                if i == len(columns) - 1:
                    axs[i, j].set_xlabel(labels.get(col2, col2))
                if j == 0:
                    axs[i, j].set_ylabel(labels.get(col1, col1))
            else:
                axs[i, j].set_visible(False)
    return fig, axs


def figure_14() -> tuple[Figure, Axes]:
    df_14 = _df_14()
    fig, axs = _figure_14(df_14)
    axs[0, 0].set_ylabel("PurpleAir")
    axs[2, 0].set_ylabel("Humedad")
    axs[2, 2].set_ylabel("Humedad")
    return fig, axs


def _df_15() -> tuple[NDArray, NDArray]:
    df_13 = _df_13()
    y: NDArray = (df_13.loc[:, "pm25pa"]).to_numpy()
    X2: NDArray = (df_13.loc[:, ["pm25aqs", "rh"]]).to_numpy()
    model2: LinearRegression = LinearRegression().fit(X=X2, y=y)
    print(
        f"PA estimate = re{model2.intercept_:.1f} ppm +",
        f"{model2.coef_[0]:.2f} ppm/ppm x AQS + ",
        f"{model2.coef_[1]:.2f} ppm/percernt x RH",
    )

    # checking the quality of the fit
    predicted_2var: NDArray = model2.predict(X2)
    error_2var: NDArray = y - predicted_2var
    return predicted_2var, error_2var


def _figure_15(
    predicted_values: np.ndarray,
    error_values: np.ndarray,
    error_threshold: float = 4.0,
    y_range: tuple[float, float] = (-12, 12),
    figsize: tuple[float, float] = (10.5, 5.5),
) -> tuple[Figure, Axes]:
    """Dispersión de error vs valores predichos."""
    fig, ax = plt.subplots(figsize=figsize)
    x_min, x_max = np.min(predicted_values), np.max(predicted_values)
    ax.fill_between(
        [x_min - 0.5, x_max + 0.5],
        [-error_threshold, -error_threshold],
        [error_threshold, error_threshold],
        alpha=0.1,
        color="green",
        label=f"±{error_threshold} error range",
    )
    ax.scatter(predicted_values, error_values, alpha=0.5)
    ax.axhline(y=0, linestyle="--", linewidth=3, color="black", alpha=1.0)
    ax.set_xlabel("Predicted PurpleAir measurement")
    ax.set_ylabel("Error")
    ax.set_ylim(y_range)
    ax.legend()
    return fig, ax


def figure_15() -> tuple[Figure, Axes]:
    df_15 = _df_15()
    fig, ax = _figure_15(*df_15)
    return fig, ax


# user_functions = [
#     name
#     for name, obj in globals().items()
#     if callable(obj) and not isinstance(obj, type)
# ]
# user_figures = [
#     name for name in user_functions if "figure" in name and not name.startswith("_")
# ]
# user_figures = [name for name in user_functions if "figure" in name or "df" in name]
# for uf in user_figures:
#     uf_object = globals()[uf]
#     print(f"Generando {uf}")
#     fig, ax = uf_object()
#     plt.show()

# figs = {
#     "figure_01": figure_01,
#     "figure_02": figure_02,
#     "figure_03": figure_03,
#     "figure_04": figure_04,
#     "figure_05": figure_05,
#     "figure_06": figure_06,
#     "figure_07": figure_07,
#     # "figure_08": figure_08,
#     # "figure_09": figure_09,
#     "figure_10": figure_10,
#     "figure_11": figure_11,
#     "figure_12": figure_12,
#     "figure_13": figure_13,
#     "figure_14": figure_14,
#     "figure_15": figure_15,
# }

# figs_dirpath = Path(".").resolve() / "air_model_figures_2"
# figs_dirpath.mkdir(exist_ok=True)
# fl = list(figs.keys())

# for i, func_name in enumerate(fl):
#     try:
#         fig, _ = figs[func_name]()
#         fig.suptitle(
#             f"Modelo de calibración de sensores\nGráfica {func_name.split("_")[1]} de 15"
#         )
#         print(f"Generando gráfica {func_name}")

#         # plt.show()

#         save_path = figs_dirpath.name + "/" + func_name + ".png"
#         fig.savefig(save_path, dpi=300)
#         plt.close()
#         print(f"\nSe guardó la imagen {func_name}.\n", end="\u2a69" * (1 + i) + "\n")
#         print(f"Quedan {len(fl)- 1 - i}")
#     except NameError:
#         print(f"Todavía no está lista {func_name}")

# print("Se guardaron todas las imágenes.")
