import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlalchemy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.style.use("seaborn-v0_8-darkgrid")
mpl.rcParams["figure.facecolor"] = "e6e6e6"
mpl.rcParams["axes.facecolor"] = "e6e6e6"

# supported linestyles:
# '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

# paths
datasets_path = Path("/home/zerserob/Documents/Projects/Python/datasets")
purpleair_path = datasets_path / "purpleair_study"
pacleaned_path = purpleair_path / "cleaned_purpleair_aqs"
cleaned24_path = pacleaned_path / "Full24hrdataset.csv"
aqssites_path = purpleair_path / "list_of_aqs_sites.csv"
pasensors_path = purpleair_path / "list_of_purpleair_sensors.json"
sacramsensor_path = purpleair_path / "aqs_06-067-0010.csv"
amtstestingadir_path = purpleair_path / "purpleair_AMTS"
pa_csvs = sorted(amtstestingadir_path.glob("*.csv"))


# paths -> dataframes
aqs_sites_full = pd.read_csv(aqssites_path)


# buscar valores duplicados
id_counts = aqs_sites_full["AQS_Site_ID"].value_counts()
dup_site = aqs_sites_full.query("AQS_Site_ID == '19-163-0015'")
some_cols = [
    "POC",
    "Monitor_Start_Date",
    "Last_Sample_Date",
    "Sample_Collection_Method",
]

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
    pd.read_csv(cleaned24_path, usecols=usecols, parse_dates=["Date"])
    .rename(columns=renamer_map)
    .dropna()
)


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


def _compute_daily_avgs(df: pd.DataFrame) -> pd.DataFrame:
    # esto ya no se usó
    should_keep = (
        df.resample("D")["PM25cf1"]
        .size()
        .to_frame()
        .apply(_has_enough_readings, axis="columns")
    )
    return df.resample("D").mean().loc[should_keep]


aqs_sites = aqs_sites_full.pipe(_rollup_dup_sites).pipe(_cols_aqs)


# ahora trabajamos con PurpleAir
with open(pasensors_path) as f:
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
aqs_full = pd.read_csv(sacramsensor_path)
aqs_date_counts = aqs_full["date_local"].value_counts()
aqs = aqs_full.pipe(_rollup_dates).pipe(_drop_cols).pipe(_parse_dates)
date_range: pd.Timedelta = aqs["date_local"].max() - aqs["date_local"].min()

# Particulate Matter (PM) - micrograms/m^3 - rango: (0,2.5)


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

# # PA = b + mAQS + error
# # True air quality = -(b/m) + (1/m)PA + error

# # una variable
# AQS, PA = full_df[["pm25aqs"]], full_df[["pm25pa"]]
# model = LinearRegression().fit(AQS, PA)
# m, b = model.coef_[0], model.intercept_
# print(f"True air quality estimate = {-b/m} + {1/m}PA")

# # dos variables
# AQS_RH, PA = full_df[["pm25aqs", "rh"]], full_df["pm25pa"]
# model_h = LinearRegression().fit(AQS_RH, PA)
# coefs: np.ndarray
# m1: float
# m2: float
# b: float
# coefs = model_h.coef_
# [m1, m2], b = coefs, model_h.intercept_


def _pa_full() -> pd.DataFrame:
    return pd.read_csv(pa_csvs[0])


def _df_01() -> pd.DataFrame:
    """aqs con tres pipes:
    rollup_dates, drop_cols, parse_dates"""
    df_01 = aqs_full.pipe(_rollup_dates).pipe(_drop_cols).pipe(_parse_dates)
    return df_01


def _figure_01(
    data: Any,
    date_col="date_local",
    pm25_col="pm25",
    figsize=(10, 6),
) -> tuple[Figure, Axes]:
    """Gráfica de dispersión PM 2.5"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(data[date_col], data[pm25_col])
    ax.set_xlabel("Date")
    ax.set_ylabel("AQS daily avg PM2.5")
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
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data[y_col])
    ax.set_xlabel("Date")
    ax.set_ylabel("Records per day")
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
    ax.set_xlabel("Date")
    ax.set_ylabel("Records per day")
    return fig, ax


def figure_03() -> tuple[Figure, Axes]:
    df_03 = _df_03()
    fig, ax = _figure_03(df_03)
    return fig, ax


def _df_04() -> pd.DataFrame:
    """ts_nc4"""
    nc4 = full_df.loc[full_df["id"] == "NC4"]
    df_04 = (
        nc4.set_index("date").resample("W")["pm25aqs", "pm25pa"].mean().reset_index()
    )
    return df_04


def _df_05() -> pd.DataFrame:
    """nc4"""
    df_05 = full_df.loc[full_df["id"] == "NC4"]
    return df_05


def _figure_05(
    data: pd.DataFrame,
    aqs_col: str = "pm25aqs",
    pa_col: str = "pm25pa",
    figsize: tuple[int, int] = (12, 6),
    line_coords: tuple[tuple[float, float], tuple[float, float]] = ((2, 1), (13, 25)),
) -> tuple[Figure, Axes]:
    """Comparación de cuantiles."""
    percs: np.ndarray = np.arange(1, 100, 1)
    aqs_qs: np.ndarray = np.percentile(data[aqs_col], percs, method="lower")
    pa_qs: np.ndarray = np.percentile(data[pa_col], percs, method="lower")
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(aqs_qs, pa_qs)
    (x1, y1), (x2, y2) = line_coords
    ax.plot([x1, x2], [y1, y2], linestyle="--", linewidth=4)
    ax.set_xlabel("AQS quantiles")
    ax.set_ylabel("PurpleAir quantiles")
    return fig, ax


def figure_05() -> tuple[Figure, Axes]:
    df_05 = _df_05()
    fig, ax = _figure_05(df_05)
    return fig, ax


def _df_06() -> pd.DataFrame:
    nc4 = full_df.loc[full_df["id"] == "NC4"]
    df_06 = nc4["pm25pa"] - nc4["pm25aqs"]
    return df_06


def _figure_06(df: pd.DataFrame) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    bin_sequence: list[float] = list(np.linspace(-10, 30, num=20))
    ax.hist(df, bins=bin_sequence, density=True)
    ax.set_title("Distribution of difference between the two readings")
    ax.set_ylabel("This should be percent")
    ax.set_xlabel(r"Difference: PA-AQS reading")
    return fig, ax


def _df_07() -> pd.DataFrame:
    df_07 = full_df.loc[(full_df["pm25aqs"] < 50)]
    return df_07


def _figure_04(
    data: DataFrame,
    pm25_threshold: float = 50.0,
    x_col: str = "pm25aqs",
    y_col: str = "pm25pa",
    figsize: tuple[float, float] = (3.5, 2.5),
) -> tuple[Figure, Axes]:
    """Dispersión con regresión local."""
    filtered_data = data.loc[data[x_col] < pm25_threshold].copy()
    lowess_trend = lowess(
        filtered_data[y_col], filtered_data[x_col], return_sorted=True
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(filtered_data[x_col], filtered_data[y_col], alpha=0.5)
    ax.plot(lowess_trend[:, 0], lowess_trend[:, 1], color="orange", linewidth=2)
    ax.set_xlabel("AQS PM2.5")
    ax.set_ylabel("PurpleAir PM2.5")
    return fig, ax


def _df_08() -> pd.DataFrame:
    quid_cols = ["Date", "ID", "region", "PM25FM", "PM25cf1"]
    quo_cols = ["date", "id", "region", "pm25aqs", "pm25pa"]
    cols_hash = dict(zip(quid_cols, quo_cols))
    full: pd.DataFrame = pd.read_csv(
        cleaned24_path, usecols=np.array(quid_cols)
    ).rename(columns=cols_hash)
    df_08: pd.DataFrame = full.loc[(full["id"] == "GA1"), :]
    return df_08


def _figure_08(
    data: DataFrame,
    x_col: str = "pm25aqs",
    y_col: str = "pm25pa",
    figsize: tuple[float, float] = (12.5, 8.5),
) -> tuple[Figure, Axes]:
    """Dispersión con regresión lineal trivial."""
    slope, intercept, *(_) = stats.linregress(data[x_col], data[y_col])
    line = slope * data[x_col] + intercept
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.scatter(data[x_col], data[y_col], alpha=0.5)
    ax.plot(data[x_col], line, color="darkorange", linewidth=2)
    ax.set_xlabel("AQS PM2.5")
    ax.set_ylabel("PurpleAir PM2.5")
    return fig, ax


def figure_08() -> tuple[Figure, Axes]:
    df_08 = _df_08()
    fig, ax = _figure_08(df_08)
    return fig, ax


def _df_09() -> pd.DataFrame:
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

    df_08 = _df_08()
    t0, t1 = calculate_best_fitting_line(df_08, srs=["pm25aqs", "pm25pa"])
    df_09 = examine_errors(df_08, t0, t1)
    return df_09


def _figure_09(
    data: DataFrame,
    x_col: str = "prediction",
    y_col: str = "error",
    figsize: tuple[float, float] = (10.5, 7.5),
) -> tuple[Figure, Axes]:
    """Dispersión de error."""
    fig, ax = plt.subplots(
        figsize=figsize,
    )
    ax.scatter(data.loc[:, x_col], data.loc[:, y_col], alpha=0.5)
    ax.axhline(0.0, linestyle="dashed", alpha=0.7)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Error")
    return fig, ax


def figure_09() -> tuple[Figure, Axes]:
    df_09 = _df_09()
    fig, ax = _figure_09(df_09)
    return fig, ax


def _df_10() -> pd.DataFrame:
    """GA; dates, bad dates, scatter"""
    quid_cols = ["Date", "ID", "region", "PM25FM", "PM25cf1", "RH"]
    quo_cols = ["date", "id", "region", "pm25aqs", "pm25pa", "rh"]
    cols_hash = dict(zip(quid_cols, quo_cols))
    full: pd.DataFrame = (
        pd.read_csv(cleaned24_path, usecols=np.array(quid_cols), parse_dates=["Date"])
        .dropna()
        .rename(columns=cols_hash)
    )
    bad_dates = pd.to_datetime(["2019-08-21", "2019-08-22", "2019-09-24"])
    df_10: pd.DataFrame = full.loc[
        (full.loc[:, "id"] == "GA1") & (~full.loc[:, "date"].isin(bad_dates)), :
    ]
    return df_10


def _figure_10(
    x_srs: Series,
    y_srs: Series,
    figsize: tuple[float, float] = (10.5, 7.5),
) -> tuple[Figure, Axes]:
    """Dispersión de fechas."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.scatter(x=x_srs, y=y_srs, alpha=0.5)
    ax.axhline(0.0, linestyle="dashed", alpha=0.7)
    ax.set_ylabel("Error")
    ax.set_xlabel("Date")
    return fig, ax


def _df_11() -> pd.DataFrame:
    df_10 = _df_10()
    df_11 = df_10.copy()
    return df_11


# facet scatter plot
def _figure_11(
    data: DataFrame,
    x_col: str = "pm25aqs",
    y_col: str = "pm25pa",
    rh_col: str = "rh",
    rh_bins: list[float] = [43, 50, 55, 60, 78],
    rh_labels: list[str] = ["<50", "50-55", "55-60", ">60"],
    figsize: tuple[float, float] = (5.5, 3.5),
) -> tuple[Figure, Axes]:
    """Paneles de dispersión con humedad relativa como variable."""
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


def figure_11() -> tuple[Figure, Axes]:
    df_11 = _df_11()
    fig, axs = _figure_11(df_11)
    return fig, axs


def _df_12() -> pd.DataFrame:
    df_10 = _df_10()
    df_12 = df_10.copy()
    return df_12


def _figure_12(
    data: DataFrame,
    columns: Sequence[str] = ["pm25pa", "pm25aqs", "rh"],
    labels: dict[str, str] = {
        "pm25aqs": "AQS",
        "pm25pa": "PurpleAir",
        "rh": "Humidity",
    },
    figsize: tuple[float, float] = (5.5, 4.0),
) -> tuple[Figure, Axes]:
    """Dispersión de matriz de correlación."""
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


def figure_12() -> tuple[Figure, Axes]:
    df_12 = _df_12()
    fig, axs = _figure_12(df_12)
    return fig, axs


def _df_13() -> tuple[NDArray, NDArray]:
    df_10 = _df_10()
    y: NDArray = (df_10.loc[:, "pm25pa"]).to_numpy()
    X2: NDArray = (df_10.loc[:, ["pm25aqs", "rh"]]).to_numpy()
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


def _figure_13(
    error_values: np.ndarray,
    predicted_values: np.ndarray,
    error_threshold: float = 4.0,
    y_range: tuple[float, float] = (-12, 12),
    figsize: tuple[float, float] = (10.5, 5.5),
) -> tuple[Figure, Axes]:
    """Dispersión de error vs valores predichos."""
    fig, ax = plt.subplots(figsize=figsize)
    x_min, x_max = np.min(predicted_values), np.max(predicted_values)
    ax.fill_between(
        [x_min, x_max],
        [-error_threshold, -error_threshold],
        [error_threshold, error_threshold],
        alpha=0.2,
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


def figure_13() -> tuple[Figure, Axes]:
    df_13 = _df_13()
    fig, ax = _figure_13(*df_13)
    return fig, ax
