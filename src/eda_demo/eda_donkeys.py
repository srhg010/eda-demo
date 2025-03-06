from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from scipy.optimize import minimize

plt.style.use("seaborn-v0_8-darkgrid")
mpl.rcParams["figure.facecolor"] = "e6e6e6"
mpl.rcParams["axes.facecolor"] = "e6e6e6"

# supported linestyles:
# '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

# paths
datasets_path = Path("/home/zerserob/Documents/Projects/Python/datasets")
donkeys_path = datasets_path / "donkeys.csv"

# examine the first few rows
first_few_lines = []
if not first_few_lines:
    with donkeys_path.open() as f:
        for _ in range(5):
            first_few_lines.append(f.readline())

donkeys = pd.read_csv(donkeys_path)
donkeys_cb_path = {
    "BCS": {
        "Data type": "float64",
        "Feature type": "Ordinal",
        "Description": "Body condition score: from 1 (emaciated) to 3 (healthy) to 5 (obese) in increments of 0.5.",
    },
    "Age": {
        "Data type": "string",
        "Feature type": "Ordinal",
        "Description": "Age in years, under 2, 2–5, 5–10, 10–15, 15–20, and over 20 years",
    },
    "Sex": {
        "Data type": "string",
        "Feature type": "Nominal",
        "Description": "Sex categories: stallion, gelding, female",
    },
    "Length": {
        "Data type": "int64",
        "Feature type": "Numeric",
        "Description": "Body length (cm) from front leg elbow to back of pelvis",
    },
    "Girth": {
        "Data type": "int64",
        "Feature type": "Numeric",
        "Description": "Body circumference (cm), measured just behind front legs",
    },
    "Height": {
        "Data type": "int64",
        "Feature type": "Numeric",
        "Description": "Body height (cm) up to point where neck connects to back",
    },
    "Weight": {
        "Data type": "int64",
        "Feature type": "Numeric",
        "Description": "Weight (kilogram)",
    },
    "WeightAlt": {
        "Data type": "float64",
        "Feature type": "Numeric",
        "Description": "Second weight measurement taken on a subset of donkeys",
    },
}
donkeys_cb_df = pd.DataFrame.from_dict(donkeys_cb_path, orient="index").rename_axis(
    "Feature"
)


# eliminamos estos dos valores atípicos
def _remove_bcs_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~df.loc[:, "BCS"].isin([1.0, 4.5]), :]


# relación entre peso y altura
def _remove_weight_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.loc[:, "Weight"] >= 40, :]


def anes_loss(x):
    # función de pérdida personalizada con ponderación
    # diferente para errores positivos y negativos
    w = (x >= 0) + 3 * (x < 0)
    return np.square(x) * w


def training_loss(X: pd.DataFrame, y: pd.DataFrame) -> Callable:
    # función de pérdida para el entrenamiento que utiliza anes_loss
    def loss(theta) -> np.float64:
        predicted = X @ theta
        return np.mean(anes_loss(100 * (y - predicted) / predicted))

    return loss


def _combine_bcs(X: pd.DataFrame) -> pd.DataFrame:
    # combina las categorías BCS 1.5 y 2.0
    new_bcs_2 = X.loc[:, "BCS_2.0"] + X.loc[:, "BCS_1.5"]
    return X.assign(**{"BCS_2.0": new_bcs_2}).drop(columns=["BCS_1.5"])


def _combine_age_and_sex(X: pd.DataFrame) -> pd.DataFrame:
    # elimina columnas específicas de edad y sexo
    return X.drop(
        columns=["Age_10-15", "Age_15-20", "Age_>20", "Sex_gelding", "Sex_stallion"]
    )


# quality checks on the data
# quality of the measurements and their distributions
def _df_01() -> tuple[pd.DataFrame, Any, Any]:
    donkeys = pd.read_csv(donkeys_path)
    donkeys = donkeys.assign(
        difference=donkeys.loc[:, "WeightAlt"] - donkeys.loc[:, "Weight"]
    )

    # values in the body condition score
    bcs_counts = (donkeys.loc[:, "BCS"].value_counts()).rename(
        donkeys_cb_df.loc["BCS", "Description"]
    )
    extr_donkeys = donkeys.loc[donkeys.loc[:, "BCS"].isin([1.0, 4.5]), :]
    return donkeys, bcs_counts, extr_donkeys


def _figure_01(df: pd.DataFrame) -> tuple[Figure, Axes]:
    x = df["difference"]
    figsize = (6 * 1.4, 6)
    bins = 15
    x_label = "Diferencias de peso (kg)"
    y_label = "Conteo"
    suptitle = """31 burros fueron pesados dos veces para tener
    una medida de la consistencia de las básculas."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.hist(x=x, bins=bins, label="bins=15", alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.suptitle(suptitle)
    return fig, ax


def figure_01() -> tuple[Figure, Axes]:
    df_01, *(_) = _df_01()
    fig, ax = _figure_01(df_01)
    return fig, ax


def _df_02() -> pd.DataFrame:
    df_02 = pd.read_csv(donkeys_path).pipe(_remove_bcs_outliers)
    return df_02


def _figure_02(
    df: pd.DataFrame,
    col: str = "Weight",
    x_label: str = "Peso (kg)",
    y_label: str = "Conteo",
    suptitle: str = "Distribución del peso de los burros",
    bins: int = 40,
    figsize: tuple[int, int] = (11, 6),
) -> tuple[Figure, Axes]:
    # crea un histograma para la columna especificada
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.hist(df.loc[:, col], bins=bins, label="bins=40")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.suptitle(suptitle)
    return fig, ax


def figure_02():
    df_02 = _df_02()
    fig, ax = _figure_02(df_02)
    return fig, ax


def _df_03() -> pd.DataFrame:
    df_02 = _df_02()
    df_03 = df_02.copy()
    return df_03


def _figure_03(
    df: pd.DataFrame,
    x_val: str = "Height",
    y_val: str = "Weight",
    figsize: tuple[int, int] = (11, 6),
) -> tuple[Figure, Axes]:
    # crea un gráfico de dispersión con las columnas especificadas
    x = df.loc[:, x_val]
    y = df.loc[:, y_val]
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.scatter(x=x, y=y, alpha=0.5)
    ax.set_ylabel("Peso (kg)")
    ax.set_xlabel("Altura (cm)")
    fig.suptitle("Gráfico de dispersión: altura vs peso")
    return fig, ax


def figure_03():
    df_03 = _df_03()
    fig, ax = _figure_03(df_03)
    return fig, ax


# splitting the data, shuffle the indices
def _df_04() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train set"""
    donkeys = (
        pd.read_csv(donkeys_path)
        .pipe(_remove_bcs_outliers)
        .pipe(_remove_weight_outliers)
    )
    np.random.seed(42)
    len_donkeys = len(donkeys)
    indices_donkeys = np.arange(len_donkeys)
    np.random.shuffle(indices_donkeys)
    indices_train = np.round((0.8 * len_donkeys)).astype("int")

    train_set = donkeys.iloc[indices_donkeys[:indices_train]]
    test_set = donkeys.iloc[indices_donkeys[indices_train:]]
    return train_set, test_set


def _figure_04(
    train_set: Any,
    age_order: list[str] = ["<2", "2-5", "5-10", "10-15", "15-20", ">20"],
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [0.7, 0.3]},
        layout="constrained",
    )
    age_data = {age: [] for age in age_order}
    for age, weight in zip(train_set["Age"], train_set["Weight"]):
        if age in age_data:
            age_data[age].append(weight)
    ax1.boxplot(
        [age_data[age] for age in age_order], patch_artist=True, tick_labels=age_order
    )
    ax1.set_xlabel("Edad (años)")
    ax1.set_ylabel("Peso (kg)")
    sex_categories = train_set["Sex"].unique()
    sex_data = {sex: [] for sex in sex_categories}
    for sex, weight in zip(train_set["Sex"], train_set["Weight"]):
        sex_data[sex].append(weight)
    ax2.boxplot(
        [sex_data[sex] for sex in sex_categories],
        patch_artist=True,
        tick_labels=sex_categories,
    )
    ax2.set_xlabel("Sexo")
    return fig, (ax1, ax2)


def figure_04() -> tuple[Figure, tuple[Axes, Axes]]:
    df_04, _ = _df_04()
    fig, (ax1, ax2) = _figure_04(df_04)
    return fig, (ax1, ax2)


def _df_05() -> pd.DataFrame:
    df_05, _ = _df_04()
    return df_05


def _figure_05(train_set: Any) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(12, 10))
    bcs_categories = sorted(train_set["BCS"].unique())
    bcs_data = {bcs: [] for bcs in bcs_categories}
    for bcs, weight in zip(train_set["BCS"], train_set["Weight"]):
        bcs_data[bcs].append(weight)
    box_plot = ax.boxplot(
        [bcs_data[bcs] for bcs in bcs_categories],
        tick_labels=bcs_categories,
        patch_artist=True,
        showfliers=True,
    )
    for i, bcs in enumerate(bcs_categories):
        x_pos = i + 1
        y_values = bcs_data[bcs]
        ax.scatter(
            x=[x_pos + (0.1 * (j % 3 - 1)) for j in range(len(y_values))],
            y=y_values,
            alpha=0.5,
            edgecolor="black",
            s=30,
        )
    ax.set_xlabel("Body condition score")
    ax.set_ylabel("Weight (kg)")
    return fig, ax


def figure_05() -> tuple[Figure, Axes]:
    df_05 = _df_05()
    fig, ax = _figure_05(df_05)
    return fig, ax


# Exploring
# features: shapes and relationships -> transformations and models
# categorical (age, sex, body, body condition) ~ weight


# examine the quantitative variables
def _df_06() -> pd.DataFrame:
    df_05 = _df_05()
    df_06 = df_05.loc[:, ["Weight", "Length", "Girth", "Height"]]
    df_06.columns = ["Peso (kg)", "Largo (cm)", "Grosor (cm)", "Altura (cm)"]
    return df_06


def _figure_06(train_numeric: Any) -> tuple[Figure, list[Axes]]:
    n_vars = len(train_numeric.columns)
    column_names = ["Peso (kg)", "Largo (cm)", "Grosor (cm)", "Altura (cm)"]
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = GridSpec(n_vars, n_vars, figure=fig)
    axes = []

    # Create a color cycle
    # colors = plt.cm.tab10.colors  # This gives 10 distinct colors
    colors = mpl.color_sequences["tab10"]
    color_idx = 0

    for i in range(n_vars):
        for j in range(n_vars):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
            if i < n_vars - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            if i == n_vars - 1:
                ax.set_xlabel(column_names[j])
            if j == 0:
                ax.set_ylabel(column_names[i])
            if i == j:
                sns.histplot(
                    train_numeric.iloc[:, i],
                    ax=ax,
                    kde=True,
                    color=colors[i % len(colors)],  # Use color cycle for histograms too
                )
                ax.set_ylabel("")
            else:
                ax.scatter(
                    train_numeric.iloc[:, j],
                    train_numeric.iloc[:, i],
                    alpha=0.6,
                    s=15,
                    edgecolor="none",
                    color=colors[color_idx % len(colors)],  # Use color from cycle
                )
                color_idx += 1  # Move to next color
    axes[0].set_ylabel("Peso (kg)")

    return fig, axes


def figure_06() -> tuple[Figure, list[Axes]]:
    df_06 = _df_06()
    fig, axs = _figure_06(df_06)
    return fig, axs


# train_numeric.corr()
#           Weight    Length     Girth    Height
# Weight  1.000000  0.777575  0.899036  0.711616
# Length  0.777575  1.000000  0.663904  0.580503
# Girth   0.899036  0.663904  1.000000  0.699031
# Height  0.711616  0.580503  0.699031  1.000000


# Modeling a Donkey's Weight
def _df_07() -> tuple[NDArray, NDArray]:
    """Función de error."""
    xs = np.linspace(-40, 40, 500)
    loss = anes_loss(xs)
    return xs, loss


def _figure_07(xs, loss) -> tuple[Figure, Axes]:
    x_label = "{:^}{:<30}{:>30}".format(
        "<" + "-" * 13 + " " + "Error relativo (%)" + " " + "-" * 13 + ">",
        "\nSobreestimación",
        "Subestimación",
    )
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.plot(xs, loss)
    ax.set_title("Error cuadrático ad hoc")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Error")
    ax.set_xlim(-40, 40)
    ax.set_ylim(-30, anes_loss(-45))
    return fig, ax


def figure_07() -> tuple[Figure, Axes]:
    df_07 = _df_07()
    fig, ax = _figure_07(*df_07)
    return fig, ax


def _df_08() -> pd.DataFrame:
    """resid"""
    train_set, _ = _df_04()
    # fitting a simple linear model
    # model of the form: \theta_{0} + \theta_{1}Girth
    # find the best \theta_{0} and \theta_{1}
    # create a design matrix
    X = train_set.assign(intr=1).loc[:, ["intr", "Girth"]]
    y = train_set.loc[:, "Weight"]
    results = minimize(training_loss(X, y), np.ones(2))
    theta_hat = results["x"]
    print(
        f"After fitting",
        f"theta_{0} = {theta_hat[0]:>7.2f}",
        f"theta_{1} = {theta_hat[1]:>7.2f}",
        sep="\n",
    )
    predicted = X @ theta_hat
    resids = 100 * (y - predicted) / predicted
    # scatter plot of the relative errors
    resid = pd.DataFrame(
        {
            "Predicted weight (kg)": predicted,
            "Percent error": resids,
        }
    )
    return resid


def _figure_08(df: pd.DataFrame) -> tuple[Figure, Axes]:
    """crea un gráfico de dispersión de errores relativos"""
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], alpha=0.5)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("Peso predicho (kg)")
    ax.set_ylabel("Error en porcentaje (%)")
    return fig, ax


def figure_08() -> tuple[Figure, Axes]:
    df_08 = _df_08()
    fig, ax = _figure_08(df_08)
    return fig, ax


# Fitting a Multiple Linear Model
def _df_09():
    train_set, _ = _df_04()
    X = train_set.assign(intr=1).loc[:, ["intr", "Girth"]]
    y = train_set.loc[:, "Weight"]

    def training_error(model):
        # calcula el error de entrenamiento para un modelo dado
        X = train_set.assign(intr=1).loc[
            :, ["intr", *model]
        ]  # resolver esta variable no definida
        theta_hat = minimize(training_loss(X, y), np.ones(X.shape[1]))["x"]
        predicted = X @ theta_hat
        return np.mean(anes_loss(100 * (y - predicted) / predicted))

    numeric_var = ["Girth", "Length", "Height"]
    models = [
        list(subset) for n in [1, 2, 3] for subset in combinations(numeric_var, n)
    ]
    model_risks = [training_error(model) for model in models]
    models_errors = pd.DataFrame({"model": models, "mean_training_error": model_risks})

    # selecting the ["Girth", "Length"] model
    # we'll use feature engineering:
    #  incorporate categorical variables into the model
    #  categories -> {0,1} -- hot-encoding
    X_one_hot = (
        train_set.assign(intr=1)
        .loc[
            :,
            ["intr", "Length", "Girth", "BCS", "Age", "Sex"],
        ]
        .pipe(pd.get_dummies, columns=["BCS", "Age", "Sex"])
        .drop(columns=["BCS_3.0", "Age_5-10", "Sex_female"])
        .astype("int")
    )

    results = minimize(training_loss(X_one_hot, y), np.ones(X_one_hot.shape[1]))
    theta_hat = results["x"]
    y_pred = X_one_hot @ theta_hat
    training_error = np.mean(anes_loss(100 * (y - y_pred) / y_pred))
    print(f"Training error: {training_error:.2f}")

    # making the model simpler while keeping its accuracy
    # coefficients of the dummy variables
    # how close the are to 0 and to one another
    vars_dict = {
        "bcs_vars": ["BCS_1.5", "BCS_2.0", "BCS_2.5", "BCS_3.5", "BCS_4.0"],
        "age_vars": ["Age_<2", "Age_2-5", "Age_10-15", "Age_15-20", "Age_>20"],
        "sex_vars": ["Sex_gelding", "Sex_stallion"],
    }
    thetas = pd.DataFrame(
        {
            "var": X_one_hot.columns,
            "theta_hat": theta_hat,
        }
    ).set_index("var")
    return thetas, X_one_hot


def _figure_09(thetas: pd.DataFrame) -> tuple[Figure, list[Axes]]:
    # visualiza los coeficientes del modelo agrupados por categorías
    vars_dict = {
        "bcs_vars": ["BCS_1.5", "BCS_2.0", "BCS_2.5", "BCS_3.5", "BCS_4.0"],
        "age_vars": ["Age_<2", "Age_2-5", "Age_10-15", "Age_15-20", "Age_>20"],
        "sex_vars": ["Sex_gelding", "Sex_stallion"],
    }
    axes_titles = ["Body Score", "Edad", "Sexo"]
    xtick_labels = [
        ["1.5", "2.0", "2.5", "3.5", "4.0"],
        ["<2", "2-5", "10-15", "15-20", ">20"],
        ["castrado", "semental"],
    ]
    thetas = thetas.reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
    for i, cat_var in enumerate(vars_dict.keys()):
        mask = thetas["var"].isin(vars_dict[cat_var])
        axes[i].scatter(
            x=thetas.loc[mask, "var"],
            y=(y := thetas.loc[mask, "theta_hat"]),
            alpha=0.5,
        )
        # axes[i].set_yticks(y.to_list())
        axes[i].set_title(axes_titles[i])
        axes[i].axhline(0, c="k", alpha=0.5)
        axes[i].set_xticks(range(len(xtick_labels[i])))
        axes[i].set_xticklabels(xtick_labels[i])
        # axes[i].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Coeficientes parametrales prospectivos")
    return fig, axes


def figure_09():
    df_09, _ = _df_09()
    fig, axes = _figure_09(df_09)
    return fig, axes


def _df_10():
    train_set, test_set = _df_04()
    _, X_one_hot = _df_09()
    X = train_set.assign(intr=1).loc[:, ["intr", "Girth"]]
    y = train_set.loc[:, "Weight"]
    X_one_hot_simple = X_one_hot.pipe(_combine_bcs).pipe(_combine_age_and_sex)

    results = minimize(
        training_loss(X_one_hot_simple, y), np.ones(X_one_hot_simple.shape[1])
    )
    theta_hat = results.x
    y_pred = X_one_hot_simple @ theta_hat
    training_error = np.mean(anes_loss(100 * (y - y_pred) / y_pred))
    print(f"Training error: {training_error:.2f}")

    thetas = pd.DataFrame({"var": X_one_hot_simple.columns, "theta_hat": theta_hat})
    print(thetas)

    # Model Assessment:
    #  1. Preparing the test set
    y_test = test_set.loc[:, "Weight"]
    X_test = (
        test_set.assign(intr=1)
        .loc[:, ["intr", "Length", "Girth", "BCS", "Age", "Sex"]]
        .pipe(pd.get_dummies, columns=["BCS", "Age", "Sex"])
        .drop(columns=["BCS_3.0", "Age_5-10", "Sex_female"])
        .pipe(_combine_bcs)
        .pipe(_combine_age_and_sex)
    )
    y_pred_test: pd.Series = X_test @ theta_hat
    test_set_error: pd.Series = 100 * (y_test - y_pred_test) / y_pred_test
    return y_pred_test, test_set_error


def _figure_10(x: pd.Series, y: pd.Series):
    # visualiza los errores relativos en el conjunto de prueba
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x=x, y=y, s=4, alpha=0.5)
    ax.axhline(10.0, linewidth=3, c="green", alpha=0.5)
    ax.axhline(-10.0, linewidth=3, c="green", alpha=0.5)
    ax.axhline(0.0, linewidth=3, c="green", alpha=0.3)
    ax.set_xlabel("Error predicho (kg)")
    ax.set_ylabel("Error relativo (%)")
    return fig, ax


def figure_10():
    df_10 = _df_10()
    fig, ax = _figure_10(*df_10)
    return fig, ax


def _df_11():
    train_set, test_set = _df_04()
    _, X_one_hot = _df_09()
    X = train_set.assign(intr=1).loc[:, ["intr", "Girth"]]
    y = train_set.loc[:, "Weight"]
    X_one_hot_simple = X_one_hot.pipe(_combine_bcs).pipe(_combine_age_and_sex)
    results = minimize(
        training_loss(X_one_hot_simple, y), np.ones(X_one_hot_simple.shape[1])
    )
    theta_hat = results.x
    y_pred = X_one_hot_simple @ theta_hat
    training_error = np.mean(anes_loss(100 * (y - y_pred) / y_pred))
    print(f"Training error: {training_error:.2f}")
    # displaying the coefficients
    # summarizing the model
    thetas = pd.DataFrame({"var": X_one_hot_simple.columns, "theta_hat": theta_hat})
    print(thetas)
    # Model Assessment:
    #  1. Preparing the test set
    y_test = test_set.loc[:, "Weight"]
    X_test = (
        test_set.assign(intr=1)
        .loc[:, ["intr", "Length", "Girth", "BCS", "Age", "Sex"]]
        .pipe(pd.get_dummies, columns=["BCS", "Age", "Sex"])
        .drop(columns=["BCS_3.0", "Age_5-10", "Sex_female"])
        .pipe(_combine_bcs)
        .pipe(_combine_age_and_sex)
    )
    y_pred_test: pd.Series = X_test @ theta_hat
    return y_pred_test, y_test


def _figure_11(x: pd.Series, y: pd.Series):
    # visualiza una visión general completa de las predicciones en el conjunto de prueba
    fig, ax = plt.subplots(figsize=(11.2, 8))
    ax.scatter(x=x, y=y, s=4, alpha=0.5)
    ax.plot([60, 200], [60, 200], label="0% error", alpha=0.3)
    ax.plot([60, 200], [66, 220], label="10% error", alpha=0.4, c="g")
    ax.plot([60, 200], [54, 180], label="-10% error", alpha=0.4, c="r")
    ax.set_xlabel("Peso predicho (kg)")
    ax.set_ylabel("Peso real (kg)")
    ax.legend()
    fig.suptitle("Predicción de peso en el conjunto de prueba.")
    return fig, ax


def figure_11():
    df_11 = _df_11()
    fig, ax = _figure_11(*df_11)
    return fig, ax


figs = {
    # "figure_01": figure_01,
    # "figure_02": figure_02,
    # "figure_03": figure_03,
    # "figure_04": figure_04,
    "figure_05": figure_05,
    # "figure_06": figure_06,
    # "figure_08": figure_08,
    # "figure_09": figure_09,
    # "figure_10": figure_10,
    # "figure_11": figure_11,
}

figs_dirpath = Path(".").resolve() / "eda_donkeys_figures"
figs_dirpath.mkdir(exist_ok=True)
fl = list(figs.keys())

for i, func_name in enumerate(fl):
    fig, _ = figs[func_name]()
    save_path = figs_dirpath.name + "/" + func_name + ".png"
    fig.savefig(save_path, dpi=300)
    print(f"\nSe guardó la imagen {func_name}.\n", end="\u2a69" * (1 + i) + "\n")
    print(f"Quedan {len(fl)- 1 - i}")

print("Se guardaron todas las imágenes.")
