import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.typing import CoordsType
import nltk
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from nltk.stem.porter import PorterStemmer
from numpy.typing import NDArray
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

# algunas especificaciones para el estilo de las gráficas
plt.style.use("seaborn-v0_8-darkgrid")
mpl.rcParams["figure.facecolor"] = "e6e6e6"
mpl.rcParams["axes.facecolor"] = "e6e6e6"

# un diccionario para mostrar mensajes
messages: dict[int, str] = {}

# paths
datasets_path = Path("/home/zerserob/Documents/Projects/Python/datasets/")
# debería añadir los de 2023 y 2024 (the end of the woke era)
# text_path = datasets_path / "stateoftheunion1790-2022.txt"
text_alt_path = datasets_path / "stateoftheunion1790-2025.txt"
text_path = text_alt_path

# leer el documento
text: str = text_path.read_text()

# crear un regex para contar el número de discursos
num_speeches: int = len(re.findall(pattern=r"\*\*\*", string=text))
messages[0] = f"There are {num_speeches} speeches total."

# dividimos por discurso
records: list[str] = text.split("***")


# creamos un DataFrame con los discursos
def extract_parts(speech: str) -> list[str]:
    speech_list = speech.strip().split("\n")[1:]
    [name, date, *lines] = speech_list
    body = "\n".join(lines).strip()
    return [name, date, body]


def read_speeches(records: list[str]) -> pd.DataFrame:
    predf_list = []
    columns = pd.Index(["name", "date", "text"])
    for l in records[1:]:
        predf_list.append(extract_parts(l))
    return pd.DataFrame(data=predf_list, columns=columns, dtype="str")


# normalización del texto
def _clean_text(df: pd.DataFrame) -> pd.DataFrame:
    bracket_re: re.Pattern = re.compile(r"\[[^\]]+\]")
    not_a_word_re: re.Pattern = re.compile(r"[^a-z\s]")
    cleaned: pd.Series = (
        df.loc[:, "text"]
        .str.lower()
        .str.replace(bracket_re, "", regex=True)
        .str.replace(not_a_word_re, " ", regex=True)
    ).astype("str")
    return df.assign(text=cleaned)


df = read_speeches(records).pipe(_clean_text)

# stop words
# conjugación de verbos
stop_words = set(nltk.corpus.stopwords.words("english"))
porter_stemmer = PorterStemmer()


def stemming_tokenizer(document: str) -> list[str]:
    return [
        porter_stemmer.stem(word)
        for word in nltk.word_tokenize(document)
        if word not in stop_words
    ]


tfidf = TfidfVectorizer(tokenizer=stemming_tokenizer)
# type(speech_vectors)
# <class 'scipy.sparse._csr.csr_matrix'>
# speech_vectors.shape
# (232, 13211)
speech_vectors = tfidf.fit_transform(df.loc[:, "text"])
messages[1] = "Son 232 discursos. Se transformaron en vectores de 13,211 palabras."


def compute_pcs(data: NDArray, *, k: int | None = None) -> NDArray:
    if k == None:
        print("k no puede estar sin asignar")
    centered = data - data.mean(axis=0)
    U: NDArray
    s: NDArray
    U, s, _ = svds(centered, k=k)  # type: ignore
    return U @ np.diag(s)


pcs = compute_pcs(speech_vectors, k=2)  # type: ignore
# esto "ajusta" las componentes para que la figura
# se vea "bien"
if pcs[0, 0] < 0:
    pcs[:,] *= -1
if pcs[0, 1] < 0:
    pcs[:, 1] *= -1

df_pcas = df.assign(
    year=df.loc[:, "date"].str[-4:].astype(int), pc1=pcs[:, 0], pc2=pcs[:, 1]
)


# df_pcas["distance"] = np.sqrt(np.square(df_pcas["pc1"]) + np.square(df_pcas["pc2"]))
# min_distance = df_pcas[df_pcas["distance"] == min(df_pcas["distance"])]
# max_distance = df_pcas[df_pcas["distance"] == max(df_pcas["distance"])]
# min_pc2 = df_pcas[df_pcas["pc2"] == min(df_pcas["pc2"])]
# max_pc2 = df_pcas[df_pcas["pc2"] == max(df_pcas["pc2"])]
# min_pc1 = df_pcas[df_pcas["pc1"] == min(df_pcas["pc1"])]
# max_pc1 = df_pcas[df_pcas["pc1"] == max(df_pcas["pc1"])]
# most_recent = df_pcas[df_pcas["year"] == max(df_pcas["year"])]
# annotations_dict = {
#     "min_distance": min_distance,
#     "max_distance": max_distance,
#     "min_pc2": min_pc2,
#     "max_pc2": max_pc2,
#     "min_pc1": min_pc1,
#     "max_pc1": max_pc1,
#     "most_recent": most_recent,
# }


# def extract_annotations(annotations_dict: dict[str, pd.Series | Any | pd.DataFrame]):
#     text: str
#     xy: tuple[float, float]
#     xytext: tuple[float, float] | None
#     textcoords: CoordsType | None
#     fontsize: int
#     annotations = {}
#     for k, v in annotations_dict.items():
#         text = f"{v["name"]}\n{v["year"]}"
#         xy = tuple(v[["pc1", "pc2"]].to_list())
#         xytext = (5, 5)
#         textcoords = "offset points"
#         fontsize = 8

#         annotations[k] = {
#             "text": text,
#             "xy": xy,
#             "xytext": xytext,
#             "textcoords": textcoords,
#             "fontsize": fontsize,
#         }

#     return annotations

from typing import Any, Dict, Tuple, Optional, Union

import pandas as pd
import numpy as np


def create_annotation_points(df_pcas: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create annotation points from a DataFrame with PCA components.

    Args:
        df_pcas: DataFrame with pc1, pc2, name, and year columns

    Returns:
        Dictionary of DataFrames with key annotation points
    """
    # Calculate distance from origin
    df_pcas["distance"] = np.sqrt(np.square(df_pcas["pc1"]) + np.square(df_pcas["pc2"]))

    # Find extreme points
    min_distance = df_pcas.loc[df_pcas["distance"].idxmin()]
    max_distance = df_pcas.loc[df_pcas["distance"].idxmax()]
    min_pc2 = df_pcas.loc[df_pcas["pc2"].idxmin()]
    max_pc2 = df_pcas.loc[df_pcas["pc2"].idxmax()]
    min_pc1 = df_pcas.loc[df_pcas["pc1"].idxmin()]
    max_pc1 = df_pcas.loc[df_pcas["pc1"].idxmax()]
    most_recent = df_pcas.loc[df_pcas["year"].idxmax()]

    # Create dictionary of annotation points
    annotations_dict = {
        "min_distance": min_distance,
        "max_distance": max_distance,
        "min_pc2": min_pc2,
        "max_pc2": max_pc2,
        "min_pc1": min_pc1,
        "max_pc1": max_pc1,
        "most_recent": most_recent,
    }

    return annotations_dict


def extract_annotations(
    annotations_dict: Dict[str, Union[pd.Series, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Extract annotation parameters from a dictionary of Series containing annotation data.

    Args:
        annotations_dict: Dictionary of pandas Series objects with annotation data

    Returns:
        Dictionary of annotation parameters for matplotlib
    """
    annotations = {}

    for k, v in annotations_dict.items():
        # Extract text from name and year
        text = f"{v['name']}\n{v['year']}"

        # Create xy tuple from pc1 and pc2
        xy = (v["pc1"], v["pc2"])

        # Set other annotation parameters
        xytext = (5, 5)
        textcoords = "offset points"
        fontsize = 8

        annotations[k] = {
            "text": text,
            "xy": xy,
            "xytext": xytext,
            "textcoords": textcoords,
            "fontsize": fontsize,
        }

    return annotations


# Example usage:
# First create the annotation points
annotation_points = create_annotation_points(df_pcas)

# Then extract the annotation parameters
annotations = extract_annotations(annotation_points)

# # Use in a plot
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.scatter(df_pcas['pc1'], df_pcas['pc2'], c=df_pcas['year'], cmap='nipy_spectral')

# # Add annotations
# for k, anno in annotations.items():
#     ax.annotate(
#         text=anno['text'],
#         xy=anno['xy'],
#         xytext=anno['xytext'],
#         textcoords=anno['textcoords'],
#         fontsize=anno['fontsize']
#     )


def create_pca_plot(df: pd.DataFrame) -> tuple[Figure, Axes]:
    # df = df_pcas

    years: NDArray = df.loc[:, "year"].values
    pc1: NDArray = df.loc[:, "pc1"].values
    pc2: NDArray = df.loc[:, "pc2"].values
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")

    min_year, max_year = years.min(), years.max()
    norm = mcolors.Normalize(vmin=min_year, vmax=max_year)

    scatter: PathCollection = ax.scatter(
        x=pc1,
        y=pc2,
        c=years,
        cmap="nipy_spectral",
        alpha=0.8,
        s=30,
        norm=norm,
    )

    # tal vez estaría mejor colorearlos por décadas
    # por presidente
    # por partido
    # por guerras internacionales
    #
    cbar: Colorbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label("Año")

    ax.set_xlabel(
        "PC1: Lenguaje Político Formal ↔ Lenguaje Personal Directo", fontsize=12
    )
    ax.set_ylabel("PC2: Enfoque Constitucional ↔ Identidad Nacional", fontsize=12)
    ax.set_title(
        "Análisis PCA (primeras 2) de los\ndiscursos de los presidentes de USofA."
    )
    ax.axhline(0, c="k", alpha=0.2)
    ax.axvline(0, c="k", alpha=0.2)

    for k, anno in annotations.items():
        # ax.annotate(
        #     text=anno["text"],
        #     xy=anno["xy"],
        #     xytext=anno["xytext"],
        #     textcoords=anno["textcoords"],
        #     fontsize=anno["fontsize"],
        # )
        ax.plot(
            [0, anno["xy"][0]],
            [0, anno["xy"][1]],
            alpha=0.5,
            label=f"{k}:\n {anno["text"]}",
        )

    fig.legend(loc="outside right upper")
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# Get feature names (words)
feature_names = tfidf.get_feature_names_out()

# Get the components (loadings)
components = svds(speech_vectors - speech_vectors.mean(axis=0), k=2)[2]

# For each component, print the top words
for i, component in enumerate(components):
    # Get indices of top weighted words
    sorted_indices = np.argsort(np.abs(component))[::-1][:10]
    top_words = [(feature_names[idx], component[idx]) for idx in sorted_indices]
    print(f"PC{i+1} top words:")
    for word, weight in top_words:
        print(f"  {word}: {weight:.4f}")


def save_plot(fig: Figure, fname: str) -> None:
    fig.savefig(fname=fname, dpi=400, format="png", bbox_inches="tight")
    print("Se ha guardado la figura al ordenador.")
    return None


fig, _ = create_pca_plot(df_pcas)
save_plot(fig, "text_analysis_figures/figure_01.png")
