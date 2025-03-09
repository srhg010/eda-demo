from eda_demo import (
    TEXT_PATH,
    Any,
    Axes,
    Colorbar,
    Dict,
    Figure,
    NDArray,
    Path,
    PathCollection,
    PorterStemmer,
    TfidfVectorizer,
    Union,
    mcolors,
    mpl,
    nltk,
    np,
    pd,
    plt,
    re,
    svds,
)
from scipy.sparse import csr_matrix, spmatrix

# un diccionario para mostrar mensajes
messages: dict[int, str] = {}

# leer el documento
text: str = TEXT_PATH.read_text()

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


def _speech_vectors() -> tuple[pd.DataFrame, csr_matrix | spmatrix | NDArray]:
    """
    Returns:
        scipy.sparse.csr_matrix: Sparse matrix de vectores TF-IDF.
            Shape: (232, 13211)
            Cada fila representa un discurso.
            Cada columna representa un token.
    """
    import warnings

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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'token_pattern' will not be used "
            + "since 'tokenizer' is not None",
            category=UserWarning,
        )
        tfidf = TfidfVectorizer(tokenizer=stemming_tokenizer)
        speech_vectors: csr_matrix | spmatrix | NDArray = tfidf.fit_transform(
            raw_documents=df.loc[:, "text"], y=None
        )  # type: ignore
        return df, speech_vectors


messages[1] = "Son 232 discursos. Se transformaron en vectores de 13,211 palabras."


def compute_pcs(data: NDArray, *, k: int | None = None) -> NDArray:
    if k == None:
        print("k no puede estar sin asignar")
    centered = data - data.mean(axis=0)
    U: NDArray
    s: NDArray
    U, s, _ = svds(centered, k=k)  # type: ignore
    return U @ np.diag(s)


df, speech_vectors = _speech_vectors()

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


def create_annotation_points(df_pcas: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df_pcas["distance"] = np.sqrt(np.square(df_pcas["pc1"]) + np.square(df_pcas["pc2"]))

    min_distance = df_pcas.loc[df_pcas["distance"].idxmin()]
    max_distance = df_pcas.loc[df_pcas["distance"].idxmax()]
    min_pc2 = df_pcas.loc[df_pcas["pc2"].idxmin()]
    max_pc2 = df_pcas.loc[df_pcas["pc2"].idxmax()]
    min_pc1 = df_pcas.loc[df_pcas["pc1"].idxmin()]
    max_pc1 = df_pcas.loc[df_pcas["pc1"].idxmax()]
    most_recent = df_pcas.loc[df_pcas["year"].idxmax()]

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
    annotations = {}

    for k, v in annotations_dict.items():
        text = f"{v['name']}\n{v['year']}"

        xy = (v["pc1"], v["pc2"])

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


annotation_points = create_annotation_points(df_pcas)

annotations = extract_annotations(annotation_points)


def create_pca_plot(df: pd.DataFrame = df_pcas) -> tuple[Figure, Axes]:
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
        ax.plot(
            [0, anno["xy"][0]],
            [0, anno["xy"][1]],
            alpha=0.5,
            label=f"{k}:\n {anno["text"]}",
        )

    fig.legend(loc="outside right upper")
    return fig, ax


# feature_names = tfidf.get_feature_names_out()

# components = svds(speech_vectors - speech_vectors.mean(axis=0), k=2)[2]

# for i, component in enumerate(components):
#     sorted_indices = np.argsort(np.abs(component))[::-1][:10]
#     top_words = [(feature_names[idx], component[idx]) for idx in sorted_indices]
#     print(f"PC{i+1} top words:")
#     for word, weight in top_words:
#         print(f"  {word}: {weight:.4f}")


# def save_plot(fig: Figure, fname: str) -> None:
#     fig.savefig(fname=fname, dpi=400, format="png", bbox_inches="tight")
#     print("Se ha guardado la figura al ordenador.")
#     return None


# fig, _ = create_pca_plot(df_pcas)
# save_plot(fig, "text_analysis_figures/figure_01.png")

figure_01 = create_pca_plot
figs = {
    "figure_01": figure_01,
}

figs_dirpath = Path(".").resolve() / "text_analysis_figures_02"
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

        save_path = figs_dirpath.name + "/" + func_name + ".png"
        fig.savefig(save_path, dpi=300)
        plt.close()
        print(f"\nSe guardó la imagen {func_name}.\n", end="\u2a69" * (1 + i) + "\n")
        print(f"Quedan {len(fl)- 1 - i}")
    except NameError:
        print(f"Todavía no está lista {func_name}")

print("Se guardaron todas las imágenes.")
