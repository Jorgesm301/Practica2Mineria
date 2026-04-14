import time
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


SEEDS = [0, 10, 42, 100]


def medir_memoria_y_tiempo(func, *args, **kwargs):
    """Ejecuta una funcion y devuelve resultado, tiempo y pico de memoria en MB."""
    tracemalloc.start()
    inicio = time.perf_counter()
    salida = func(*args, **kwargs)
    fin = time.perf_counter()
    _, memoria_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return salida, fin - inicio, memoria_pico / (1024 * 1024)


def cargar_datos(path_csv="news_reducido.csv"):
    df = pd.read_csv(path_csv)
    cols_basura = [c for c in df.columns if c.startswith("Unnamed")]
    if cols_basura:
        df = df.drop(columns=cols_basura)

    # Requisito del usuario: usar solo text para todo.
    df["text"] = df["text"].fillna("").astype(str)
    df["category"] = df["category"].astype(str)

    X_text = df["text"]
    y = df["category"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return df, X_text, y, y_encoded


def construir_representaciones(X_text):
    vect_bin = CountVectorizer(binary=True, stop_words="english", min_df=2)
    vect_freq = CountVectorizer(binary=False, stop_words="english", min_df=2)
    vect_tfidf = TfidfVectorizer(stop_words="english", min_df=2)

    X_bin = vect_bin.fit_transform(X_text)
    X_freq = vect_freq.fit_transform(X_text)
    X_tfidf = vect_tfidf.fit_transform(X_text)

    representaciones = {
        "Binaria": X_bin,
        "Frecuencia": X_freq,
        "TF-IDF": X_tfidf,
    }
    return representaciones


def evaluar_clustering(X, labels, y_true):
    if hasattr(X, "toarray"):
        svd_eval = TruncatedSVD(n_components=50, random_state=42)
        X_eval_dense = svd_eval.fit_transform(X)
    else:
        X_eval_dense = X

    return {
        "silhouette": silhouette_score(X, labels, sample_size=2000, random_state=42),
        "calinski_harabasz": calinski_harabasz_score(X_eval_dense, labels),
        "davies_bouldin": davies_bouldin_score(X_eval_dense, labels),
        "ari": adjusted_rand_score(y_true, labels),
        "nmi": normalized_mutual_info_score(y_true, labels),
    }


def ejecutar_agrupamiento(df, representaciones, y_encoded):
    filas_clustering = []

    print("\n=== 3.2 Agrupamiento: KMeans (K=4) con 3+ semillas ===")
    for rep_nombre, X in representaciones.items():
        print(f"\n[{rep_nombre}]")
        for seed in SEEDS:
            kmeans = KMeans(n_clusters=4, random_state=seed, n_init=10)
            labels, duracion, memoria = medir_memoria_y_tiempo(kmeans.fit_predict, X)
            metricas = evaluar_clustering(X, labels, y_encoded)

            fila = {
                "algoritmo": "KMeans",
                "representacion": rep_nombre,
                "seed": seed,
                "tiempo_s": duracion,
                "memoria_mb": memoria,
                **metricas,
            }
            filas_clustering.append(fila)
            print(
                f"Seed {seed} | ARI={metricas['ari']:.4f} NMI={metricas['nmi']:.4f} "
                f"Sil={metricas['silhouette']:.4f}"
            )

    # EM solo en TF-IDF
    print("\n=== EM (GaussianMixture) solo en TF-IDF ===")
    X_tfidf = representaciones["TF-IDF"]
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_tfidf_red = svd.fit_transform(X_tfidf)
    for seed in SEEDS:
        gmm = GaussianMixture(n_components=4, random_state=seed)
        labels_gmm, duracion, memoria = medir_memoria_y_tiempo(gmm.fit_predict, X_tfidf_red)
        metricas = {
            "silhouette": silhouette_score(X_tfidf_red, labels_gmm, sample_size=2000, random_state=42),
            "calinski_harabasz": calinski_harabasz_score(X_tfidf_red, labels_gmm),
            "davies_bouldin": davies_bouldin_score(X_tfidf_red, labels_gmm),
            "ari": adjusted_rand_score(y_encoded, labels_gmm),
            "nmi": normalized_mutual_info_score(y_encoded, labels_gmm),
        }
        filas_clustering.append(
            {
                "algoritmo": "GaussianMixture",
                "representacion": "TF-IDF",
                "seed": seed,
                "tiempo_s": duracion,
                "memoria_mb": memoria,
                **metricas,
            }
        )
        print(f"Seed {seed} | ARI={metricas['ari']:.4f} NMI={metricas['nmi']:.4f}")

    df_clustering = pd.DataFrame(filas_clustering)
    df_clustering.to_csv("resultados_clustering.csv", index=False)

    mejores = df_clustering[df_clustering["algoritmo"] == "KMeans"].sort_values(
        by=["ari", "nmi", "silhouette"], ascending=False
    )
    mejor = mejores.iloc[0]

    kmeans_best = KMeans(
        n_clusters=4,
        random_state=int(mejor["seed"]),
        n_init=10,
    )
    labels_best = kmeans_best.fit_predict(representaciones[mejor["representacion"]])
    df_asig = df[["text", "category"]].copy()
    df_asig["cluster"] = labels_best
    df_asig["representacion"] = mejor["representacion"]
    df_asig["seed"] = int(mejor["seed"])
    df_asig.to_csv("clusters_resultado.csv", index=False)
    print("Asignaciones guardadas en clusters_resultado.csv")

    # Visualizacion t-SNE del mejor clustering (sobre TF-IDF por consistencia).
    X_tsne_base = representaciones["TF-IDF"]
    svd_tsne = TruncatedSVD(n_components=50, random_state=42)
    X_tsne_red = svd_tsne.fit_transform(X_tsne_base)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    X_2d = tsne.fit_transform(X_tsne_red)

    plt.figure(figsize=(9, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_best, s=6)
    plt.title("t-SNE de clusters (labels del mejor KMeans)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig("tsne_clusters.png", dpi=200)
    plt.close()
    print("Visualizacion guardada en tsne_clusters.png")

    return df_clustering


def ejecutar_clasificacion(representaciones, y_encoded):
    print("\n=== 3.3 Clasificacion: k-NN y Naive Bayes ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    filas = []

    configuraciones_knn = [
        {"n_neighbors": 1, "weights": "uniform", "p": 1},
        {"n_neighbors": 1, "weights": "uniform", "p": 2},
        {"n_neighbors": 3, "weights": "uniform", "p": 2},
        {"n_neighbors": 5, "weights": "distance", "p": 2},
        {"n_neighbors": 7, "weights": "distance", "p": 1},
    ]

    for rep_nombre, X in representaciones.items():
        print(f"\n[{rep_nombre}]")
        for cfg in configuraciones_knn:
            accs = []
            f1s = []
            t_total = 0.0
            m_total = 0.0
            for train_idx, test_idx in skf.split(X, y_encoded):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

                modelo = KNeighborsClassifier(
                    n_neighbors=cfg["n_neighbors"],
                    weights=cfg["weights"],
                    p=cfg["p"],
                    metric="minkowski",
                    n_jobs=-1,
                )

                (_, y_pred), duracion, memoria = medir_memoria_y_tiempo(
                    lambda: (modelo.fit(X_train, y_train), modelo.predict(X_test))
                )
                accs.append(accuracy_score(y_test, y_pred))
                f1s.append(f1_score(y_test, y_pred, average="macro"))
                t_total += duracion
                m_total += memoria

            filas.append(
                {
                    "algoritmo": "k-NN",
                    "representacion": rep_nombre,
                    "params": f"k={cfg['n_neighbors']},w={cfg['weights']},p={cfg['p']}",
                    "accuracy_mean": float(np.mean(accs)),
                    "f1_macro_mean": float(np.mean(f1s)),
                    "tiempo_total_s": t_total,
                    "memoria_media_mb": m_total / skf.n_splits,
                }
            )
            print(
                f"  k-NN {cfg['n_neighbors']}, {cfg['weights']}, p={cfg['p']} -> "
                f"acc={np.mean(accs):.4f}, f1={np.mean(f1s):.4f}, tiempo={t_total:.2f}s"
            )

        svd_nb = TruncatedSVD(n_components=200, random_state=42)
        X_dense = svd_nb.fit_transform(X)
        accs_g = []
        f1s_g = []
        t_total_g = 0.0
        m_total_g = 0.0
        for train_idx, test_idx in skf.split(X_dense, y_encoded):
            X_train, X_test = X_dense[train_idx], X_dense[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            modelo = GaussianNB()
            (_, y_pred), duracion, memoria = medir_memoria_y_tiempo(
                lambda: (modelo.fit(X_train, y_train), modelo.predict(X_test))
            )
            accs_g.append(accuracy_score(y_test, y_pred))
            f1s_g.append(f1_score(y_test, y_pred, average="macro"))
            t_total_g += duracion
            m_total_g += memoria

        filas.append(
            {
                "algoritmo": "GaussianNB",
                "representacion": rep_nombre,
                "params": "default",
                "accuracy_mean": float(np.mean(accs_g)),
                "f1_macro_mean": float(np.mean(f1s_g)),
                "tiempo_total_s": t_total_g,
                "memoria_media_mb": m_total_g / skf.n_splits,
            }
        )
        print(
            f"  GaussianNB -> acc={np.mean(accs_g):.4f}, f1={np.mean(f1s_g):.4f}, tiempo={t_total_g:.2f}s"
        )

        accs_m = []
        f1s_m = []
        t_total_m = 0.0
        m_total_m = 0.0
        for train_idx, test_idx in skf.split(X, y_encoded):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            modelo = MultinomialNB()
            (_, y_pred), duracion, memoria = medir_memoria_y_tiempo(
                lambda: (modelo.fit(X_train, y_train), modelo.predict(X_test))
            )
            accs_m.append(accuracy_score(y_test, y_pred))
            f1s_m.append(f1_score(y_test, y_pred, average="macro"))
            t_total_m += duracion
            m_total_m += memoria

        filas.append(
            {
                "algoritmo": "MultinomialNB",
                "representacion": rep_nombre,
                "params": "default",
                "accuracy_mean": float(np.mean(accs_m)),
                "f1_macro_mean": float(np.mean(f1s_m)),
                "tiempo_total_s": t_total_m,
                "memoria_media_mb": m_total_m / skf.n_splits,
            }
        )
        print(
            f"  MultinomialNB -> acc={np.mean(accs_m):.4f}, f1={np.mean(f1s_m):.4f}, tiempo={t_total_m:.2f}s"
        )

    df_clf = pd.DataFrame(filas)
    df_clf.to_csv("resultados_clasificacion.csv", index=False)

    ranking = df_clf.sort_values(
        by=["accuracy_mean", "f1_macro_mean", "tiempo_total_s"],
        ascending=[False, False, True],
    )
    mejor = ranking.iloc[0]
    print("Resultados guardados en resultados_clasificacion.csv")
    return df_clf



def main():
    df, X_text, _, y_encoded = cargar_datos("news_reducido.csv")
    representaciones = construir_representaciones(X_text)
    df_cluster = ejecutar_agrupamiento(df, representaciones, y_encoded)
    df_clf = ejecutar_clasificacion(representaciones, y_encoded)

if __name__ == "__main__":
    main()
