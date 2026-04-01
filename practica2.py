import pandas as pd
import numpy as np

# Cargar dataset reducido
df = pd.read_csv("news_reducido.csv")

# Eliminar columnas basura
cols_basura = [c for c in df.columns if c.startswith("Unnamed")]
df = df.drop(columns=cols_basura)

# Rellenar nulos en columnas de texto
for col in ["headline", "short_description", "text", "authors"]:
    if col in df.columns:
        df[col] = df[col].fillna("")

# Crear columna de texto final
df["content"] = (
    df["headline"].astype(str) + " " +
    df["short_description"].astype(str) + " " +
    df["text"].astype(str)
)

# Mostrar comprobaciones
print(df.shape)
print(df.columns)
print(df["category"].value_counts())
print(df["content"].head())

from sklearn.preprocessing import LabelEncoder

# Variable de entrada (texto)
X_text = df["content"]

# Variable objetivo (clase)
y = df["category"]

# Convertir etiquetas a números
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Comprobaciones
print("Ejemplo texto:")
print(X_text.iloc[0])

print("\nEtiqueta original:", y.iloc[0])
print("Etiqueta codificada:", y_encoded[0])

print("\nClases:")
print(le.classes_)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Representación binaria
vect_bin = CountVectorizer(binary=True, stop_words='english', min_df=2)
X_bin = vect_bin.fit_transform(X_text)

# Representación por frecuencia
vect_freq = CountVectorizer(binary=False, stop_words='english', min_df=2)
X_freq = vect_freq.fit_transform(X_text)

# Representación TF-IDF
vect_tfidf = TfidfVectorizer(stop_words='english', min_df=2)
X_tfidf = vect_tfidf.fit_transform(X_text)

# Comprobaciones
print("Forma X_bin:", X_bin.shape)
print("Forma X_freq:", X_freq.shape)
print("Forma X_tfidf:", X_tfidf.shape)

print("\nNúmero de términos binaria:", len(vect_bin.get_feature_names_out()))
print("Número de términos frecuencia:", len(vect_freq.get_feature_names_out()))
print("Número de términos tfidf:", len(vect_tfidf.get_feature_names_out()))

print("\nPrimeros 20 términos:")
print(vect_tfidf.get_feature_names_out()[:20])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


def clustering_experimento(X, y, nombre):
    print(f"\n--- {nombre} ---")

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Métrica interna
    sil = silhouette_score(X, labels)

    # Métricas externas
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)

    print("Silhouette:", sil)
    print("ARI:", ari)
    print("NMI:", nmi)


# Ejecutar para cada representación
clustering_experimento(X_bin, y_encoded, "Binaria")
clustering_experimento(X_freq, y_encoded, "Frecuencia")
clustering_experimento(X_tfidf, y_encoded, "TF-IDF")

def clustering_semillas(X, y, nombre):
    print(f"\n--- {nombre} (diferentes semillas) ---")

    for seed in [0, 10, 42, 100]:
        kmeans = KMeans(n_clusters=4, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X)

        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        print(f"Seed {seed} -> ARI: {ari:.4f} | NMI: {nmi:.4f}")


# Probar distintas semillas en TF-IDF
clustering_semillas(X_tfidf, y_encoded, "TF-IDF")


# Mejor modelo (según resultados)
kmeans_best = KMeans(n_clusters=4, random_state=10, n_init=10)
labels_best = kmeans_best.fit_predict(X_tfidf)

# Guardar en dataframe
df_result = df.copy()
df_result["cluster"] = labels_best

# Guardar a CSV
df_result[["content", "category", "cluster"]].to_csv("clusters_resultado.csv", index=False)

print("\nClusters guardados en clusters_resultado.csv")



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Convertir TF-IDF a denso (cuidado, tarda un poco)
X_dense = X_tfidf.toarray()

# Reducir dimensiones
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X_dense)

# Dibujar clusters
plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels_best, s=5)
plt.title("t-SNE de clusters (TF-IDF)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()




from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD

print("\n--- Gaussian Mixture (TF-IDF) ---")

# Reducir dimensionalidad (IMPORTANTE)
svd = TruncatedSVD(n_components=100, random_state=42)
X_red = svd.fit_transform(X_tfidf)

# Modelo GMM
gmm = GaussianMixture(n_components=4, random_state=42)
labels_gmm = gmm.fit_predict(X_red)

# Evaluación
ari_gmm = adjusted_rand_score(y_encoded, labels_gmm)
nmi_gmm = normalized_mutual_info_score(y_encoded, labels_gmm)

print("ARI (GMM):", ari_gmm)
print("NMI (GMM):", nmi_gmm)