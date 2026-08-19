"""
Correções para img2palette - funções Numba-compatíveis
"""
import numpy as np
import numba


@numba.njit
def int_to_hex(integer):
    """Converte inteiro para string hex (sem usar hex() built-in)"""
    hex_chars = "0123456789abcdef"
    if integer == 0:
        return "00"
    result = ""
    while integer > 0:
        result = hex_chars[integer % 16] + result
        integer = integer // 16
    if len(result) < 2:
        result = "0" + result
    return result


@numba.njit
def get_hex_string_numba(rgb_arr):
    """Versão Numba-compatível de get_hex_string"""
    result = ""
    for i in range(len(rgb_arr)):
        result += int_to_hex(int(rgb_arr[i]))
    return result.upper()


@numba.njit
def pairwise_distances_numba(data):
    """Calcula matriz de distâncias euclidianas (substitui sklearn.pairwise_distances)"""
    n = len(data)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = 0.0
            for k in range(data.shape[1]):
                diff = data[i, k] - data[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat


@numba.njit
def compute_medoid_numba(data):
    """Versão Numba-compatível de compute_medoid"""
    dist_mat = pairwise_distances_numba(data)
    return data[np.argmin(dist_mat.sum(axis=0))]


@numba.njit
def get_cluster_numba(centers, labels):
    """Versão Numba-compatível de get_cluster"""
    # Conta ocorrências de cada label
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    counts = np.zeros(n_clusters, dtype=np.int64)
    
    for i in range(len(labels)):
        for j in range(n_clusters):
            if labels[i] == unique_labels[j]:
                counts[j] += 1
                break
    
    # Ordena por contagem (decrescente)
    sort_idx = np.argsort(-counts)
    
    # Mapeia labels para índices ordenados
    sort_labels = np.zeros(len(labels), dtype=np.int64)
    for i in range(len(labels)):
        for j in range(n_clusters):
            if labels[i] == unique_labels[sort_idx[j]]:
                sort_labels[i] = j
                break
    
    return centers[sort_idx], sort_labels, counts[sort_idx]


# Funções que NÃO podem ser JIT (usam sklearn)
# Estas devem permanecer como wrappers Python puros

def cluster_kmeans(data, n_clusters):
    """Clustering K-Means (requer sklearn - não JIT)"""
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return centers, labels


def cluster_agglo(data, n_clusters):
    """Clustering aglomerativo (requer sklearn - não JIT)"""
    from sklearn.cluster import AgglomerativeClustering
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    labels = ac.fit_predict(data)
    centers = np.empty([n_clusters, 3])
    for i in range(n_clusters):
        subset = data[labels == i]
        if len(subset) > 0:
            centers[i] = compute_medoid_numba(subset)
    return centers, labels
