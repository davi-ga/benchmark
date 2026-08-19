import numba
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import requests
from io import BytesIO
import os
from dotenv import load_dotenv
from _numba_helpers import _sanitize_for_numba, _deep_to_list
load_dotenv()
sls_stage = os.getenv('SLS_STAGE')
if sls_stage == 'local':
    import plotly.graph_objects as go
default_k = 4
xyz_ref_white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
xyz_from_rgb = np.array([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]], dtype=np.float64)
rgb_from_xyz = np.linalg.inv(xyz_from_rgb)

@numba.njit
def rgb2xyz(rgb_arr):
    tmp = np.zeros_like(rgb_arr)
    for i in range(rgb_arr.shape[0]):
        for j in range(rgb_arr.shape[1]):
            val = rgb_arr[i, j]
            if val > 0.04045:
                tmp[i, j] = ((val + 0.055) / 1.055) ** 2.4
            else:
                tmp[i, j] = val / 12.92
    out = np.zeros_like(tmp)
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            for k in range(3):
                out[i, j] += tmp[i, k] * xyz_from_rgb[k, j]
    return out

@numba.njit
def xyz2lab(xyz_arr):
    tmp = np.zeros_like(xyz_arr)
    for i in range(xyz_arr.shape[0]):
        for j in range(xyz_arr.shape[1]):
            val = xyz_arr[i, j] / xyz_ref_white[j]
            if val > 0.008856:
                tmp[i, j] = val ** (1.0 / 3.0)
            else:
                tmp[i, j] = 7.787 * val + 16.0 / 116.0
    out = np.zeros_like(xyz_arr)
    for i in range(tmp.shape[0]):
        x = tmp[i, 0]
        y = tmp[i, 1]
        z = tmp[i, 2]
        out[i, 0] = 116.0 * y - 16.0
        out[i, 1] = 500.0 * (x - y)
        out[i, 2] = 200.0 * (y - z)
    return out

@numba.njit
def lab2xyz(lab_arr):
    tmp = np.zeros_like(lab_arr)
    for i in range(lab_arr.shape[0]):
        L = lab_arr[i, 0]
        a = lab_arr[i, 1]
        b = lab_arr[i, 2]
        y = (L + 16.0) / 116.0
        x = a / 500.0 + y
        z = y - b / 200.0
        if z < 0.0:
            z = 0.0
        tmp[i, 0] = x
        tmp[i, 1] = y
        tmp[i, 2] = z
    out = np.zeros_like(tmp)
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            val = tmp[i, j]
            if val > 0.2068966:
                out[i, j] = val ** 3.0 * xyz_ref_white[j]
            else:
                out[i, j] = (val - 16.0 / 116.0) / 7.787 * xyz_ref_white[j]
    return out

@numba.njit
def xyz2rgb(xyz_arr):
    tmp = np.zeros_like(xyz_arr)
    for i in range(xyz_arr.shape[0]):
        for j in range(xyz_arr.shape[1]):
            for k in range(3):
                tmp[i, j] += xyz_arr[i, k] * rgb_from_xyz[k, j]
    out = np.zeros_like(tmp)
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            val = tmp[i, j]
            if val > 0.0031308:
                res = 1.055 * val ** (1.0 / 2.4) - 0.055
            else:
                res = val * 12.92
            if res < 0.0:
                res = 0.0
            elif res > 1.0:
                res = 1.0
            out[i, j] = res
    return out

@numba.njit
def rgb2lab(rgb_arr):
    return xyz2lab(rgb2xyz(rgb_arr))

@numba.njit
def lab2rgb(lab_arr):
    return xyz2rgb(lab2xyz(lab_arr))

def get_lab_data(im):
    """Extrai dados LAB de imagem (requer PIL - não JIT)"""
    img_size = (150, 150)
    im.thumbnail(img_size)
    pixel_rgb = np.asarray(im)
    reshaped = pixel_rgb.reshape(-1, pixel_rgb.shape[-1]) / 255.0
    pixel_lab = rgb2lab(reshaped)
    return pixel_lab.reshape(-1, pixel_lab.shape[-1])

def make_img(colors, counts):
    """Cria imagem de paleta (requer PIL - não JIT)"""
    img_size = (512, 512)
    n_clusters = len(colors)
    lengths = ((counts / np.sum(counts) + 1.0 / n_clusters) / 2.0 * img_size[0]).astype(np.uint16)
    lengths[0] = lengths[0] + (img_size[0] - np.sum(lengths))
    pixel_group = np.array([np.tile(colors[i], (lengths[i], img_size[1], 1)) for i in range(n_clusters)])
    pixel_rgb = np.transpose(np.concatenate(pixel_group), (1, 0, 2))
    return Image.fromarray(pixel_rgb, mode='RGB')

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
def get_hex_string(rgb_arr):
    """Versão Numba-compatível de get_hex_string"""
    result = ""
    for i in range(len(rgb_arr)):
        result += int_to_hex(int(rgb_arr[i]))
    return result.upper()

def cluster_kmeans(data, n_clusters):
    """Clustering K-Means (requer sklearn - não JIT)"""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return centers, labels

def cluster_agglo(data, n_clusters):
    """Clustering aglomerativo (requer sklearn - não JIT)"""
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    labels = ac.fit_predict(data)
    centers = np.empty([n_clusters, 3])
    for i in range(n_clusters):
        subset = data[labels == i]
        if len(subset) > 0:
            centers[i] = compute_medoid(subset)
    return centers, labels

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
def compute_medoid(data):
    dist_mat = pairwise_distances_numba(data)
    return data[np.argmin(dist_mat.sum(axis=0))]

def cluster_agglo(data, n_clusters):
    """Clustering aglomerativo (requer sklearn - não JIT)"""
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    labels = ac.fit_predict(data)
    centers = np.empty([n_clusters, 3])
    for i in range(n_clusters):
        subset = data[labels == i]
        if len(subset) > 0:
            centers[i] = compute_medoid(subset)
    return centers, labels

@numba.njit
def get_cluster(centers, labels):
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

def get_palette(im, k):
    """Cria paleta de cores (wrapper Python puro - não JIT)"""
    if k is None or k < 2 or k > 10:
        k = default_k
    data = get_lab_data(im)
    centers, labels = cluster_agglo(data, k)
    sorted_centers, _, counts = get_cluster(centers, labels)
    centers_rgb = (255 * lab2rgb(sorted_centers)).astype(np.uint8)
    hex_strings = []
    for i in range(len(centers_rgb)):
        hex_strings.append(get_hex_string(centers_rgb[i]))
    return make_img(centers_rgb, counts), np.array(hex_strings)

def get_palette_plot(im, k):
    """Cria paleta com plot 3D (wrapper Python puro - não JIT)"""
    if k is None or k < 2 or k > 10:
        k = default_k
    data = get_lab_data(im)
    centers, labels = cluster_agglo(data, k)
    sorted_centers, sorted_labels, counts = get_cluster(centers, labels)
    centers_rgb = (255 * lab2rgb(sorted_centers)).astype(np.uint8)
    hex_strings = []
    for i in range(len(centers_rgb)):
        hex_strings.append(get_hex_string(centers_rgb[i]))
    centers_hex = np.array(hex_strings)
    plot_3d(data, sorted_labels, centers_hex)
    return make_img(centers_rgb, counts), centers_hex

def plot_3d(data, labels, centers_hex):
    """Plot 3D dos clusters (requer plotly - não JIT)"""
    l, a, b = np.transpose(data)
    colors = ['#' + centers_hex[labels[i]] for i in range(len(labels))]
    fig = go.Figure(data=[go.Scatter3d(x=a, y=b, z=l, mode='markers', marker={'size': 3, 'color': colors, 'opacity': 0.1})])
    fig.update_layout(scene={'xaxis_title': 'a', 'yaxis_title': 'b', 'zaxis_title': 'L'})
    fig.show()

def get_image_from_url(url):
    """Download imagem de URL (requer requests/PIL - não JIT)"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content))