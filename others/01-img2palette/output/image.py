import numba
'\nHelper functions for image processing\n\nRefactored to separate CPU-bound loops for Numba compilation.\n'
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import requests
from io import BytesIO
import os
from dotenv import load_dotenv
from numba import njit
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _np_unique_with_counts(arr):
    """Implementação Numba-compatível de np.unique(arr, return_counts=True)"""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    unique_count = 1
    for i in range(1, n):
        if sorted_arr[i] != sorted_arr[i - 1]:
            unique_count += 1
    unique_vals = np.empty(unique_count, dtype=arr.dtype)
    counts = np.empty(unique_count, dtype=np.int64)
    unique_vals[0] = sorted_arr[0]
    counts[0] = 1
    idx = 0
    for i in range(1, n):
        if sorted_arr[i] != sorted_arr[i - 1]:
            idx += 1
            unique_vals[idx] = sorted_arr[i]
            counts[idx] = 1
        else:
            counts[idx] += 1
    return (unique_vals, counts)

@numba.njit
def _int_to_hex(integer):
    """Converte inteiro para string hex (sem usar hex() built-in)"""
    hex_chars = '0123456789abcdef'
    if integer == 0:
        return '00'
    result = ''
    while integer > 0:
        result = hex_chars[integer % 16] + result
        integer = integer // 16
    if len(result) < 2:
        result = '0' + result
    return result

@numba.njit
def _int_to_bin(integer):
    """Converte inteiro para string binária (sem usar bin() built-in)"""
    if integer == 0:
        return '0'
    result = ''
    while integer > 0:
        result = str(integer % 2) + result
        integer = integer // 2
    return result

@numba.njit
def _int_to_oct(integer):
    """Converte inteiro para string octal (sem usar oct() built-in)"""
    if integer == 0:
        return '0'
    result = ''
    while integer > 0:
        result = str(integer % 8) + result
        integer = integer // 8
    return result
load_dotenv()
sls_stage = os.getenv('SLS_STAGE')
if sls_stage == 'local':
    import plotly.graph_objects as go
default_k = 4
xyz_ref_white = np.asarray((0.95047, 1.0, 1.08883), dtype=np.float64)
xyz_from_rgb = np.array([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]], dtype=np.float64)
rgb_from_xyz = np.linalg.inv(xyz_from_rgb)

@njit
def rgb2xyz(rgb_arr):
    n = rgb_arr.shape[0]
    xyz_arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for j in range(3):
            val = rgb_arr[i, j]
            if val > 0.04045:
                val = np.power((val + 0.055) / 1.055, 2.4)
            else:
                val /= 12.92
            xyz_arr[i, j] = val
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += xyz_arr[i, k] * xyz_from_rgb[j, k]
            out[i, j] = s
    return out

@njit
def xyz2lab(xyz_arr):
    n = xyz_arr.shape[0]
    lab_arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for j in range(3):
            val = xyz_arr[i, j] / xyz_ref_white[j]
            if val > 0.008856:
                val = np.power(val, 1.0 / 3.0)
            else:
                val = 7.787 * val + 16.0 / 116.0
            lab_arr[i, j] = val
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        x, y, z = (lab_arr[i, 0], lab_arr[i, 1], lab_arr[i, 2])
        L = 116.0 * y - 16.0
        a = 500.0 * (x - y)
        b = 200.0 * (y - z)
        out[i, 0] = L
        out[i, 1] = a
        out[i, 2] = b
    return out

@njit
def lab2xyz(lab_arr):
    n = lab_arr.shape[0]
    xyz_arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        L, a, b = (lab_arr[i, 0], lab_arr[i, 1], lab_arr[i, 2])
        y = (L + 16.0) / 116.0
        x = a / 500.0 + y
        z = y - b / 200.0
        if z < 0:
            z = 0.0
        for j, val in enumerate((x, y, z)):
            if val > 0.2068966:
                val = np.power(val, 3.0)
            else:
                val = (val - 16.0 / 116.0) / 7.787
            xyz_arr[i, j] = val * xyz_ref_white[j]
    return xyz_arr

@njit
def xyz2rgb(xyz_arr):
    n = xyz_arr.shape[0]
    rgb_arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += xyz_arr[i, k] * rgb_from_xyz[j, k]
            rgb_arr[i, j] = s
        for j in range(3):
            val = rgb_arr[i, j]
            if val > 0.0031308:
                val = 1.055 * np.power(val, 1 / 2.4) - 0.055
            else:
                val *= 12.92
            if val < 0:
                val = 0.0
            if val > 1:
                val = 1.0
            rgb_arr[i, j] = val
    return rgb_arr

@njit
def rgb2lab(rgb_arr):
    xyz = rgb2xyz(rgb_arr)
    return xyz2lab(xyz)

@njit
def lab2rgb(lab_arr):
    xyz = lab2xyz(lab_arr)
    return xyz2rgb(xyz)

def get_lab_data(im):
    img_size = (150, 150)
    im.thumbnail(img_size)
    pixel_rgb = np.asarray(im, dtype=np.float64)
    reshaped_rgb = pixel_rgb.reshape(-1, pixel_rgb.shape[-1]) / 255.0
    pixel_lab = rgb2lab(reshaped_rgb)
    return pixel_lab

def make_img(colors, counts):
    img_size = (512, 512)
    n_clusters = len(colors)
    lengths = ((counts / np.sum(counts) + 1.0 / n_clusters) / 2.0 * img_size[0]).astype(np.uint16)
    lengths[0] = lengths[0] + (img_size[0] - np.sum(lengths))
    pixel_group = np.array([np.tile(colors[i], (lengths[i], img_size[1], 1)) for i in range(n_clusters)])
    pixel_rgb = np.transpose(np.concatenate(pixel_group), (1, 0, 2))
    return Image.fromarray(pixel_rgb, mode='RGB')

@numba.njit
def _get_hex_string_jit(rgb_arr):

    def int2hex(integer):
        hex_string = _int_to_hex(integer)
        if len(hex_string) < 2:
            return '0' + hex_string
        return hex_string
    return ''.join([int2hex(x) for x in rgb_arr]).upper()

def get_hex_string(rgb_arr):
    return _deep_to_list(_get_hex_string_jit(_sanitize_for_numba(rgb_arr)))

def _cluster_kmeans_jit(data, n_clusters):
    kmeans = KMeans(n_clusters)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return (centers, labels)

def cluster_kmeans(data, n_clusters):
    return _deep_to_list(_cluster_kmeans_jit(_sanitize_for_numba(data), _sanitize_for_numba(n_clusters)))

def compute_medoid(data):
    dist_mat = pairwise_distances(data)
    return data[np.argmin(dist_mat.sum(axis=0))]

def cluster_agglo(data, n_clusters):
    ac = AgglomerativeClustering(n_clusters)
    labels = ac.fit_predict(data)
    centers = np.empty([n_clusters, 3])
    for i in range(n_clusters):
        subset = data[labels == i]
        if len(subset) > 0:
            centers[i] = compute_medoid(subset)
    return (centers, labels)

@numba.njit
def get_cluster(centers, labels):
    _, counts = _np_unique_with_counts(labels)
    sort_idx = (-counts).argsort()
    sort_labels = [(lambda i: list(sort_idx).index(i))(x) for x in labels]
    return (centers[sort_idx], sort_labels, counts[sort_idx])

@numba.njit
def _get_palette_jit(im, k):
    if k is None:
        k = default_k
    elif k < 2 or k > 10:
        k = default_k
    data = get_lab_data(im)
    centers, labels = cluster_agglo(data, k)
    sorted_centers, _, counts = get_cluster(centers, labels)
    centers_rgb = (255 * lab2rgb(sorted_centers)).astype(np.uint8)
    return (make_img(centers_rgb, counts), np.apply_along_axis(get_hex_string, 1, centers_rgb))

def get_palette(im, k):
    return _deep_to_list(_get_palette_jit(_sanitize_for_numba(im), _sanitize_for_numba(k)))

@numba.njit
def _get_palette_plot_jit(im, k):
    if k is None:
        k = default_k
    elif k < 2 or k > 10:
        k = default_k
    data = get_lab_data(im)
    centers, labels = cluster_agglo(data, k)
    sorted_centers, sorted_labels, counts = get_cluster(centers, labels)
    centers_rgb = (255 * lab2rgb(sorted_centers)).astype(np.uint8)
    centers_hex = np.apply_along_axis(get_hex_string, 1, centers_rgb)
    plot_3d(data, sorted_labels, centers_hex)
    return (make_img(centers_rgb, counts), centers_hex)

def get_palette_plot(im, k):
    return _deep_to_list(_get_palette_plot_jit(_sanitize_for_numba(im), _sanitize_for_numba(k)))

@numba.njit
def plot_3d(data, labels, centers_hex):
    l, a, b = np.transpose(data)
    fig = go.Figure(data=[go.Scatter3d(x=a, y=b, z=l, mode='markers', marker={'size': 3, 'color': [(lambda hex: '#' + hex)(x) for x in centers_hex][labels], 'opacity': 0.1})])
    fig.update_layout(scene={'xaxis_title': 'a', 'yaxis_title': 'b', 'zaxis_title': 'L'})
    fig.show()

def _get_image_from_url_jit(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def get_image_from_url(url):
    return _deep_to_list(_get_image_from_url_jit(_sanitize_for_numba(url)))