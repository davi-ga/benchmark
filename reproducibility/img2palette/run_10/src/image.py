"""
Helper functions for image processing
"""
import numpy as np
import os
try:
    from io import BytesIO
except ImportError:
    BytesIO, = (None,)
try:
    import requests
except ImportError:
    requests = None
try:
    from PIL import Image
except ImportError:
    Image, = (None,)
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
except ImportError:
    KMeans, AgglomerativeClustering = (None, None)
try:
    from sklearn.metrics import pairwise_distances
except ImportError:
    pairwise_distances, = (None,)
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
import numba
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
sls_stage = os.getenv('SLS_STAGE')
default_k = 4
xyz_ref_white = np.asarray((0.95047, 1.0, 1.08883))
xyz_from_rgb = np.array([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]])
rgb_from_xyz = np.linalg.inv(xyz_from_rgb)

@numba.njit
def _rgb2xyz_core(rgb_arr, xyz_from_rgb_mat):
    n = rgb_arr.shape[0]
    xyz_arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for j in range(3):
            val = rgb_arr[i, j]
            if val > 0.04045:
                xyz_arr[i, j] = np.power((val + 0.055) / 1.055, 2.4)
            else:
                xyz_arr[i, j] = val / 12.92
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for j in range(3):
            out[i, j] = xyz_arr[i, 0] * xyz_from_rgb_mat[j, 0] + xyz_arr[i, 1] * xyz_from_rgb_mat[j, 1] + xyz_arr[i, 2] * xyz_from_rgb_mat[j, 2]
    return out

@numba.njit
def rgb2xyz(rgb_arr):
    """Convert color from RGB to CIE 1931 XYZ"""
    return _rgb2xyz_core(rgb_arr, xyz_from_rgb)

@numba.njit
def _xyz2lab_core(xyz_arr, ref_white):
    n = xyz_arr.shape[0]
    lab_arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for j in range(3):
            val = xyz_arr[i, j] / ref_white[j]
            if val > 0.008856:
                lab_arr[i, j] = np.power(val, 1.0 / 3.0)
            else:
                lab_arr[i, j] = 7.787 * val + 16.0 / 116.0
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        x, y, z = (lab_arr[i, 0], lab_arr[i, 1], lab_arr[i, 2])
        out[i, 0] = 116.0 * y - 16.0
        out[i, 1] = 500.0 * (x - y)
        out[i, 2] = 200.0 * (y - z)
    return out

@numba.njit
def xyz2lab(xyz_arr):
    """Convert color from CIE 1931 XYZ to CIE 1976 L*a*b*"""
    return _xyz2lab_core(xyz_arr, xyz_ref_white)

@numba.njit
def _lab2xyz_core(lab_arr, ref_white):
    n = lab_arr.shape[0]
    xyz_arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        L, a, b = (lab_arr[i, 0], lab_arr[i, 1], lab_arr[i, 2])
        y = (L + 16.0) / 116.0
        x = a / 500.0 + y
        z = y - b / 200.0
        if z < 0:
            z = 0.0
        xyz_arr[i, 0] = x
        xyz_arr[i, 1] = y
        xyz_arr[i, 2] = z
    for i in range(n):
        for j in range(3):
            val = xyz_arr[i, j]
            if val > 0.2068966:
                xyz_arr[i, j] = np.power(val, 3.0)
            else:
                xyz_arr[i, j] = (val - 16.0 / 116.0) / 7.787
            xyz_arr[i, j] *= ref_white[j]
    return xyz_arr

@numba.njit
def lab2xyz(lab_arr):
    """Convert color from CIE 1976 L*a*b* to CIE 1931 XYZ"""
    return _lab2xyz_core(lab_arr, xyz_ref_white)

@numba.njit
def _xyz2rgb_core(xyz_arr, rgb_from_xyz_mat):
    n = xyz_arr.shape[0]
    rgb_arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for j in range(3):
            rgb_arr[i, j] = xyz_arr[i, 0] * rgb_from_xyz_mat[j, 0] + xyz_arr[i, 1] * rgb_from_xyz_mat[j, 1] + xyz_arr[i, 2] * rgb_from_xyz_mat[j, 2]
    for i in range(n):
        for j in range(3):
            val = rgb_arr[i, j]
            if val > 0.0031308:
                val = 1.055 * np.power(val, 1 / 2.4) - 0.055
            else:
                val = val * 12.92
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            rgb_arr[i, j] = val
    return rgb_arr

@numba.njit
def xyz2rgb(xyz_arr):
    """Convert color from CIE 1931 XYZ to RGB"""
    return _xyz2rgb_core(xyz_arr, rgb_from_xyz)

@numba.njit
def rgb2lab(rgb_arr):
    """Convert color from RGB to CIE 1976 L*a*b*"""
    return xyz2lab(rgb2xyz(rgb_arr))

@numba.njit
def lab2rgb(lab_arr):
    """Convert color from CIE 1976 L*a*b* to RGB"""
    return xyz2rgb(lab2xyz(lab_arr))

def get_lab_data(im):
    """Convert colur from CIE 1976 L*a*b* to RGB"""
    img_size = (150, 150)
    im.thumbnail(img_size)
    pixel_rgb = np.asarray(im)
    pixel_lab = rgb2lab(pixel_rgb.reshape(-1, pixel_rgb.shape[-1]) / 255.0)
    return pixel_lab.reshape(-1, pixel_lab.shape[-1])

def make_img(colors, counts):
    """Create image from colors"""
    img_size = (512, 512)
    n_clusters = len(colors)
    lengths = ((counts / np.sum(counts) + 1.0 / n_clusters) / 2.0 * img_size[0]).astype(np.uint16)
    lengths[0] = lengths[0] + (img_size[0] - np.sum(lengths))
    pixel_group = np.concatenate([np.tile(colors[i], (lengths[i], img_size[1], 1)) for i in range(n_clusters)], axis=0)
    pixel_rgb = np.transpose(pixel_group, (1, 0, 2))
    return Image.fromarray(pixel_rgb, mode='RGB')

def get_hex_string(rgb_arr):
    """Covert RGB color to HEX values"""

    def int2hex(integer):
        hex_string = _int_to_hex(integer)
        if len(hex_string) < 2:
            return '0' + hex_string
        return hex_string
    return ''.join([int2hex(x) for x in rgb_arr]).upper()

def cluster_kmeans(data, n_clusters):
    """Partition data with k-means clustering"""
    kmeans = KMeans(n_clusters)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return (centers, labels)

def compute_medoid(data):
    """Get medoid of data"""
    dist_mat = pairwise_distances(data)
    return data[np.argmin(dist_mat.sum(axis=0))]

def cluster_agglo(data, n_clusters):
    """Partition data with agglomerative clustering"""
    ac = AgglomerativeClustering(n_clusters)
    labels = ac.fit_predict(data)
    centers = np.empty([n_clusters, 3])
    for i in range(n_clusters):
        centers[i] = compute_medoid(data[labels == i])
    return (centers, labels)

def get_cluster(centers, labels):
    """Sort cluster centers and count number of labels"""
    _, counts = _np_unique_with_counts(labels)
    sort_idx = (-counts).argsort()
    sort_labels = [(lambda i: list(sort_idx).index(i))(x) for x in labels]
    return (centers[sort_idx], sort_labels, counts[sort_idx])

def get_palette(im, k):
    """Create a palette from an image"""
    if k is None:
        k = default_k
    elif k < 2 or k > 10:
        k = default_k
    data = get_lab_data(im)
    centers, labels = cluster_agglo(data, k)
    sorted_centers, _, counts = get_cluster(centers, labels)
    centers_rgb = (255 * lab2rgb(sorted_centers)).astype(np.uint8)
    return (make_img(centers_rgb, counts), np.apply_along_axis(get_hex_string, 1, centers_rgb))

def get_palette_plot(im, k):
    """Create a palette from an image and plot clusters in 3D """
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

def plot_3d(data, labels, centers_hex):
    """Plot clustered data in 3D """
    l, a, b = np.transpose(data)
    fig = go.Figure(data=[go.Scatter3d(x=a, y=b, z=l, mode='markers', marker={'size': 3, 'color': [(lambda hex: '#' + hex)(x) for x in centers_hex][labels], 'opacity': 0.1})])
    fig.update_layout(scene={'xaxis_title': 'a', 'yaxis_title': 'b', 'zaxis_title': 'L'})
    fig.show()

def get_image_from_url(url):
    """Download and create image"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content))