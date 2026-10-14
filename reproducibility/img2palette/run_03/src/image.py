import numba
import numpy as np
import os
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
sls_stage = os.getenv('SLS_STAGE')
default_k = 4
xyz_ref_white = np.asarray((0.95047, 1.0, 1.08883))
xyz_from_rgb = np.array([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]])
rgb_from_xyz = np.linalg.inv(xyz_from_rgb)

@njit
def _rgb2xyz_compiled(rgb_arr):
    n, c = rgb_arr.shape
    xyz_arr = np.zeros((n, c), dtype=np.float64)
    for i in range(n):
        for j in range(c):
            val = rgb_arr[i, j]
            if val > 0.04045:
                xyz_arr[i, j] = np.power((val + 0.055) / 1.055, 2.4)
            else:
                xyz_arr[i, j] = val / 12.92
    res = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        res[i, 0] = xyz_arr[i, 0] * 0.412453 + xyz_arr[i, 1] * 0.35758 + xyz_arr[i, 2] * 0.180423
        res[i, 1] = xyz_arr[i, 0] * 0.212671 + xyz_arr[i, 1] * 0.71516 + xyz_arr[i, 2] * 0.072169
        res[i, 2] = xyz_arr[i, 0] * 0.019334 + xyz_arr[i, 1] * 0.119193 + xyz_arr[i, 2] * 0.950227
    return res

def _rgb2xyz_jit(rgb_arr):
    return _rgb2xyz_compiled(rgb_arr)

def rgb2xyz(rgb_arr):
    return _deep_to_list(_rgb2xyz_jit(_sanitize_for_numba(np.array(rgb_arr) if isinstance(rgb_arr, list) else rgb_arr)))

@njit
def _xyz2lab_compiled(xyz_arr):
    n, c = xyz_arr.shape
    lab_arr = np.zeros((n, c), dtype=np.float64)
    ref_x, ref_y, ref_z = (xyz_ref_white[0], xyz_ref_white[1], xyz_ref_white[2])
    for i in range(n):
        val_x = xyz_arr[i, 0] / ref_x
        val_y = xyz_arr[i, 1] / ref_y
        val_z = xyz_arr[i, 2] / ref_z
        if val_x > 0.008856:
            fx = np.power(val_x, 1.0 / 3.0)
        else:
            fx = 7.787 * val_x + 16.0 / 116.0
        if val_y > 0.008856:
            fy = np.power(val_y, 1.0 / 3.0)
        else:
            fy = 7.787 * val_y + 16.0 / 116.0
        if val_z > 0.008856:
            fz = np.power(val_z, 1.0 / 3.0)
        else:
            fz = 7.787 * val_z + 16.0 / 116.0
        lab_arr[i, 0] = 116.0 * fy - 16.0
        lab_arr[i, 1] = 500.0 * (fx - fy)
        lab_arr[i, 2] = 200.0 * (fy - fz)
    return lab_arr

def _xyz2lab_jit(xyz_arr):
    return _xyz2lab_compiled(xyz_arr)

def xyz2lab(xyz_arr):
    return _deep_to_list(_xyz2lab_jit(_sanitize_for_numba(np.array(xyz_arr) if isinstance(xyz_arr, list) else xyz_arr)))

@njit
def _lab2xyz_compiled(lab_arr):
    n = lab_arr.shape[0]
    xyz_arr = np.zeros((n, 3), dtype=np.float64)
    ref_x, ref_y, ref_z = (xyz_ref_white[0], xyz_ref_white[1], xyz_ref_white[2])
    for i in range(n):
        L = lab_arr[i, 0]
        a = lab_arr[i, 1]
        b = lab_arr[i, 2]
        y = (L + 16.0) / 116.0
        x = a / 500.0 + y
        z = y - b / 200.0
        if z < 0.0:
            z = 0.0
        if x > 0.2068966:
            fx = np.power(x, 3.0)
        else:
            fx = (x - 16.0 / 116.0) / 7.787
        if y > 0.2068966:
            fy = np.power(y, 3.0)
        else:
            fy = (y - 16.0 / 116.0) / 7.787
        if z > 0.2068966:
            fz = np.power(z, 3.0)
        else:
            fz = (z - 16.0 / 116.0) / 7.787
        xyz_arr[i, 0] = fx * ref_x
        xyz_arr[i, 1] = fy * ref_y
        xyz_arr[i, 2] = fz * ref_z
    return xyz_arr

def _lab2xyz_jit(lab_arr):
    return _lab2xyz_compiled(lab_arr)

def lab2xyz(lab_arr):
    return _deep_to_list(_lab2xyz_jit(_sanitize_for_numba(np.array(lab_arr) if isinstance(lab_arr, list) else lab_arr)))

@njit
def _xyz2rgb_compiled(xyz_arr):
    n = xyz_arr.shape[0]
    rgb_arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        x, y, z = (xyz_arr[i, 0], xyz_arr[i, 1], xyz_arr[i, 2])
        r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
        g = x * -0.969266 + y * 1.8760108 + z * 0.041556
        b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
        if r > 0.0031308:
            r = 1.055 * np.power(r, 1 / 2.4) - 0.055
        else:
            r *= 12.92
        if g > 0.0031308:
            g = 1.055 * np.power(g, 1 / 2.4) - 0.055
        else:
            g *= 12.92
        if b > 0.0031308:
            b = 1.055 * np.power(b, 1 / 2.4) - 0.055
        else:
            b *= 12.92
        if r < 0.0:
            r = 0.0
        if r > 1.0:
            r = 1.0
        if g < 0.0:
            g = 0.0
        if g > 1.0:
            g = 1.0
        if b < 0.0:
            b = 0.0
        if b > 1.0:
            b = 1.0
        rgb_arr[i, 0] = r
        rgb_arr[i, 1] = g
        rgb_arr[i, 2] = b
    return rgb_arr

def _xyz2rgb_jit(xyz_arr):
    return _xyz2rgb_compiled(xyz_arr)

def xyz2rgb(xyz_arr):
    return _deep_to_list(_xyz2rgb_jit(_sanitize_for_numba(np.array(xyz_arr) if isinstance(xyz_arr, list) else xyz_arr)))

def rgb2lab(rgb_arr):
    return _xyz2lab_compiled(_rgb2xyz_compiled(rgb_arr))

def lab2rgb(lab_arr):
    return _xyz2rgb_compiled(_lab2xyz_compiled(lab_arr))

def get_lab_data(im):
    img_size = (150, 150)
    im.thumbnail(img_size)
    pixel_rgb = np.asarray(im)
    pixel_lab = rgb2lab(pixel_rgb.reshape(-1, pixel_rgb.shape[-1]) / 255.0)
    return pixel_lab.reshape(-1, pixel_lab.shape[-1])

def make_img(colors, counts):
    try:
        from PIL import Image
    except ImportError:
        Image, = (None,)
    img_size = (512, 512)
    n_clusters = len(colors)
    lengths = ((counts / np.sum(counts) + 1.0 / n_clusters) / 2.0 * img_size[0]).astype(np.uint16)
    lengths[0] = lengths[0] + (img_size[0] - np.sum(lengths))
    pixel_group = np.concatenate([np.tile(colors[i], (lengths[i], img_size[1], 1)) for i in range(n_clusters)], axis=0)
    pixel_rgb = np.transpose(pixel_group, (1, 0, 2))
    return Image.fromarray(pixel_rgb, mode='RGB')

def get_hex_string(rgb_arr):

    def int2hex(integer):
        hex_string = _int_to_hex(integer)
        if len(hex_string) < 2:
            return '0' + hex_string
        return hex_string
    return ''.join([int2hex(x) for x in rgb_arr]).upper()

def cluster_kmeans(data, n_clusters):
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        KMeans, = (None,)
    kmeans = KMeans(n_clusters)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return (centers, labels)

def compute_medoid(data):
    try:
        from sklearn.metrics import pairwise_distances
    except ImportError:
        pairwise_distances, = (None,)
    dist_mat = pairwise_distances(data)
    return data[np.argmin(dist_mat.sum(axis=0))]

def cluster_agglo(data, n_clusters):
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        AgglomerativeClustering, = (None,)
    ac = AgglomerativeClustering(n_clusters)
    labels = ac.fit_predict(data)
    centers = np.empty([n_clusters, 3])
    for i in range(n_clusters):
        centers[i] = compute_medoid(data[labels == i])
    return (centers, labels)

def get_cluster(centers, labels):
    _, counts = _np_unique_with_counts(labels)
    sort_idx = (-counts).argsort()
    sort_labels = [(lambda i: list(sort_idx).index(i))(x) for x in labels]
    return (centers[sort_idx], sort_labels, counts[sort_idx])

def get_palette(im, k):
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
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = None
    l, a, b = np.transpose(data)
    fig = go.Figure(data=[go.Scatter3d(x=a, y=b, z=l, mode='markers', marker={'size': 3, 'color': [(lambda hex: '#' + hex)(x) for x in centers_hex][labels], 'opacity': 0.1})])
    fig.update_layout(scene={'xaxis_title': 'a', 'yaxis_title': 'b', 'zaxis_title': 'L'})
    fig.show()

def get_image_from_url(url):
    try:
        import requests
    except ImportError:
        requests = None
    try:
        from io import BytesIO
    except ImportError:
        BytesIO, = (None,)
    try:
        from PIL import Image
    except ImportError:
        Image, = (None,)
    response = requests.get(url)
    return Image.open(BytesIO(response.content))