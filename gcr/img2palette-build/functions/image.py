"""Funções puras de conversão de cor e agrupamento.

Recorte NumPy de `image.py` da aplicação original. As funções com PIL,
scikit-learn, requests e plotly permanecem na camada de serviço (`web/app.py`).
As conversões de espaço de cor derivam do pacote scikit-image.
"""

import numpy as np

xyz_ref_white = np.asarray((0.95047, 1.0, 1.08883))

xyz_from_rgb = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)

rgb_from_xyz = np.linalg.inv(xyz_from_rgb)


def rgb2xyz(rgb_arr):
    xyz_arr = np.copy(rgb_arr)
    mask = xyz_arr > 0.04045
    xyz_arr[mask] = np.power((xyz_arr[mask] + 0.055) / 1.055, 2.4)
    xyz_arr[~mask] /= 12.92
    return xyz_arr @ np.transpose(xyz_from_rgb)


def xyz2lab(xyz_arr):
    lab_arr = np.copy(xyz_arr) / xyz_ref_white
    mask = lab_arr > 0.008856
    lab_arr[mask] = np.cbrt(lab_arr[mask])
    lab_arr[~mask] = 7.787 * lab_arr[~mask] + 16.0 / 116.0
    x, y, z = lab_arr[:, 0], lab_arr[:, 1], lab_arr[:, 2]
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    return np.transpose(np.asarray((L, a, b)))


def lab2xyz(lab_arr):
    L, a, b = lab_arr[:, 0], lab_arr[:, 1], lab_arr[:, 2]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)
    if np.any(z < 0):
        invalid = np.nonzero(z < 0)
        z[invalid] = 0
    xyz_arr = np.transpose(np.asarray((x, y, z)))
    mask = xyz_arr > 0.2068966
    xyz_arr[mask] = np.power(xyz_arr[mask], 3.0)
    xyz_arr[~mask] = (xyz_arr[~mask] - 16.0 / 116.0) / 7.787
    xyz_arr *= xyz_ref_white
    return xyz_arr


def xyz2rgb(xyz_arr):
    rgb_arr = xyz_arr @ np.transpose(rgb_from_xyz)
    mask = rgb_arr > 0.0031308
    rgb_arr[mask] = 1.055 * np.power(rgb_arr[mask], 1 / 2.4) - 0.055
    rgb_arr[~mask] *= 12.92
    rgb_arr = np.clip(rgb_arr, 0, 1)
    return rgb_arr


def rgb2lab(rgb_arr):
    return xyz2lab(rgb2xyz(rgb_arr))


def lab2rgb(lab_arr):
    return xyz2rgb(lab2xyz(lab_arr))


def get_hex_string(rgb_arr):
    def int2hex(integer):
        hex_string = hex(integer)[2:]
        if len(hex_string) < 2:
            return "0" + hex_string
        return hex_string

    return "".join(np.vectorize(int2hex)(rgb_arr)).upper()


def get_cluster(centers, labels):
    _, counts = np.unique(labels, return_counts=True)
    sort_idx = (-counts).argsort()
    sort_labels = np.vectorize(lambda i: list(sort_idx).index(i))(labels)
    return centers[sort_idx], sort_labels, counts[sort_idx]
