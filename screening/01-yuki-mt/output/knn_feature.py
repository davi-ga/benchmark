import numba
import numpy as np
from typing import List
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _compute_knn_features_jit(neighs_dist: np.ndarray, neighs_y: np.ndarray, k_list: np.ndarray, n_classes: int, eps: float=1e-06) -> np.ndarray:
    n_k = len(k_list)
    total_size = n_k * n_classes + 1 + n_classes + n_classes + n_k * 2 + n_k * n_classes
    out = np.zeros(total_size, dtype=np.float64)
    idx = 0
    for ki in range(n_k):
        k = k_list[ki]
        bincount = np.bincount(neighs_y[:k])
        b_len = len(bincount)
        for i in range(n_classes):
            if i < b_len:
                out[idx] = float(bincount[i]) / k
            else:
                out[idx] = 0.0
            idx += 1
    match_idx = -1
    for i in range(len(neighs_y)):
        if neighs_y[i] == neighs_y[0]:
            match_idx = i
            break
    if match_idx != -1:
        out[idx] = float(match_idx + 1)
    else:
        out[idx] = 1.0
    idx += 1
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_dist:
                    min_dist = float(neighs_dist[i])
        out[idx] = min_dist
        idx += 1
    closest_dist = neighs_dist[0]
    norm_dist = np.zeros_like(neighs_dist)
    for i in range(len(neighs_dist)):
        norm_dist[i] = neighs_dist[i] / (closest_dist + eps)
    for c in range(n_classes):
        min_ndist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                if norm_dist[i] < min_ndist:
                    min_ndist = float(norm_dist[i])
        out[idx] = min_ndist
        idx += 1
    for ki in range(n_k):
        k = k_list[ki]
        feat_51 = float(neighs_dist[k - 1])
        feat_52 = float(feat_51 / (neighs_dist[0] + eps))
        out[idx] = feat_51
        out[idx + 1] = feat_52
        idx += 2
    for ki in range(n_k):
        k = k_list[ki]
        for c in range(n_classes):
            c_sum = 0.0
            c_count = 0.0
            for i in range(k):
                if neighs_y[i] == c:
                    c_sum += neighs_dist[i]
                    c_count += 1.0
            if c_count > 0:
                out[idx] = c_sum / (c_count + eps)
            else:
                out[idx] = 999.0
            idx += 1
    return out

def compute_knn_features(neighs_dist: np.ndarray, neighs_y: np.ndarray, k_list: np.ndarray, n_classes: int, eps: float=1e-06):
    return _deep_to_list(_compute_knn_features_jit(_sanitize_for_numba(neighs_dist), _sanitize_for_numba(neighs_y), _sanitize_for_numba(k_list), _sanitize_for_numba(n_classes), _sanitize_for_numba(eps)))

@numba.njit
def _compute_class_fractions_jit(neighs_y: np.ndarray, k: int, n_classes: int) -> np.ndarray:
    out = np.zeros(n_classes, dtype=np.float64)
    bincount = np.bincount(neighs_y[:k])
    for i in range(len(bincount)):
        if i < n_classes:
            out[i] = float(bincount[i]) / k
    return out

def compute_class_fractions(neighs_y: np.ndarray, k: int, n_classes: int):
    return _deep_to_list(_compute_class_fractions_jit(_sanitize_for_numba(neighs_y), _sanitize_for_numba(k), _sanitize_for_numba(n_classes)))

@numba.njit
def _compute_min_distances_jit(neighs_dist: np.ndarray, neighs_y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full(n_classes, 999.0, dtype=np.float64)
    for i in range(len(neighs_y)):
        c = int(neighs_y[i])
        if c < n_classes:
            if neighs_dist[i] < out[c]:
                out[c] = neighs_dist[i]
    return out

def compute_min_distances(neighs_dist: np.ndarray, neighs_y: np.ndarray, n_classes: int):
    return _deep_to_list(_compute_min_distances_jit(_sanitize_for_numba(neighs_dist), _sanitize_for_numba(neighs_y), _sanitize_for_numba(n_classes)))

@numba.njit
def _compute_normalized_distances_jit(neighs_dist: np.ndarray, neighs_y: np.ndarray, n_classes: int, eps: float=1e-06) -> np.ndarray:
    out = np.full(n_classes, 999.0, dtype=np.float64)
    closest_dist = neighs_dist[0]
    norm_dist = np.zeros_like(neighs_dist)
    for i in range(len(neighs_dist)):
        norm_dist[i] = neighs_dist[i] / (closest_dist + eps)
    for i in range(len(neighs_y)):
        c = int(neighs_y[i])
        if c < n_classes:
            if norm_dist[i] < out[c]:
                out[c] = norm_dist[i]
    return out

def compute_normalized_distances(neighs_dist: np.ndarray, neighs_y: np.ndarray, n_classes: int, eps: float=1e-06):
    return _deep_to_list(_compute_normalized_distances_jit(_sanitize_for_numba(neighs_dist), _sanitize_for_numba(neighs_y), _sanitize_for_numba(n_classes), _sanitize_for_numba(eps)))

@numba.njit
def _compute_mean_distances_jit(neighs_dist: np.ndarray, neighs_y: np.ndarray, k: int, n_classes: int, eps: float=1e-06) -> np.ndarray:
    c_sum = np.zeros(n_classes, dtype=np.float64)
    c_count = np.zeros(n_classes, dtype=np.float64)
    for i in range(k):
        c = int(neighs_y[i])
        if c < n_classes:
            c_sum[c] += neighs_dist[i]
            c_count[c] += 1.0
    out = np.full(n_classes, 999.0, dtype=np.float64)
    for i in range(n_classes):
        if c_count[i] > 0:
            out[i] = c_sum[i] / (c_count[i] + eps)
    return out

def compute_mean_distances(neighs_dist: np.ndarray, neighs_y: np.ndarray, k: int, n_classes: int, eps: float=1e-06):
    return _deep_to_list(_compute_mean_distances_jit(_sanitize_for_numba(neighs_dist), _sanitize_for_numba(neighs_y), _sanitize_for_numba(k), _sanitize_for_numba(n_classes), _sanitize_for_numba(eps)))