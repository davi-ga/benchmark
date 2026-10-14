import numba
import numpy as np
import os
from typing import List
from numba import njit
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@njit
def _compute_knn_features(neighs: np.ndarray, neighs_dist: np.ndarray, neighs_y: np.ndarray, eps: float, k_list: np.ndarray, n_classes: int) -> np.ndarray:
    n_k = len(k_list)
    total_feats = n_k * n_classes + 1 + n_classes + n_classes + n_k * 2 + n_k * n_classes
    out = np.zeros(total_feats, dtype=np.float64)
    idx = 0
    for i in range(n_k):
        k = k_list[i]
        bincount = np.zeros(n_classes, dtype=np.float64)
        for j in range(min(k, len(neighs_y))):
            val = neighs_y[j]
            if 0 <= val < n_classes:
                bincount[val] += 1.0
        for j in range(n_classes):
            out[idx] = bincount[j] / k
            idx += 1
    streak = 1.0
    for i in range(1, len(neighs_y)):
        if neighs_y[i] == neighs_y[0]:
            streak += 1.0
        else:
            break
    out[idx] = streak
    idx += 1
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                min_dist = min(min_dist, neighs_dist[i])
        out[idx] = min_dist
        idx += 1
    closest_dist = neighs_dist[0] if len(neighs_dist) > 0 else 0.0
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                norm_val = neighs_dist[i] / (closest_dist + eps)
                min_dist = min(min_dist, norm_val)
        out[idx] = min_dist
        idx += 1
    for i in range(n_k):
        k = k_list[i]
        if k - 1 < len(neighs_dist):
            feat_51 = neighs_dist[k - 1]
        else:
            feat_51 = 999.0
        feat_52 = feat_51 / (closest_dist + eps)
        out[idx] = feat_51
        idx += 1
        out[idx] = feat_52
        idx += 1
    for i in range(n_k):
        k = k_list[i]
        for c in range(n_classes):
            c_sum = 0.0
            c_count = 0.0
            for j in range(min(k, len(neighs_y))):
                if neighs_y[j] == c:
                    c_sum += neighs_dist[j]
                    c_count += 1.0
            if c_count > 0.0:
                out[idx] = c_sum / (c_count + eps)
            else:
                out[idx] = 999.0
            idx += 1
    return out

def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    neighs_y = y_train[neighs]
    k_list_arr = k_list
    knn_feats = _compute_knn_features(neighs, neighs_dist, neighs_y, float(eps), k_list_arr, int(n_classes))
    return knn_feats

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass