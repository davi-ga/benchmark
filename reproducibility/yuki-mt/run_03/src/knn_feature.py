import numpy as np
import os
from typing import List
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _nearestneighborsfeats__get_features_for_one_compiled(neighs: np.ndarray, neighs_dist: np.ndarray, eps: float, k_list_arr: np.ndarray, n_classes: int, y_train: np.ndarray) -> np.ndarray:
    neighs_y = np.zeros(neighs.shape[0], dtype=np.int64)
    for i in range(neighs.shape[0]):
        neighs_y[i] = y_train[neighs[i]]
    num_k = k_list_arr.shape[0]
    total_feats = num_k * n_classes + 1 + n_classes + n_classes + num_k * 2 + num_k * n_classes
    out_feats = np.zeros(total_feats, dtype=np.float64)
    idx = 0
    for k_idx in range(num_k):
        k = k_list_arr[k_idx]
        bincount = np.zeros(n_classes, dtype=np.float64)
        limit = k if k < neighs_y.shape[0] else neighs_y.shape[0]
        for i in range(limit):
            c = neighs_y[i]
            if c < n_classes:
                bincount[c] += 1.0
        for i in range(n_classes):
            out_feats[idx] = bincount[i] / k
            idx += 1
    streak = 1
    if neighs_y.shape[0] > 0:
        first_lbl = neighs_y[0]
        for i in range(1, neighs_y.shape[0]):
            if neighs_y[i] == first_lbl:
                streak += 1
            else:
                break
    out_feats[idx] = float(streak)
    idx += 1
    for c in range(n_classes):
        min_d = 999.0
        for i in range(neighs_y.shape[0]):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_d:
                    min_d = neighs_dist[i]
                break
        out_feats[idx] = min_d
        idx += 1
    closest_dist = neighs_dist[0]
    for i in range(1, neighs_dist.shape[0]):
        if neighs_dist[i] < closest_dist:
            closest_dist = neighs_dist[i]
    for c in range(n_classes):
        min_norm_d = 999.0
        for i in range(neighs_y.shape[0]):
            if neighs_y[i] == c:
                norm_d = neighs_dist[i] / (closest_dist + eps)
                if norm_d < min_norm_d:
                    min_norm_d = norm_d
        out_feats[idx] = min_norm_d
        idx += 1
    for k_idx in range(num_k):
        k = k_list_arr[k_idx]
        d_k = neighs_dist[k - 1] if k - 1 < neighs_dist.shape[0] else neighs_dist[-1]
        d_0 = neighs_dist[0]
        out_feats[idx] = d_k
        idx += 1
        out_feats[idx] = d_k / (d_0 + eps)
        idx += 1
    for k_idx in range(num_k):
        k = k_list_arr[k_idx]
        for c in range(n_classes):
            sum_d = 0.0
            count_d = 0
            limit = k if k < neighs_y.shape[0] else neighs_y.shape[0]
            for i in range(limit):
                if neighs_y[i] == c:
                    sum_d += neighs_dist[i]
                    count_d += 1
            if count_d > 0:
                out_feats[idx] = sum_d / (count_d + eps)
            else:
                out_feats[idx] = 999.0
            idx += 1
    return out_feats

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    k_list_arr = k_list
    return _nearestneighborsfeats__get_features_for_one_compiled(neighs, neighs_dist, eps, k_list_arr, n_classes, y_train)

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass