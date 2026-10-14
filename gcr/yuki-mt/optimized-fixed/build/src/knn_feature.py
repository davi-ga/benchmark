import numba
import numpy as np
from numba import njit
import os
from typing import List
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@njit
def _bincount_custom(arr, minlength):
    res = np.zeros(minlength, dtype=np.int64)
    for i in range(len(arr)):
        if arr[i] >= 0 and arr[i] < minlength:
            res[arr[i]] += 1
    return res

@njit
def _get_nearestneighborsfeats_for_one_compiled(neighs_dist, neighs_y, eps, k_list, n_classes):
    out_len = 0
    for k in k_list:
        out_len += n_classes
    out_len += 1
    out_len += n_classes
    out_len += n_classes
    for k in k_list:
        out_len += 2
    for k in k_list:
        out_len += n_classes
    out = np.zeros(out_len, dtype=np.float64)
    idx = 0
    for k in k_list:
        bc = _bincount_custom(neighs_y[:k], n_classes)
        for i in range(n_classes):
            out[idx] = bc[i] / k
            idx += 1
    val_a = -1
    for i in range(len(neighs_y)):
        if neighs_y[i] == neighs_y[0] and i == val_a + 1:
            val_a = i
        elif neighs_y[i] != neighs_y[0]:
            break
    out[idx] = val_a + 1
    idx += 1
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_dist:
                    min_dist = neighs_dist[i]
        out[idx] = min_dist
        idx += 1
    closest_dist = 999.0
    if len(neighs_dist) > 0:
        closest_dist = neighs_dist[0]
        for i in range(1, len(neighs_dist)):
            if neighs_dist[i] < closest_dist:
                closest_dist = neighs_dist[i]
    for c in range(n_classes):
        min_norm = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                nd = neighs_dist[i] / (closest_dist + eps)
                if nd < min_norm:
                    min_norm = nd
        out[idx] = min_norm
        idx += 1
    for k in k_list:
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        out[idx] = feat_51
        idx += 1
        out[idx] = feat_52
        idx += 1
    for k in k_list:
        for c in range(n_classes):
            sum_dist = 0.0
            count_dist = 0
            for i in range(k):
                if neighs_y[i] == c:
                    sum_dist += neighs_dist[i]
                    count_dist += 1
            if count_dist > 0:
                out[idx] = sum_dist / (count_dist + eps)
            else:
                out[idx] = 999.0
            idx += 1
    return out

def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    neighs_y = y_train[neighs]
    k_list_arr = k_list
    return _get_nearestneighborsfeats_for_one_compiled(neighs_dist, neighs_y, eps, k_list_arr, n_classes)

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass