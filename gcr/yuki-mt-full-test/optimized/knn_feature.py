import numba
import numpy as np
from typing import List
import os
from numba import njit
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@njit
def nearestneighborsfeats__get_features_for_one_compiled(neighs, neighs_dist, neighs_y, eps, k_list_arr, n_classes):
    total_features = 0
    for i in range(len(k_list_arr)):
        total_features += n_classes
    total_features += 1
    total_features += n_classes
    total_features += n_classes
    for i in range(len(k_list_arr)):
        total_features += 2
    for i in range(len(k_list_arr)):
        total_features += n_classes
    out = np.zeros(total_features, dtype=np.float64)
    idx = 0
    for k_idx in range(len(k_list_arr)):
        k = k_list_arr[k_idx]
        bincount = np.zeros(n_classes, dtype=np.float64)
        for i in range(k):
            if neighs_y[i] < n_classes:
                bincount[neighs_y[i]] += 1.0
        for c in range(n_classes):
            out[idx] = bincount[c] / k
            idx += 1
    streak = 0
    for i in range(len(neighs_y)):
        if neighs_y[i] == neighs_y[0]:
            streak += 1
        else:
            break
    out[idx] = streak
    idx += 1
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_dist:
                    min_dist = neighs_dist[i]
        out[idx] = min_dist
        idx += 1
    closest_dist = neighs_dist[0]
    for i in range(len(neighs_dist)):
        if neighs_dist[i] < closest_dist:
            closest_dist = neighs_dist[i]
    for c in range(n_classes):
        min_norm_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                d = neighs_dist[i] / (closest_dist + eps)
                if d < min_norm_dist:
                    min_norm_dist = d
        out[idx] = min_norm_dist
        idx += 1
    for k_idx in range(len(k_list_arr)):
        k = k_list_arr[k_idx]
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        out[idx] = feat_51
        idx += 1
        out[idx] = feat_52
        idx += 1
    for k_idx in range(len(k_list_arr)):
        k = k_list_arr[k_idx]
        for c in range(n_classes):
            sum_dist = 0.0
            count = 0
            for i in range(k):
                if neighs_y[i] == c:
                    sum_dist += neighs_dist[i]
                    count += 1
            if count > 0:
                out[idx] = sum_dist / (count + eps)
            else:
                out[idx] = 999.0
            idx += 1
    return out

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    neighs_y = y_train[neighs]
    k_list_arr = k_list
    return nearestneighborsfeats__get_features_for_one_compiled(neighs, neighs_dist, neighs_y, eps, k_list_arr, n_classes)

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

@numba.njit
def _get_test_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    return _deep_to_list(_get_test_feature_jit(_sanitize_for_numba(np.array(X) if isinstance(X, list) else X), _sanitize_for_numba(np.array(y) if isinstance(y, list) else y), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list)))