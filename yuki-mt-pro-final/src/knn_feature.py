import numba
import numpy as np
from typing import List
import os
from numba import njit
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@njit
def _compute_knn_feats(neighs, neighs_dist, neighs_y, eps, k_list, n_classes):
    max_k = k_list[-1]
    feat_len = len(k_list) * n_classes + 1 + n_classes + n_classes + len(k_list) * 2 + len(k_list) * n_classes
    out = np.zeros(feat_len, dtype=np.float64)
    idx = 0
    for i in range(len(k_list)):
        k = k_list[i]
        for c in range(n_classes):
            cnt = 0
            for j in range(k):
                if neighs_y[j] == c:
                    cnt += 1
            out[idx] = cnt / k
            idx += 1
    same_streak = 0
    first_label = neighs_y[0]
    for j in range(len(neighs_y)):
        if neighs_y[j] == first_label:
            same_streak += 1
        else:
            break
    out[idx] = same_streak
    idx += 1
    for c in range(n_classes):
        min_dist = 999.0
        for j in range(len(neighs_y)):
            if neighs_y[j] == c:
                min_dist = neighs_dist[j]
                break
        out[idx] = min_dist
        idx += 1
    closest_dist = neighs_dist[0]
    for c in range(n_classes):
        min_dist = 999.0
        for j in range(len(neighs_y)):
            if neighs_y[j] == c:
                min_dist = neighs_dist[j] / (closest_dist + eps)
                break
        out[idx] = min_dist
        idx += 1
    for i in range(len(k_list)):
        k = k_list[i]
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        out[idx] = feat_51
        idx += 1
        out[idx] = feat_52
        idx += 1
    for i in range(len(k_list)):
        k = k_list[i]
        for c in range(n_classes):
            sum_dist = 0.0
            cnt = 0
            for j in range(k):
                if neighs_y[j] == c:
                    sum_dist += neighs_dist[j]
                    cnt += 1
            if cnt > 0:
                out[idx] = sum_dist / (cnt + eps)
            else:
                out[idx] = 999.0
            idx += 1
    return out

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    neighs_y = y_train[neighs]
    k_arr = k_list
    return _compute_knn_feats(neighs, neighs_dist, neighs_y, eps, k_arr, n_classes)

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

@numba.njit
def _get_test_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    pass

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    return _deep_to_list(_get_test_feature_jit(_sanitize_for_numba(np.array(X) if isinstance(X, list) else X), _sanitize_for_numba(np.array(y) if isinstance(y, list) else y), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list)))