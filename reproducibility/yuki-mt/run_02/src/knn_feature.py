import numpy as np
import os
from typing import List
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _nearestneighborsfeats__get_features_for_one_core(neighs_dist, neighs, eps, k_list, n_classes, y_train):
    neighs_y = np.zeros(neighs.shape[1], dtype=np.int64)
    for i in range(neighs.shape[1]):
        neighs_y[i] = y_train[int(neighs[0, i])]
    neighs_dist_1d = neighs_dist[0]
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
    res = np.zeros(out_len, dtype=np.float64)
    idx = 0
    for k_val in k_list:
        bincount = np.zeros(n_classes, dtype=np.int64)
        for i in range(k_val):
            if neighs_y[i] < n_classes:
                bincount[neighs_y[i]] += 1
        for i in range(n_classes):
            res[idx] = bincount[i] / float(k_val)
            idx += 1
    streak = 0
    first_lbl = neighs_y[0]
    for i in range(neighs_y.shape[0]):
        if neighs_y[i] == first_lbl:
            streak += 1
        else:
            break
    res[idx] = streak
    idx += 1
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(neighs_y.shape[0]):
            if neighs_y[i] == c:
                if neighs_dist_1d[i] < min_dist:
                    min_dist = neighs_dist_1d[i]
        res[idx] = min_dist
        idx += 1
    closest_dist = neighs_dist_1d[0]
    for i in range(1, neighs_dist_1d.shape[0]):
        if neighs_dist_1d[i] < closest_dist:
            closest_dist = neighs_dist_1d[i]
    for c in range(n_classes):
        min_norm_dist = 999.0
        found = False
        for i in range(neighs_y.shape[0]):
            if neighs_y[i] == c:
                norm_val = neighs_dist_1d[i] / (closest_dist + eps)
                if not found or norm_val < min_norm_dist:
                    min_norm_dist = norm_val
                    found = True
        if not found:
            res[idx] = 999.0
        else:
            res[idx] = min_norm_dist
        idx += 1
    for k_val in k_list:
        f51 = neighs_dist_1d[k_val - 1]
        f52 = f51 / (neighs_dist_1d[0] + eps)
        res[idx] = f51
        idx += 1
        res[idx] = f52
        idx += 1
    for k_val in k_list:
        for c in range(n_classes):
            s = 0.0
            cnt = 0
            for i in range(k_val):
                if neighs_y[i] == c:
                    s += neighs_dist_1d[i]
                    cnt += 1
            if cnt > 0:
                res[idx] = s / (cnt + eps)
            else:
                res[idx] = 999.0
            idx += 1
    return res

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    k_arr = k_list
    return _nearestneighborsfeats__get_features_for_one_core(NN_output[0], NN_output[1], eps, k_arr, n_classes, y_train)

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass