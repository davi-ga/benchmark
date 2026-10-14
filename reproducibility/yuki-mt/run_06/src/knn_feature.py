import numpy as np
import os
from typing import List
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def nearestneighborsfeats__get_features_for_one_compiled(neighs, neighs_dist, neighs_y, eps, k_list, n_classes):
    total_feats = len(k_list) * n_classes + 1 + n_classes + n_classes + len(k_list) * 2 + len(k_list) * n_classes
    out = np.zeros(total_feats, dtype=np.float64)
    idx = 0
    for k in k_list:
        for i in range(n_classes):
            count = 0
            for j in range(k):
                if neighs_y[j] == i:
                    count += 1
            out[idx] = count / k
            idx += 1
    same_streak = 0
    first_lbl = neighs_y[0]
    for i in range(len(neighs_y)):
        if neighs_y[i] == first_lbl:
            same_streak += 1
        else:
            break
    out[idx] = same_streak
    idx += 1
    for c in range(n_classes):
        min_d = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_d:
                    min_d = neighs_dist[i]
        out[idx] = min_d
        idx += 1
    closest_dist = neighs_dist[0]
    for i in range(len(neighs_dist)):
        if neighs_dist[i] < closest_dist:
            closest_dist = neighs_dist[i]
    for c in range(n_classes):
        min_d = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                nd = neighs_dist[i] / (closest_dist + eps)
                if nd < min_d:
                    min_d = nd
        out[idx] = min_d
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
            s = 0.0
            count = 0
            for i in range(k):
                if neighs_y[i] == c:
                    s += neighs_dist[i]
                    count += 1
            if count > 0:
                out[idx] = s / (count + eps)
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
    return nearestneighborsfeats__get_features_for_one_compiled(neighs, neighs_dist, neighs_y, float(eps), k_list_arr, int(n_classes))

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass