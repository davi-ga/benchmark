import numpy as np
import os
from typing import List, Tuple
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _nearestneighborsfeats__get_features_for_one_compiled(neighs: np.ndarray, neighs_dist: np.ndarray, eps: float, k_list: np.ndarray, n_classes: int, y_train: np.ndarray):
    neighs_y = np.zeros(len(neighs), dtype=np.int64)
    for i in range(len(neighs)):
        neighs_y[i] = y_train[neighs[i]]
    num_k = len(k_list)
    feat_len = num_k * n_classes + 1 + n_classes + n_classes + num_k * 2 + num_k * n_classes
    out_feats = np.zeros(feat_len, dtype=np.float64)
    out_idx = 0
    for k_idx in range(len(k_list)):
        k = k_list[k_idx]
        bincount = np.zeros(n_classes, dtype=np.float64)
        for i in range(k):
            c = neighs_y[i]
            if c < n_classes:
                bincount[c] += 1.0
        for c in range(n_classes):
            out_feats[out_idx] = bincount[c] / k
            out_idx += 1
    streak = 1
    first_y = neighs_y[0]
    for i in range(1, len(neighs_y)):
        if neighs_y[i] == first_y:
            streak += 1
        else:
            break
    out_feats[out_idx] = streak
    out_idx += 1
    for c in range(n_classes):
        min_d = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_d:
                    min_d = neighs_dist[i]
        out_feats[out_idx] = min_d
        out_idx += 1
    closest_dist = neighs_dist[0]
    for i in range(1, len(neighs_dist)):
        if neighs_dist[i] < closest_dist:
            closest_dist = neighs_dist[i]
    for c in range(n_classes):
        min_d = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                norm_d = neighs_dist[i] / (closest_dist + eps)
                if norm_d < min_d:
                    min_d = norm_d
        out_feats[out_idx] = min_d
        out_idx += 1
    for k_idx in range(len(k_list)):
        k = k_list[k_idx]
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        out_feats[out_idx] = feat_51
        out_idx += 1
        out_feats[out_idx] = feat_52
        out_idx += 1
    for k_idx in range(len(k_list)):
        k = k_list[k_idx]
        for c in range(n_classes):
            c_sum = 0.0
            c_count = 0
            for i in range(k):
                if neighs_y[i] == c:
                    c_sum += neighs_dist[i]
                    c_count += 1
            if c_count > 0:
                out_feats[out_idx] = c_sum / (c_count + eps)
            else:
                out_feats[out_idx] = 999.0
            out_idx += 1
    return out_feats

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    k_arr = k_list
    y_arr = y_train
    return _nearestneighborsfeats__get_features_for_one_compiled(neighs, neighs_dist, float(eps), k_arr, int(n_classes), y_arr)

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass