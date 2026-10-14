import numba
import numpy as np
from numba import njit
import os
from typing import List
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@njit
def _nearestneighborsfeats__get_features_for_one_core(neighs: np.ndarray, neighs_dist: np.ndarray, eps: float, k_list: np.ndarray, n_classes: int, y_train: np.ndarray) -> np.ndarray:
    neighs_y = np.zeros(len(neighs), dtype=np.int64)
    for i in range(len(neighs)):
        neighs_y[i] = y_train[neighs[i]]
    num_k = len(k_list)
    total_feats = num_k * n_classes + 1 + n_classes + n_classes + num_k * 2 + num_k * n_classes
    out = np.zeros(total_feats, dtype=np.float64)
    idx = 0
    for k_idx in range(num_k):
        k = k_list[k_idx]
        bincount = np.zeros(n_classes, dtype=np.float64)
        for i in range(k):
            if neighs_y[i] < n_classes:
                bincount[neighs_y[i]] += 1.0
        for c in range(n_classes):
            out[idx] = bincount[c] / k
            idx += 1
    streak = 0
    first_label = neighs_y[0]
    for i in range(len(neighs_y)):
        if neighs_y[i] == first_label:
            streak = i + 1
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
    for i in range(1, len(neighs_dist)):
        if neighs_dist[i] < closest_dist:
            closest_dist = neighs_dist[i]
    for c in range(n_classes):
        min_norm_dist = 999.0
        found = False
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                norm_d = neighs_dist[i] / (closest_dist + eps)
                if not found or norm_d < min_norm_dist:
                    min_norm_dist = norm_d
                    found = True
        out[idx] = min_norm_dist
        idx += 1
    for k_idx in range(num_k):
        k = k_list[k_idx]
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        out[idx] = feat_51
        idx += 1
        out[idx] = feat_52
        idx += 1
    for k_idx in range(num_k):
        k = k_list[k_idx]
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

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = np.asarray(NN_output[1][0], dtype=np.int64)
    neighs_dist = np.asarray(NN_output[0][0], dtype=np.float64)
    k_list_arr = np.asarray(k_list, dtype=np.int64)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    return _nearestneighborsfeats__get_features_for_one_core(neighs, neighs_dist, float(eps), k_list_arr, int(n_classes), y_train_arr)

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass