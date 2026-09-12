import os
import numpy as np
import numba
from typing import List
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def nearestneighborsfeats__get_features_for_one(neighs_dist: np.ndarray, neighs_y: np.ndarray, k_list: np.ndarray, n_classes: int, eps: float) -> np.ndarray:
    num_feats = len(k_list) * n_classes + 1 + n_classes + n_classes + len(k_list) * 2 + len(k_list) * n_classes
    feats = np.zeros(num_feats, dtype=np.float64)
    idx = 0
    for i in range(len(k_list)):
        k = k_list[i]
        bincount = np.zeros(n_classes, dtype=np.float64)
        for j in range(k):
            bincount[neighs_y[j]] += 1
        for j in range(n_classes):
            feats[idx] = bincount[j] / k
            idx += 1
    streak = 1
    for i in range(1, len(neighs_y)):
        if neighs_y[i] == neighs_y[0]:
            streak += 1
        else:
            break
    feats[idx] = streak
    idx += 1
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c and neighs_dist[i] < min_dist:
                min_dist = neighs_dist[i]
        feats[idx] = min_dist
        idx += 1
    closest_dist = neighs_dist[0]
    for c in range(n_classes):
        min_norm_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                norm_dist = neighs_dist[i] / (closest_dist + eps)
                if norm_dist < min_norm_dist:
                    min_norm_dist = norm_dist
        feats[idx] = min_norm_dist
        idx += 1
    for i in range(len(k_list)):
        k = k_list[i]
        feats[idx] = neighs_dist[k - 1]
        idx += 1
        feats[idx] = neighs_dist[k - 1] / (neighs_dist[0] + eps)
        idx += 1
    for i in range(len(k_list)):
        k = k_list[i]
        for c in range(n_classes):
            sum_dist = 0.0
            count = 0.0
            for j in range(k):
                if neighs_y[j] == c:
                    sum_dist += neighs_dist[j]
                    count += 1.0
            if count > 0:
                feats[idx] = sum_dist / (count + eps)
            else:
                feats[idx] = 999.0
            idx += 1
    return feats

@numba.njit
def _get_test_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    return _deep_to_list(_get_test_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))