import numpy as np
import os
from typing import List
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def nearestneighborsfeats__get_features_for_one_compiled(neighs_dist: np.ndarray, neighs_y: np.ndarray, eps: float, k_list: np.ndarray, n_classes: int) -> np.ndarray:
    k_list_len = len(k_list)
    num_feats = k_list_len * n_classes + 1 + n_classes + n_classes + k_list_len * 2 + k_list_len * n_classes
    knn_feats = np.zeros(num_feats, dtype=np.float64)
    feat_idx = 0
    for k in k_list:
        bincount = np.zeros(n_classes, dtype=np.float64)
        for i in range(k):
            if neighs_y[i] < n_classes:
                bincount[neighs_y[i]] += 1.0
        for i in range(n_classes):
            knn_feats[feat_idx] = bincount[i] / k
            feat_idx += 1
    first_y = neighs_y[0]
    streak = 0
    for i in range(len(neighs_y)):
        if neighs_y[i] == first_y:
            streak += 1
        else:
            break
    knn_feats[feat_idx] = float(streak)
    feat_idx += 1
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_dist:
                    min_dist = neighs_dist[i]
        knn_feats[feat_idx] = min_dist
        feat_idx += 1
    closest_dist = neighs_dist[0]
    for i in range(1, len(neighs_dist)):
        if neighs_dist[i] < closest_dist:
            closest_dist = neighs_dist[i]
    for c in range(n_classes):
        min_norm_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                norm_d = neighs_dist[i] / (closest_dist + eps)
                if norm_d < min_norm_dist:
                    min_norm_dist = norm_d
        knn_feats[feat_idx] = min_norm_dist
        feat_idx += 1
    for k in k_list:
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        knn_feats[feat_idx] = feat_51
        feat_idx += 1
        knn_feats[feat_idx] = feat_52
        feat_idx += 1
    for k in k_list:
        for c in range(n_classes):
            sum_dist = 0.0
            count = 0.0
            for i in range(k):
                if neighs_y[i] == c:
                    sum_dist += neighs_dist[i]
                    count += 1.0
            if count > 0.0:
                knn_feats[feat_idx] = sum_dist / (count + eps)
            else:
                knn_feats[feat_idx] = 999.0
            feat_idx += 1
    return knn_feats

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    neighs_y = y_train[neighs]
    return nearestneighborsfeats__get_features_for_one_compiled(neighs_dist, neighs_y, float(eps), k_list, int(n_classes))

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass