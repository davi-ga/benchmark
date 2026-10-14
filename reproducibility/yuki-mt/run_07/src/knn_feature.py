import numpy as np
import os
from typing import List
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def nearestneighborsfeats__get_features_for_one_core(neighs_dist, neighs_y, eps, k_list, n_classes):
    num_k = len(k_list)
    num_features = num_k * n_classes + 1 + n_classes + n_classes + num_k * 2 + num_k * n_classes
    knn_feats = np.zeros(num_features, dtype=np.float64)
    feat_idx = 0
    for k_idx in range(num_k):
        k = k_list[k_idx]
        bincount = np.zeros(n_classes, dtype=np.float64)
        for i in range(k):
            if i < len(neighs_y):
                bincount[neighs_y[i]] += 1.0
        for i in range(n_classes):
            knn_feats[feat_idx] = bincount[i] / k
            feat_idx += 1
    streak = 0
    if len(neighs_y) > 0:
        first_label = neighs_y[0]
        for i in range(len(neighs_y)):
            if neighs_y[i] == first_label:
                streak = i + 1
            else:
                break
    else:
        streak = 1
    knn_feats[feat_idx] = streak
    feat_idx += 1
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_dist:
                    min_dist = neighs_dist[i]
        knn_feats[feat_idx] = min_dist
        feat_idx += 1
    closest_dist = 999.0
    if len(neighs_dist) > 0:
        closest_dist = neighs_dist[0]
        for i in range(len(neighs_dist)):
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
        knn_feats[feat_idx] = min_norm_dist
        feat_idx += 1
    d0 = neighs_dist[0] if len(neighs_dist) > 0 else 0.0
    for k_idx in range(num_k):
        k = k_list[k_idx]
        feat_51 = neighs_dist[k - 1] if k - 1 < len(neighs_dist) else 999.0
        feat_52 = feat_51 / (d0 + eps)
        knn_feats[feat_idx] = feat_51
        feat_idx += 1
        knn_feats[feat_idx] = feat_52
        feat_idx += 1
    for k_idx in range(num_k):
        k = k_list[k_idx]
        for c in range(n_classes):
            dist_sum = 0.0
            count = 0
            for i in range(k):
                if i < len(neighs_y) and neighs_y[i] == c:
                    dist_sum += neighs_dist[i]
                    count += 1
            if count > 0:
                knn_feats[feat_idx] = dist_sum / (count + eps)
            else:
                knn_feats[feat_idx] = 999.0
            feat_idx += 1
    return knn_feats

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    neighs_y = y_train[neighs]
    k_list_arr = k_list
    return nearestneighborsfeats__get_features_for_one_core(neighs_dist, neighs_y, eps, k_list_arr, n_classes)

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass