import numba
import os
import numpy as np
from numba import njit
from typing import List
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@njit
def _compute_knn_features_for_one(neighs: np.ndarray, neighs_dist: np.ndarray, eps: float, k_list: np.ndarray, n_classes: int, y_train: np.ndarray) -> np.ndarray:
    neighs_y = np.zeros(neighs.shape[0], dtype=np.int64)
    for i in range(neighs.shape[0]):
        neighs_y[i] = y_train[neighs[i]]
    num_k = k_list.shape[0]
    total_feats = num_k * n_classes + 1 + n_classes + n_classes + num_k * 2 + num_k * n_classes
    out = np.zeros(total_feats, dtype=np.float64)
    idx = 0
    for k_idx in range(num_k):
        k = k_list[k_idx]
        bincount = np.zeros(n_classes, dtype=np.float64)
        for i in range(k):
            c = neighs_y[i]
            if c < n_classes:
                bincount[c] += 1.0
        for c in range(n_classes):
            out[idx] = bincount[c] / float(k)
            idx += 1
    streak = 1.0
    for i in range(1, neighs_y.shape[0]):
        if neighs_y[i] == neighs_y[0]:
            streak += 1.0
        else:
            break
    out[idx] = streak
    idx += 1
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(neighs.shape[0]):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_dist:
                    min_dist = neighs_dist[i]
        out[idx] = min_dist
        idx += 1
    closest_dist = neighs_dist[0]
    for i in range(1, neighs.shape[0]):
        if neighs_dist[i] < closest_dist:
            closest_dist = neighs_dist[i]
    for c in range(n_classes):
        min_norm_dist = 999.0
        found = False
        for i in range(neighs.shape[0]):
            if neighs_y[i] == c:
                nd = neighs_dist[i] / (closest_dist + eps)
                if not found or nd < min_norm_dist:
                    min_norm_dist = nd
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
            c_sum = 0.0
            c_count = 0.0
            for i in range(k):
                if neighs_y[i] == c:
                    c_sum += neighs_dist[i]
                    c_count += 1.0
            if c_count > 0.0:
                out[idx] = c_sum / (c_count + eps)
            else:
                out[idx] = 999.0
            idx += 1
    return out

def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    k_list_arr = k_list
    return _compute_knn_features_for_one(neighs, neighs_dist, float(eps), k_list_arr, int(n_classes), y_train)

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        try:
            NNF = NearestNeighborsFeats(n_jobs=4, k_list=k_list, metric=metric)
            NNF.fit(X, y)
            test_knn_feats = NNF.predict(X)
            np.save(f'data/knn_feats_{metric}_test.npy', test_knn_feats)
        except NameError:
            pass