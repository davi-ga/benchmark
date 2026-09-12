import numba
import numpy as np
from sklearn.neighbors import NearestNeighbors
from numba import njit
from numba.typed import List
import os
from typing import List as PyList
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@njit
def _compute_features_for_one(neighs_dist, neighs_y, k_list_arr, n_classes, eps):
    n_k = len(k_list_arr)
    n_feats_per_k1 = n_classes
    n_feats_per_k2 = 2
    n_feats_per_k3 = n_classes
    total_feats = n_k * n_classes + 1 + n_classes + n_classes + n_k * 2 + n_k * n_classes
    feats = np.zeros(total_feats, dtype=np.float64)
    idx = 0
    for k in k_list_arr:
        bincount = np.zeros(n_classes, dtype=np.float64)
        for i in range(k):
            bincount[neighs_y[i]] += 1.0
        for i in range(n_classes):
            feats[idx] = bincount[i] / k
            idx += 1
    streak = 0
    first_y = neighs_y[0]
    for i in range(len(neighs_y)):
        if neighs_y[i] == first_y:
            streak += 1
        else:
            break
    feats[idx] = streak
    idx += 1
    for c in range(n_classes):
        min_d = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_d:
                    min_d = neighs_dist[i]
        feats[idx] = min_d
        idx += 1
    closest_dist = neighs_dist[0]
    for c in range(n_classes):
        min_norm_d = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                norm_d = neighs_dist[i] / (closest_dist + eps)
                if norm_d < min_norm_d:
                    min_norm_d = norm_d
        feats[idx] = min_norm_d
        idx += 1
    for k in k_list_arr:
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (closest_dist + eps)
        feats[idx] = feat_51
        idx += 1
        feats[idx] = feat_52
        idx += 1
    for k in k_list_arr:
        for c in range(n_classes):
            sum_d = 0.0
            count_d = 0.0
            for i in range(k):
                if neighs_y[i] == c:
                    sum_d += neighs_dist[i]
                    count_d += 1.0
            if count_d > 0.0:
                feats[idx] = sum_d / (count_d + eps)
            else:
                feats[idx] = 999.0
            idx += 1
    return feats

@njit
def compute_all_features(dists, indices, y_train, k_list_arr, n_classes, eps):
    n_samples = dists.shape[0]
    n_k = len(k_list_arr)
    total_feats = n_k * n_classes + 1 + n_classes + n_classes + n_k * 2 + n_k * n_classes
    out = np.zeros((n_samples, total_feats), dtype=np.float64)
    for i in range(n_samples):
        neighs_dist = dists[i]
        neighs_idx = indices[i]
        neighs_y = np.zeros(len(neighs_idx), dtype=np.int64)
        for j in range(len(neighs_idx)):
            neighs_y[j] = y_train[neighs_idx[j]]
        out[i, :] = _compute_features_for_one(neighs_dist, neighs_y, k_list_arr, n_classes, eps)
    return out

@numba.njit
def NearestNeighborsFeats_fit(X, y, k_list, metric='euclidean'):
    max_k = max(k_list)
    NN = NearestNeighbors(n_neighbors=max_k, metric=metric, n_jobs=1, algorithm='brute' if metric == 'cosine' else 'auto')
    NN.fit(X)
    n_classes = np.unique(y).shape[0]
    return (NN, n_classes)

@numba.njit
def NearestNeighborsFeats_predict(X, NN, y_train, k_list, n_classes, eps=1e-06):
    dists, indices = NN.kneighbors(X)
    k_list_arr = np.array(k_list, dtype=np.int64)
    return compute_all_features(dists, indices, y_train, k_list_arr, n_classes, eps)

@numba.njit
def _get_test_feature_jit(X: np.ndarray, y: np.ndarray, k_list: PyList[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    eps = 1e-06
    for metric in ['minkowski', 'cosine']:
        NN, n_classes = NearestNeighborsFeats_fit(X, y, k_list, metric)
        test_knn_feats = NearestNeighborsFeats_predict(X, NN, y, k_list, n_classes, eps)
        np.save(f'data/knn_feats_{metric}_test.npy', test_knn_feats)

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: PyList[int]=[3, 8]):
    return _deep_to_list(_get_test_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))