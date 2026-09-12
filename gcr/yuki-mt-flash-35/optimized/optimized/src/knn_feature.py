import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
from typing import List
from sklearn.model_selection import StratifiedKFold
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def compute_knn_features_numba(neighs_dist_all: np.ndarray, neighs_y_all: np.ndarray, k_list: np.ndarray, n_classes: int, eps: float) -> np.ndarray:
    n_samples = neighs_dist_all.shape[0]
    n_k = len(k_list)
    n_feats = n_k * (2 * n_classes + 2) + 2 * n_classes + 1
    out = np.zeros((n_samples, n_feats), dtype=np.float64)
    for i in range(n_samples):
        neighs_dist = neighs_dist_all[i]
        neighs_y = neighs_y_all[i]
        idx = 0
        for k_idx in range(n_k):
            k = k_list[k_idx]
            counts = np.zeros(n_classes, dtype=np.float64)
            for j in range(k):
                val = int(neighs_y[j])
                if 0 <= val < n_classes:
                    counts[val] += 1.0
            for c in range(n_classes):
                out[i, idx] = counts[c] / k
                idx += 1
        streak = 1
        first_label = neighs_y[0]
        for j in range(1, len(neighs_y)):
            if neighs_y[j] == first_label:
                streak += 1
            else:
                break
        out[i, idx] = float(streak)
        idx += 1
        for c in range(n_classes):
            min_dist = 999.0
            for j in range(len(neighs_y)):
                if int(neighs_y[j]) == c:
                    min_dist = neighs_dist[j]
                    break
            out[i, idx] = min_dist
            idx += 1
        closest_dist = neighs_dist[0]
        for c in range(n_classes):
            min_norm_dist = 999.0
            for j in range(len(neighs_y)):
                if int(neighs_y[j]) == c:
                    min_norm_dist = neighs_dist[j] / (closest_dist + eps)
                    break
            out[i, idx] = min_norm_dist
            idx += 1
        for k_idx in range(n_k):
            k = k_list[k_idx]
            feat_51 = neighs_dist[k - 1]
            feat_52 = feat_51 / (neighs_dist[0] + eps)
            out[i, idx] = feat_51
            out[i, idx + 1] = feat_52
            idx += 2
        for k_idx in range(n_k):
            k = k_list[k_idx]
            for c in range(n_classes):
                sum_dist = 0.0
                count_c = 0
                for j in range(k):
                    if int(neighs_y[j]) == c:
                        sum_dist += neighs_dist[j]
                        count_c += 1
                if count_c > 0:
                    out[i, idx] = sum_dist / (float(count_c) + eps)
                else:
                    out[i, idx] = 999.0
                idx += 1
    return out

@numba.njit
def nearestneighborsfeats_fit(self_list: list, X: np.ndarray, y: np.ndarray):
    k_list = self_list[0]
    metric = self_list[1]
    n_jobs = self_list[2]
    max_k = max(k_list)
    algorithm = 'brute' if metric == 'cosine' else 'auto'
    NN = NearestNeighbors(n_neighbors=max_k, metric=metric, n_jobs=n_jobs, algorithm=algorithm)
    NN.fit(X)
    self_list[4] = NN
    self_list[5] = y
    self_list[6] = len(np.unique(y))

@numba.njit
def nearestneighborsfeats_predict(self_list: list, X: np.ndarray) -> np.ndarray:
    NN = self_list[4]
    y_train = self_list[5]
    n_classes = self_list[6]
    k_list = self_list[0]
    eps = self_list[3]
    neighs_dist, neighs = NN.kneighbors(X)
    neighs_y = y_train[neighs]
    k_list_arr = np.array(k_list, dtype=np.int64)
    return compute_knn_features_numba(neighs_dist, neighs_y, k_list_arr, n_classes, eps)

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(self_list: list, x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    res = nearestneighborsfeats_predict(self_list, x)
    return res[0]

def nearestneighborsfeats__get_features_for_one(self_list: list, x: np.ndarray):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(self_list), _sanitize_for_numba(x)))

@numba.njit
def _get_test_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        state = [k_list, metric, 4, 1e-05, None, None, 2]
        nearestneighborsfeats_fit(state, X, y)
        test_knn_feats = nearestneighborsfeats_predict(state, X)
        np.save(f'data/knn_feats_{metric}_test.npy', test_knn_feats)

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    return _deep_to_list(_get_test_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))

@numba.njit
def _get_train_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    n_classes = len(np.unique(y))
    n_feats = len(k_list) * (2 * n_classes + 2) + 2 * n_classes + 1
    for metric in ['minkowski', 'cosine']:
        skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        preds = np.zeros((X.shape[0], n_feats), dtype=np.float64)
        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = (X[train_idx], y[train_idx])
            X_val = X[val_idx]
            state = [k_list, metric, 4, 1e-05, None, None, n_classes]
            nearestneighborsfeats_fit(state, X_train, y_train)
            preds[val_idx] = nearestneighborsfeats_predict(state, X_val)
        np.save(f'data/knn_feats_{metric}_train.npy', preds)

def get_train_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    return _deep_to_list(_get_train_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))