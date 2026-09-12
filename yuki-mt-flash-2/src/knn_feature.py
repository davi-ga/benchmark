import os
from typing import List, Tuple
import numpy as np
import numba
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def compute_single_sample_feats(neighs_dist: np.ndarray, neighs_y: np.ndarray, k_list: np.ndarray, n_classes: int, eps: float) -> np.ndarray:
    n_k = len(k_list)
    total_feats = 2 * n_k * n_classes + 2 * n_classes + 2 * n_k + 1
    out = np.zeros(total_feats, dtype=np.float64)
    idx = 0
    for i in range(n_k):
        k = k_list[i]
        for c in range(n_classes):
            cnt = 0
            for j in range(k):
                if neighs_y[j] == c:
                    cnt += 1
            out[idx] = float(cnt) / float(k)
            idx += 1
    streak = 0
    if len(neighs_y) > 0:
        first_label = neighs_y[0]
        for j in range(len(neighs_y)):
            if neighs_y[j] == first_label:
                streak += 1
            else:
                break
    out[idx] = float(streak)
    idx += 1
    for c in range(n_classes):
        min_d = 999.0
        for j in range(len(neighs_y)):
            if neighs_y[j] == c:
                if neighs_dist[j] < min_d:
                    min_d = neighs_dist[j]
        out[idx] = min_d
        idx += 1
    closest_dist = 999.0
    if len(neighs_dist) > 0:
        closest_dist = neighs_dist[0]
        for j in range(1, len(neighs_dist)):
            if neighs_dist[j] < closest_dist:
                closest_dist = neighs_dist[j]
    for c in range(n_classes):
        min_norm_d = 999.0
        for j in range(len(neighs_y)):
            if neighs_y[j] == c:
                norm_d = neighs_dist[j] / (closest_dist + eps)
                if norm_d < min_norm_d:
                    min_norm_d = norm_d
        out[idx] = min_norm_d
        idx += 1
    for i in range(n_k):
        k = k_list[i]
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        out[idx] = feat_51
        idx += 1
        out[idx] = feat_52
        idx += 1
    for i in range(n_k):
        k = k_list[i]
        for c in range(n_classes):
            dist_sum = 0.0
            cnt = 0
            for j in range(k):
                if neighs_y[j] == c:
                    dist_sum += neighs_dist[j]
                    cnt += 1
            if cnt > 0:
                out[idx] = dist_sum / (float(cnt) + eps)
            else:
                out[idx] = 999.0
            idx += 1
    return out

@numba.njit
def compute_all_features(neighs_dist_all: np.ndarray, neighs_y_all: np.ndarray, k_list: np.ndarray, n_classes: int, eps: float) -> np.ndarray:
    n_samples = neighs_dist_all.shape[0]
    n_k = len(k_list)
    total_feats = 2 * n_k * n_classes + 2 * n_classes + 2 * n_k + 1
    out = np.zeros((n_samples, total_feats), dtype=np.float64)
    for i in range(n_samples):
        feats = compute_single_sample_feats(neighs_dist_all[i], neighs_y_all[i], k_list, n_classes, eps)
        for j in range(total_feats):
            out[i, j] = feats[j]
    return out

@numba.njit
def _NearestNeighborsFeats_jit(n_jobs: int=1, k_list: List[int]=None, metric: float='euclidean', eps: float=1e-05) -> tuple:
    if k_list is None:
        k_list = [3, 8]
    k_list_arr = np.array(k_list, dtype=np.int64)
    return (None, np.empty(0, dtype=np.int64), 0, k_list_arr, metric, n_jobs, eps)

def NearestNeighborsFeats(n_jobs: int=1, k_list: List[int]=None, metric: float='euclidean', eps: float=1e-05):
    return _deep_to_list(_NearestNeighborsFeats_jit(_sanitize_for_numba(n_jobs), _sanitize_for_numba(k_list), _sanitize_for_numba(metric), _sanitize_for_numba(eps)))

def nearestneighborsfeats_fit(X: np.ndarray, y: np.ndarray, k_list: List[int]=None, metric: float='euclidean', n_jobs: int=1, eps: float=1e-05, n_classes: int=None) -> tuple:
    if k_list is None:
        k_list = [3, 8]
    k_list_arr = np.array(k_list, dtype=np.int64)
    max_k = int(np.max(k_list_arr))
    algorithm = 'brute' if metric == 'cosine' else 'auto'
    NN = NearestNeighbors(n_neighbors=max_k, metric=metric, n_jobs=1, algorithm=algorithm)
    NN.fit(X)
    y_train = np.asarray(y, dtype=np.int64)
    if n_classes is None:
        n_classes = int(len(np.unique(y_train)))
    return (NN, y_train, int(n_classes), k_list_arr, metric, n_jobs, float(eps))

@numba.njit
def nearestneighborsfeats_predict(model: tuple, X: np.ndarray) -> np.ndarray:
    NN, y_train, n_classes, k_list, metric, n_jobs, eps = model
    neighs_dist_all, neighs_indices_all = NN.kneighbors(X)
    n_samples = X.shape[0]
    max_k = neighs_dist_all.shape[1]
    neighs_y_all = np.zeros((n_samples, max_k), dtype=np.int64)
    for i in range(n_samples):
        for j in range(max_k):
            neighs_y_all[i, j] = y_train[neighs_indices_all[i, j]]
    return compute_all_features(neighs_dist_all, neighs_y_all, k_list, n_classes, eps)

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(model: tuple, x: np.ndarray) -> np.ndarray:
    NN, y_train, n_classes, k_list, metric, n_jobs, eps = model
    if x.ndim == 1:
        x = x.reshape(1, -1)
    neighs_dist, neighs_indices = NN.kneighbors(x)
    neighs_dist_1d = neighs_dist[0]
    neighs_indices_1d = neighs_indices[0]
    max_k = len(neighs_indices_1d)
    neighs_y = np.zeros(max_k, dtype=np.int64)
    for j in range(max_k):
        neighs_y[j] = y_train[neighs_indices_1d[j]]
    return compute_single_sample_feats(neighs_dist_1d, neighs_y, k_list, n_classes, eps)

def nearestneighborsfeats__get_features_for_one(model: tuple, x: np.ndarray):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(model), _sanitize_for_numba(x)))

@numba.njit
def _get_test_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=None):
    if k_list is None:
        k_list = [3, 8]
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        model = nearestneighborsfeats_fit(X, y, k_list=k_list, metric=metric, n_jobs=4)
        test_knn_feats = nearestneighborsfeats_predict(model, X)
        np.save(f'data/knn_feats_{metric}_test.npy', test_knn_feats)

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=None):
    return _deep_to_list(_get_test_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))

@numba.njit
def _get_train_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=None):
    if k_list is None:
        k_list = [3, 8]
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        n_classes = len(np.unique(y))
        n_k = len(k_list)
        total_feats = 2 * n_k * n_classes + 2 * n_classes + 2 * n_k + 1
        preds = np.zeros((X.shape[0], total_feats), dtype=np.float64)
        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = (X[train_idx], y[train_idx])
            X_va = X[val_idx]
            model = nearestneighborsfeats_fit(X_tr, y_tr, k_list=k_list, metric=metric, n_jobs=4, n_classes=n_classes)
            preds[val_idx] = nearestneighborsfeats_predict(model, X_va)
        np.save(f'data/knn_feats_{metric}_train.npy', preds)

def get_train_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=None):
    return _deep_to_list(_get_train_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))