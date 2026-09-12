import numpy as np
import os
from typing import List, Tuple
import numba
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _compute_all_features(dists: np.ndarray, indices: np.ndarray, y_train: np.ndarray, k_list: np.ndarray, n_classes: int, eps: float) -> np.ndarray:
    n_samples = dists.shape[0]
    num_k = len(k_list)
    total_feats = num_k * n_classes + 1 + n_classes + n_classes + 2 * num_k + num_k * n_classes
    out = np.zeros((n_samples, total_feats), dtype=np.float64)
    for i in range(n_samples):
        sample_dists = dists[i]
        sample_indices = indices[i]
        n_neighs = len(sample_indices)
        sample_y = np.zeros(n_neighs, dtype=np.int64)
        for j in range(n_neighs):
            sample_y[j] = y_train[sample_indices[j]]
        idx = 0
        for ki in range(num_k):
            k = k_list[ki]
            counts = np.zeros(n_classes, dtype=np.float64)
            for j in range(k):
                c = sample_y[j]
                if c < n_classes:
                    counts[c] += 1.0
            for c in range(n_classes):
                out[i, idx] = counts[c] / float(k)
                idx += 1
        streak = 0
        first_label = sample_y[0]
        for j in range(n_neighs):
            if sample_y[j] == first_label:
                streak += 1
            else:
                break
        out[i, idx] = float(streak)
        idx += 1
        for c in range(n_classes):
            min_d = 999.0
            for j in range(n_neighs):
                if sample_y[j] == c:
                    if sample_dists[j] < min_d:
                        min_d = sample_dists[j]
            out[i, idx] = min_d
            idx += 1
        closest_dist = sample_dists[0]
        for j in range(1, n_neighs):
            if sample_dists[j] < closest_dist:
                closest_dist = sample_dists[j]
        for c in range(n_classes):
            min_norm_d = 999.0
            for j in range(n_neighs):
                if sample_y[j] == c:
                    nd = sample_dists[j] / (closest_dist + eps)
                    if nd < min_norm_d:
                        min_norm_d = nd
            out[i, idx] = min_norm_d
            idx += 1
        for ki in range(num_k):
            k = k_list[ki]
            feat_51 = sample_dists[k - 1]
            feat_52 = feat_51 / (sample_dists[0] + eps)
            out[i, idx] = feat_51
            idx += 1
            out[i, idx] = feat_52
            idx += 1
        for ki in range(num_k):
            k = k_list[ki]
            for c in range(n_classes):
                sum_d = 0.0
                count_c = 0
                for j in range(k):
                    if sample_y[j] == c:
                        sum_d += sample_dists[j]
                        count_c += 1
                if count_c > 0:
                    out[i, idx] = sum_d / (float(count_c) + eps)
                else:
                    out[i, idx] = 999.0
                idx += 1
    return out

def nearestneighborsfeats_fit(X: np.ndarray, y: np.ndarray, k_list: np.ndarray=None, metric: float='euclidean', n_classes: int=None, eps: float=1e-05) -> Tuple:
    if k_list is None:
        k_list = np.array([1, 3, 5, 7, 9], dtype=np.int64)
    elif isinstance(k_list, list):
        k_list = np.array(k_list, dtype=np.int64)
    max_k = int(np.max(k_list))
    algo = 'brute' if metric == 'cosine' else 'auto'
    nn_model = NearestNeighbors(n_neighbors=max_k, metric=metric, n_jobs=1, algorithm=algo)
    nn_model.fit(X)
    if n_classes is None:
        num_classes = int(np.unique(y).shape[0])
    else:
        num_classes = int(n_classes)
    return (nn_model, np.asarray(y, dtype=np.int64), k_list, num_classes, float(eps))

@numba.njit
def nearestneighborsfeats_predict(state: Tuple, X: np.ndarray) -> np.ndarray:
    nn_model, y_train, k_list, n_classes, eps = state
    dists, indices = nn_model.kneighbors(X)
    return _compute_all_features(dists, indices, y_train, k_list, n_classes, eps)

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(state: Tuple, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return nearestneighborsfeats_predict(state, x)[0]

def nearestneighborsfeats__get_features_for_one(state: Tuple, x: np.ndarray):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(state), _sanitize_for_numba(x)))

@numba.njit
def _get_test_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]) -> None:
    os.makedirs('data', exist_ok=True)
    k_arr = np.array(k_list, dtype=np.int64)
    for metric in ['minkowski', 'cosine']:
        state = nearestneighborsfeats_fit(X, y, k_list=k_arr, metric=metric)
        test_knn_feats = nearestneighborsfeats_predict(state, X)
        np.save(f'data/knn_feats_{metric}_test.npy', test_knn_feats)

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    return _deep_to_list(_get_test_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))

@numba.njit
def _get_train_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]) -> None:
    k_arr = np.array(k_list, dtype=np.int64)
    for metric in ['minkowski', 'cosine']:
        skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        dummy_state = nearestneighborsfeats_fit(X[:10], y[:10], k_list=k_arr, metric=metric)
        dummy_pred = nearestneighborsfeats_predict(dummy_state, X[:1])
        n_feats = dummy_pred.shape[1]
        preds = np.zeros((X.shape[0], n_feats), dtype=np.float64)
        for train_idx, test_idx in skf.split(X, y):
            X_tr, y_tr = (X[train_idx], y[train_idx])
            X_te = X[test_idx]
            state = nearestneighborsfeats_fit(X_tr, y_tr, k_list=k_arr, metric=metric)
            preds[test_idx] = nearestneighborsfeats_predict(state, X_te)
        np.save(f'data/knn_feats_{metric}_train.npy', preds)

def get_train_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    return _deep_to_list(_get_train_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))