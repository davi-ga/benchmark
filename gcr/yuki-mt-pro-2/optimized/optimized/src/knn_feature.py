import numpy as np
import numba
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _compute_all_features(X, neighs, neighs_dist, y_train, k_list, n_classes, eps):
    n_samples = X.shape[0]
    n_neighbors = neighs.shape[1]
    n_ks = k_list.shape[0]
    n_feats = n_ks * n_classes + 1 + n_classes + n_classes + n_ks * 2 + n_ks * n_classes
    out = np.zeros((n_samples, n_feats), dtype=np.float64)
    for i in range(n_samples):
        n_y = np.zeros(n_neighbors, dtype=np.int64)
        for j in range(n_neighbors):
            n_y[j] = y_train[neighs[i, j]]
        dists = neighs_dist[i]
        idx = 0
        for ki in range(n_ks):
            k = k_list[ki]
            counts = np.zeros(n_classes, dtype=np.float64)
            for j in range(k):
                c = n_y[j]
                if c < n_classes:
                    counts[c] += 1.0
            for c in range(n_classes):
                out[i, idx] = counts[c] / k
                idx += 1
        streak = 1
        first_label = n_y[0]
        for j in range(1, n_neighbors):
            if n_y[j] == first_label:
                streak += 1
            else:
                break
        out[i, idx] = float(streak)
        idx += 1
        for c in range(n_classes):
            min_d = 999.0
            for j in range(n_neighbors):
                if n_y[j] == c:
                    if dists[j] < min_d:
                        min_d = dists[j]
                    break
            out[i, idx] = min_d
            idx += 1
        closest_d = dists[0]
        for c in range(n_classes):
            min_nd = 999.0
            for j in range(n_neighbors):
                if n_y[j] == c:
                    min_nd = dists[j] / (closest_d + eps)
                    break
            out[i, idx] = min_nd
            idx += 1
        for ki in range(n_ks):
            k = k_list[ki]
            d_k = dists[k - 1]
            out[i, idx] = d_k
            idx += 1
            out[i, idx] = d_k / (closest_d + eps)
            idx += 1
        for ki in range(n_ks):
            k = k_list[ki]
            for c in range(n_classes):
                sum_d = 0.0
                count_c = 0
                for j in range(k):
                    if n_y[j] == c:
                        sum_d += dists[j]
                        count_c += 1
                if count_c > 0:
                    out[i, idx] = sum_d / (count_c + eps)
                else:
                    out[i, idx] = 999.0
                idx += 1
    return out

@numba.njit
def _nearestneighborsfeats_fit_wrapper_jit(X, y, k_list, metric):
    NN = NearestNeighbors(n_neighbors=max(k_list), metric=metric, n_jobs=1, algorithm='brute' if metric == 'cosine' else 'auto')
    NN.fit(X)
    return NN

def nearestneighborsfeats_fit_wrapper(X, y, k_list, metric):
    return _deep_to_list(_nearestneighborsfeats_fit_wrapper_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list), _sanitize_for_numba(metric)))

@numba.njit
def _nearestneighborsfeats_predict_wrapper_jit(X, NN_model, y_train, k_list, n_classes, eps):
    neighs_dist, neighs = NN_model.kneighbors(X)
    return _compute_all_features(X, neighs, neighs_dist, y_train, k_list, n_classes, eps)

def nearestneighborsfeats_predict_wrapper(X, NN_model, y_train, k_list, n_classes, eps):
    return _deep_to_list(_nearestneighborsfeats_predict_wrapper_jit(_sanitize_for_numba(X), _sanitize_for_numba(NN_model), _sanitize_for_numba(y_train), _sanitize_for_numba(k_list), _sanitize_for_numba(n_classes), _sanitize_for_numba(eps)))

class DummyNNF:

    def __init__(self, k_list, metric, n_classes, eps=1e-06):
        self.k_list = np.array(k_list, dtype=np.int64)
        self.metric = metric
        self.n_classes = n_classes
        self.eps = eps

    def fit(self, X, y):
        self.NN = nearestneighborsfeats_fit_wrapper(X, y, self.k_list, self.metric)
        self.y_train = y
        return self

    def predict(self, X):
        return nearestneighborsfeats_predict_wrapper(X, self.NN, self.y_train, self.k_list, self.n_classes, self.eps)

@numba.njit
def _get_test_feature_jit(X: np.ndarray, y: np.ndarray, k_list: list=[3, 8]):
    os.makedirs('data', exist_ok=True)
    n_classes = np.unique(y).shape[0]
    for metric in ['minkowski', 'cosine']:
        NNF = DummyNNF(k_list, metric, n_classes)
        NNF.fit(X, y)
        test_knn_feats = NNF.predict(X)
        np.save(f'data/knn_feats_{metric}_test.npy', test_knn_feats)

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: list=[3, 8]):
    return _deep_to_list(_get_test_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))

@numba.njit
def _get_train_feature_jit(X: np.ndarray, y: np.ndarray, k_list: list=[3, 8]):
    n_classes = np.unique(y).shape[0]
    for metric in ['minkowski', 'cosine']:
        skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        NNF = DummyNNF(k_list, metric, n_classes)
        preds = cross_val_predict(NNF, X, y, cv=skf)
        np.save(f'data/knn_feats_{metric}_train.npy', preds)

def get_train_feature(X: np.ndarray, y: np.ndarray, k_list: list=[3, 8]):
    return _deep_to_list(_get_train_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))