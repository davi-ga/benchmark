import numba
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
import os
from typing import List
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def compute_fraction_of_classes(neighs_y, k_list, n_classes):
    features = numba.typed.List()
    for k in k_list:
        bincount = np.bincount(neighs_y[:k])
        bincount = np.append(bincount, np.zeros(n_classes - len(bincount)))
        feats = [bincount[i] / k for i in range(n_classes)]
        features.append(feats)
    return features

@numba.njit
def compute_same_label_streak(neighs_y):
    a = np.where(neighs_y == neighs_y[0])[0]
    feats = [a[a == np.arange(len(a))][-1] + 1]
    return [feats]

@numba.njit
def compute_min_distance_to_classes(neighs_y, neighs_dist, n_classes):
    feats = numba.typed.List()
    for c in range(n_classes):
        feats.append(np.min(np.where(neighs_y == c, neighs_dist, 999)))
    return [feats]

@numba.njit
def compute_min_normalized_distance_to_classes(neighs_y, neighs_dist, n_classes, eps):
    feats = numba.typed.List()
    closest_dist = np.min(neighs_dist)
    norm_dist = neighs_dist / (closest_dist + eps)
    for c in range(n_classes):
        dists = norm_dist[neighs_y == c]
        if len(dists) > 0:
            feats.append(dists.min())
        else:
            feats.append(999)
    return [feats]

@numba.njit
def compute_distance_to_kth_neighbor(neighs_dist, k_list, eps):
    features = numba.typed.List()
    for k in k_list:
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        features.append([feat_51, feat_52])
    return features

@numba.njit
def compute_mean_distance_to_classes(neighs_y, neighs_dist, k_list, n_classes, eps):
    features = numba.typed.List()
    for k in k_list:
        feats = numba.typed.List()
        for c in range(n_classes):
            dists = neighs_dist[:k][neighs_y[:k] == c]
            if len(dists) > 0:
                feats.append(dists.sum() / (len(dists) + eps))
            else:
                feats.append(999)
        features.append(feats)
    return features

@numba.njit
def _compute_all_features_jit(neighs_y, neighs_dist, k_list, n_classes, eps):
    return_list = numba.typed.List()
    return_list.extend(compute_fraction_of_classes(neighs_y, k_list, n_classes))
    return_list.extend(compute_same_label_streak(neighs_y))
    return_list.extend(compute_min_distance_to_classes(neighs_y, neighs_dist, n_classes))
    return_list.extend(compute_min_normalized_distance_to_classes(neighs_y, neighs_dist, n_classes, eps))
    return_list.extend(compute_distance_to_kth_neighbor(neighs_dist, k_list, eps))
    return_list.extend(compute_mean_distance_to_classes(neighs_y, neighs_dist, k_list, n_classes, eps))
    return np.hstack(return_list)

def compute_all_features(neighs_y, neighs_dist, k_list, n_classes, eps):
    return _deep_to_list(_compute_all_features_jit(_sanitize_for_numba(neighs_y), _sanitize_for_numba(neighs_dist), _sanitize_for_numba(k_list), _sanitize_for_numba(n_classes), _sanitize_for_numba(eps)))

class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):

    def __init__(self, n_jobs, k_list, metric, n_classes=None, n_neighbors=None, eps=1e-06):
        self.n_jobs = n_jobs
        self.k_list = k_list
        self.metric = metric
        if n_neighbors is None:
            self.n_neighbors = max(k_list)
        else:
            self.n_neighbors = n_neighbors
        self.eps = eps
        self.n_classes_ = n_classes

    def fit(self, X, y):
        self.NN = NearestNeighbors(n_neighbors=max(self.k_list), metric=self.metric, n_jobs=1, algorithm='brute' if self.metric == 'cosine' else 'auto')
        self.NN.fit(X)
        self.y_train = y
        self.n_classes = np.unique(y).shape[0] if self.n_classes_ is None else self.n_classes_

    def predict(self, X):
        if self.n_jobs == 1:
            test_feats = []
            for i in range(X.shape[0]):
                test_feats.append(self._get_features_for_one(X[i:i + 1]))
        else:
            with Pool(processes=self.n_jobs) as pool:
                data = [X[i:i + 1] for i in range(X.shape[0])]
                test_feats = pool.map(self._get_features_for_one, data)
        return np.vstack(test_feats)

    def _get_features_for_one(self, x):
        NN_output = self.NN.kneighbors(x)
        neighs = NN_output[1][0]
        neighs_dist = NN_output[0][0]
        neighs_y = self.y_train[neighs]
        return compute_all_features(neighs_y, neighs_dist, self.k_list, self.n_classes, self.eps)

@numba.njit
def _get_test_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        NNF = NearestNeighborsFeats(n_jobs=4, k_list=k_list, metric=metric)
        NNF.fit(X, y)
        test_knn_feats = NNF.predict(X)
        np.save(f'data/knn_feats_{metric}_test.npy', test_knn_feats)

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    return _deep_to_list(_get_test_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))

@numba.njit
def _get_train_feature_jit(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    for metric in ['minkowski', 'cosine']:
        skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        NNF = NearestNeighborsFeats(n_jobs=4, k_list=k_list, metric=metric)
        preds = cross_val_predict(NNF, X, y, cv=skf)
        np.save(f'data/knn_feats_{metric}_train.npy', preds)

def get_train_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    return _deep_to_list(_get_train_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))