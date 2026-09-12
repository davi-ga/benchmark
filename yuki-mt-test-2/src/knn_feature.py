import numpy as np
import numba
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from multiprocessing import Pool
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def compute_knn_features(neighs_dist: np.ndarray, neighs_y: np.ndarray, k_list: np.ndarray, n_classes: int, eps: float) -> np.ndarray:
    n_k = k_list.shape[0]
    out_size = n_k * n_classes + 1 + n_classes + n_classes + n_k * 2 + n_k * n_classes
    out = np.zeros(out_size, dtype=np.float64)
    idx = 0
    for i in range(n_k):
        k = k_list[i]
        bincount = np.zeros(n_classes, dtype=np.float64)
        for j in range(k):
            c = neighs_y[j]
            if c < n_classes:
                bincount[c] += 1.0
        for c in range(n_classes):
            out[idx] = bincount[c] / k
            idx += 1
    first_y = neighs_y[0]
    streak = 0
    for j in range(neighs_y.shape[0]):
        if neighs_y[j] == first_y:
            streak += 1
        else:
            break
    out[idx] = streak
    idx += 1
    for c in range(n_classes):
        min_d = 999.0
        for j in range(neighs_y.shape[0]):
            if neighs_y[j] == c:
                if neighs_dist[j] < min_d:
                    min_d = neighs_dist[j]
                break
        out[idx] = min_d
        idx += 1
    closest_dist = neighs_dist[0]
    for c in range(n_classes):
        min_nd = 999.0
        for j in range(neighs_y.shape[0]):
            if neighs_y[j] == c:
                min_nd = neighs_dist[j] / (closest_dist + eps)
                break
        out[idx] = min_nd
        idx += 1
    for i in range(n_k):
        k = k_list[i]
        d_k = neighs_dist[k - 1]
        out[idx] = d_k
        idx += 1
        out[idx] = d_k / (closest_dist + eps)
        idx += 1
    for i in range(n_k):
        k = k_list[i]
        for c in range(n_classes):
            sum_d = 0.0
            count = 0
            for j in range(k):
                if neighs_y[j] == c:
                    sum_d += neighs_dist[j]
                    count += 1
            if count > 0:
                out[idx] = sum_d / (count + eps)
            else:
                out[idx] = 999.0
            idx += 1
    return out

@numba.njit
def nearestneighborsfeats_fit(model_state: dict, X: np.ndarray, y: np.ndarray):
    k_list = model_state['k_list']
    metric = model_state.get('metric', 'euclidean')
    algorithm = 'brute' if metric == 'cosine' else 'auto'
    NN = NearestNeighbors(n_neighbors=int(np.max(k_list)), metric=metric, n_jobs=1, algorithm=algorithm)
    NN.fit(X)
    model_state['NN'] = NN
    model_state['y_train'] = y
    model_state['n_classes'] = np.unique(y).shape[0]
    if 'eps' not in model_state:
        model_state['eps'] = 1e-06

@numba.njit
def nearestneighborsfeats__get_features_for_one(model_state: dict, x: np.ndarray) -> np.ndarray:
    NN_output = model_state['NN'].kneighbors(x)
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    neighs_y = model_state['y_train'][neighs]
    return compute_knn_features(neighs_dist, neighs_y, model_state['k_list'], model_state['n_classes'], model_state['eps'])

@numba.njit
def nearestneighborsfeats_predict(model_state: dict, X: np.ndarray) -> np.ndarray:
    n_jobs = model_state.get('n_jobs', 1)
    test_feats = numba.typed.List()
    for i in range(X.shape[0]):
        test_feats.append(nearestneighborsfeats__get_features_for_one(model_state, X[i:i + 1]))
    return np.vstack(test_feats)

@numba.njit
def _get_test_feature_jit(X: np.ndarray, y: np.ndarray, k_list: list=[3, 8]):
    os.makedirs('data', exist_ok=True)
    k_arr = np.array(k_list, dtype=np.int32)
    for metric in ['minkowski', 'cosine']:
        model_state = {'k_list': k_arr, 'metric': metric, 'n_jobs': 4}
        nearestneighborsfeats_fit(model_state, X, y)
        test_knn_feats = nearestneighborsfeats_predict(model_state, X)
        np.save(f'data/knn_feats_{metric}_test.npy', test_knn_feats)

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: list=[3, 8]):
    return _deep_to_list(_get_test_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))

@numba.njit
def _get_train_feature_jit(X: np.ndarray, y: np.ndarray, k_list: list=[3, 8]):
    pass

def get_train_feature(X: np.ndarray, y: np.ndarray, k_list: list=[3, 8]):
    return _deep_to_list(_get_train_feature_jit(_sanitize_for_numba(X), _sanitize_for_numba(y), _sanitize_for_numba(k_list)))