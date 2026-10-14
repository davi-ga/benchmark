import numpy as np
import os
from typing import List
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def nearestneighborsfeats__get_features_for_one_compiled(neighs: np.ndarray, neighs_dist: np.ndarray, eps: float, k_list: np.ndarray, n_classes: int, y_train: np.ndarray) -> np.ndarray:
    neighs_y = np.zeros(neighs.shape[0], dtype=np.int64)
    for i in range(neighs.shape[0]):
        neighs_y[i] = y_train[neighs[i]]
    out_size = 0
    k_len = len(k_list)
    out_size += k_len * n_classes
    out_size += 1
    out_size += n_classes
    out_size += n_classes
    out_size += k_len * 2
    out_size += k_len * n_classes
    out_feats = np.zeros(out_size, dtype=np.float64)
    idx = 0
    for i in range(k_len):
        k = k_list[i]
        bincnt = np.zeros(n_classes, dtype=np.float64)
        for j in range(k):
            if neighs_y[j] < n_classes:
                bincnt[neighs_y[j]] += 1
        for c in range(n_classes):
            out_feats[idx] = bincnt[c] / k
            idx += 1
    streak = 1
    for i in range(1, neighs_y.shape[0]):
        if neighs_y[i] == neighs_y[0]:
            streak += 1
        else:
            break
    out_feats[idx] = streak
    idx += 1
    for c in range(n_classes):
        min_d = 999.0
        for i in range(neighs_y.shape[0]):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_d:
                    min_d = neighs_dist[i]
        out_feats[idx] = min_d
        idx += 1
    closest_dist = neighs_dist[0]
    for i in range(1, neighs_dist.shape[0]):
        if neighs_dist[i] < closest_dist:
            closest_dist = neighs_dist[i]
    for c in range(n_classes):
        min_nd = 999.0
        found = False
        for i in range(neighs_y.shape[0]):
            if neighs_y[i] == c:
                nd = neighs_dist[i] / (closest_dist + eps)
                if not found or nd < min_nd:
                    min_nd = nd
                    found = True
        if not found:
            min_nd = 999.0
        out_feats[idx] = min_nd
        idx += 1
    for i in range(k_len):
        k = k_list[i]
        f51 = neighs_dist[k - 1]
        f52 = f51 / (neighs_dist[0] + eps)
        out_feats[idx] = f51
        idx += 1
        out_feats[idx] = f52
        idx += 1
    for i in range(k_len):
        k = k_list[i]
        for c in range(n_classes):
            s = 0.0
            cnt = 0
            for j in range(k):
                if neighs_y[j] == c:
                    s += neighs_dist[j]
                    cnt += 1
            if cnt > 0:
                out_feats[idx] = s / (cnt + eps)
            else:
                out_feats[idx] = 999.0
            idx += 1
    return out_feats

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    return nearestneighborsfeats__get_features_for_one_compiled(neighs, neighs_dist, eps, k_list, n_classes, y_train)

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        pass