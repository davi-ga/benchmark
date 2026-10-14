import numpy as np
import os
from typing import List
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _nearestneighborsfeats__get_features_for_one_impl(neighs_dist: np.ndarray, neighs_y: np.ndarray, eps: float, k_list: np.ndarray, n_classes: int) -> np.ndarray:
    return_list = numba.typed.List()
    for k in k_list:
        bincount = np.zeros(n_classes, dtype=np.float64)
        for i in range(k):
            c = neighs_y[i]
            if c < n_classes:
                bincount[c] += 1.0
        feats = np.zeros(n_classes, dtype=np.float64)
        for i in range(n_classes):
            feats[i] = bincount[i] / k
        return_list.append(feats)
    same_streak = 0
    first_y = neighs_y[0]
    for i in range(len(neighs_y)):
        if neighs_y[i] == first_y:
            same_streak = i + 1
        else:
            break
    feats_1 = np.zeros(1, dtype=np.float64)
    feats_1[0] = same_streak
    return_list.append(feats_1)
    feats_2 = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        min_dist = 999.0
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                if neighs_dist[i] < min_dist:
                    min_dist = neighs_dist[i]
        feats_2[c] = min_dist
    return_list.append(feats_2)
    feats_3 = np.zeros(n_classes, dtype=np.float64)
    closest_dist = neighs_dist[0]
    for i in range(len(neighs_dist)):
        if neighs_dist[i] < closest_dist:
            closest_dist = neighs_dist[i]
    for c in range(n_classes):
        min_norm = 999.0
        found = False
        for i in range(len(neighs_y)):
            if neighs_y[i] == c:
                val = neighs_dist[i] / (closest_dist + eps)
                if not found or val < min_norm:
                    min_norm = val
                    found = True
        feats_3[c] = min_norm
    return_list.append(feats_3)
    for k in k_list:
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        arr_5 = np.zeros(2, dtype=np.float64)
        arr_5[0] = feat_51
        arr_5[1] = feat_52
        return_list.append(arr_5)
    for k in k_list:
        feats_4 = np.zeros(n_classes, dtype=np.float64)
        for c in range(n_classes):
            sum_dist = 0.0
            count = 0
            for i in range(k):
                if neighs_y[i] == c:
                    sum_dist += neighs_dist[i]
                    count += 1
            if count > 0:
                feats_4[c] = sum_dist / (count + eps)
            else:
                feats_4[c] = 999.0
        return_list.append(feats_4)
    total_len = 0
    for arr in return_list:
        total_len += arr.shape[0]
    out = np.zeros(total_len, dtype=np.float64)
    idx = 0
    for arr in return_list:
        for i in range(arr.shape[0]):
            out[idx] = arr[i]
            idx += 1
    return out

@numba.njit
def _nearestneighborsfeats__get_features_for_one_jit(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    neighs_y = y_train[neighs]
    k_list_arr = k_list
    return _nearestneighborsfeats__get_features_for_one_impl(neighs_dist, neighs_y, eps, k_list_arr, n_classes)

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    return _deep_to_list(_nearestneighborsfeats__get_features_for_one_jit(_sanitize_for_numba(np.array(x) if isinstance(x, list) else x), _sanitize_for_numba(np.array(NN_output) if isinstance(NN_output, list) else NN_output), _sanitize_for_numba(np.array(eps) if isinstance(eps, list) else eps), _sanitize_for_numba(np.array(k_list) if isinstance(k_list, list) else k_list), _sanitize_for_numba(np.array(n_classes) if isinstance(n_classes, list) else n_classes), _sanitize_for_numba(np.array(y_train) if isinstance(y_train, list) else y_train)))

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        NNF = NearestNeighborsFeats(n_jobs=4, k_list=k_list, metric=metric)
        NNF.fit(X, y)
        test_knn_feats = NNF.predict(X)
        np.save(f'data/knn_feats_{metric}_test.npy', test_knn_feats)