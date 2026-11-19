import numpy as np
import os
from typing import List

def nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train):
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    neighs_y = y_train[neighs]
    return_list = []
    for k in k_list:
        bincount = np.bincount(neighs_y[:k])
        bincount = np.append(bincount, np.zeros(n_classes - len(bincount)))
        feats = []
        for i in range(n_classes):
            feats.append(bincount[i] / k)
        assert len(feats) == n_classes
        return_list += [feats]
    a = np.where(neighs_y == neighs_y[0])[0]
    feats = [a[a == range(len(a))][-1] + 1]
    assert len(feats) == 1
    return_list += [feats]
    feats = []
    for c in range(n_classes):
        feats.append(np.min(np.where(neighs_y == c, neighs_dist, 999)))
    assert len(feats) == n_classes
    return_list += [feats]
    feats = []
    closest_dist = np.min(neighs_dist)
    norm_dist = neighs_dist / (closest_dist + eps)
    for c in range(n_classes):
        dists = norm_dist[neighs_y == c]
        if len(dists) > 0:
            feats.append(dists.min())
        else:
            feats.append(999)
    assert len(feats) == n_classes
    return_list += [feats]
    for k in k_list:
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        return_list += [[feat_51, feat_52]]
    for k in k_list:
        feats = []
        for c in range(n_classes):
            dists = neighs_dist[:k][neighs_y[:k] == c]
            if len(dists) > 0:
                feats.append(dists.sum() / (len(dists) + eps))
            else:
                feats.append(999)
        assert len(feats) == n_classes
        return_list += [feats]
    knn_feats = np.hstack(return_list)
    return knn_feats

def get_test_feature(X: np.ndarray, y: np.ndarray, k_list: List[int]=[3, 8]):
    os.makedirs('data', exist_ok=True)
    for metric in ['minkowski', 'cosine']:
        NNF = NearestNeighborsFeats(n_jobs=4, k_list=k_list, metric=metric)
        NNF.fit(X, y)
        test_knn_feats = NNF.predict(X)
        np.save(f'data/knn_feats_{metric}_test.npy', test_knn_feats)