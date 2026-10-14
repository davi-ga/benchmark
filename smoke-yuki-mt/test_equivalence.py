import sys
import os
import numpy as np
import pytest

module_dir = os.environ.get('MODULE_DIR', '.')
sys.path.insert(0, module_dir)

from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([[1.0, 2.0]])
    neighs = np.array([1, 0, 2, 3])
    neighs_dist = np.array([0.1, 0.5, 0.8, 1.2])
    NN_output = (np.array([neighs_dist]), np.array([neighs]))
    eps = 1e-6
    k_list = [2, 3]
    n_classes = 2
    y_train = np.array([0, 1, 0, 1])
    
    result = nearestneighborsfeats__get_features_for_one(
        x=x,
        NN_output=NN_output,
        eps=eps,
        k_list=k_list,
        n_classes=n_classes,
        y_train=y_train
    )
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 1
    # check sizes based on return_list additions:
    # for k in k_list: n_classes (2 * 2 = 4)
    # streak: 1
    # min dist: n_classes (2)
    # min norm dist: n_classes (2)
    # dist kth: for k in k_list: 2 (2 * 2 = 4)
    # mean dist: for k in k_list: n_classes (2 * 2 = 4)
    # total = 4 + 1 + 2 + 2 + 4 + 4 = 17
    assert len(result) == 17
