import sys
import os
import numpy as np
import pytest

module_dir = os.environ.get("MODULE_DIR", ".")
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one_invariant():
    # Setup simple input data
    x = np.array([1.0, 2.0])
    # NN_output is a tuple of (distances, indices)
    # The function uses NN_output[1][0] and NN_output[0][0]
    dists = np.array([[0.1, 0.2, 0.3, 0.4]])
    indices = np.array([[2, 0, 1, 3]])
    NN_output = (dists, indices)
    eps = 1e-6
    k_list = [2, 3]
    n_classes = 3
    # y_train elements must cover the n_classes minimally, indices up to 3 are used
    y_train = np.array([0, 1, 0, 2, 1, 2])

    # Call function
    result = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)

    # Invariant checks
    assert result is not None
    assert isinstance(result, np.ndarray)
    
    # Expected size calculation:
    # 1. fraction of objects for each K: n_classes * len(k_list) = 3 * 2 = 6
    # 2. same label streak: 1
    # 3. min distance to objects of each class: n_classes = 3
    # 4. min norm distance to each class: n_classes = 3
    # 5. dist to Kth neighbor and norm dist for each K: 2 * len(k_list) = 4
    # 6. mean distance to neighbors of each class for each K: n_classes * len(k_list) = 6
    # Total = 6 + 1 + 3 + 3 + 4 + 6 = 23
    assert len(result) == 23
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
