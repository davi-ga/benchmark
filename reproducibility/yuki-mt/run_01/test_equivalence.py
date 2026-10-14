import sys
import os
sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))
import numpy as np
import math
from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([0.0])
    y_train = np.array([0, 1, 0, 0, 1, 1])
    # NN_output format: (distances, indices)
    NN_output = (
        np.array([[0.1, 0.2, 0.3, 0.4]]),
        np.array([[0, 2, 4, 1]])
    )
    eps = 1e-6
    k_list = np.array([2, 4])
    n_classes = 2
    
    result1 = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    result2 = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    
    assert isinstance(result1, np.ndarray)
    assert result1.ndim == 1
    assert result1.shape[0] > 0
    assert not np.any(np.isnan(result1))
    assert not np.any(np.isinf(result1))
    # Check idempotence
    np.testing.assert_array_almost_equal(result1, result2)
