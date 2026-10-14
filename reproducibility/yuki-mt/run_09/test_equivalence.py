import sys
import os
import numpy as np
import math

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([[1.0, 2.0]])
    NN_output = (
        np.array([[0.1, 0.2, 0.5, 0.6, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]]),
        np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    )
    eps = 1e-6
    k_list = [3, 8]
    n_classes = 2
    y_train = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    
    res = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    
    assert isinstance(res, np.ndarray)
    assert len(res) > 0
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
    
    # Idempotence
    res2 = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    np.testing.assert_allclose(res, res2)
