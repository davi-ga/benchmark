import sys
import os
import numpy as np
import math

module_dir = os.environ.get("MODULE_DIR", ".")
sys.path.insert(0, module_dir)

from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([[1.0, 2.0]])
    NN_output = (np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]), np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))
    eps = 1e-6
    k_list = [1, 3]
    n_classes = 2
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    
    result = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    
    # Assert properties
    assert result is not None
    assert len(result) > 0
    assert len(result.shape) == 1
    
    # Idempotence
    result2 = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    assert np.allclose(result, result2, equal_nan=True)
