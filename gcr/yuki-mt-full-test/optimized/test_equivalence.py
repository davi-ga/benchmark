import sys
import os
import numpy as np
import math

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

from knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([1.0, 2.0])
    # NN_output format: (distances, indices)
    NN_output = (
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        np.array([[0, 1, 2, 3, 4]])
    )
    eps = 1e-6
    k_list = [2, 4]
    n_classes = 2
    y_train = np.array([0, 1, 0, 1, 0])

    # Warm-up call (if needed for numba)
    nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    
    result = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    
    assert result is not None
    assert len(result) > 0
    
    # Check idempotence
    result2 = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    assert np.allclose(result, result2)
    
    # Check basic non-degeneracy
    for val in result:
        assert not math.isnan(val)
