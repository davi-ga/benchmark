import sys
import os
import numpy as np
import math

module_dir = os.environ.get("MODULE_DIR", ".")
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([[1.0, 2.0]])
    NN_output_dists = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    NN_output_indices = np.array([[0, 1, 2, 3, 4]])
    NN_output = (NN_output_dists, NN_output_indices)
    eps = 1e-6
    k_list = [2, 4]
    n_classes = 3
    y_train = np.array([0, 1, 0, 2, 1])
    
    result = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    
    # The function returns a flat 1D array stacked from several feature groups:
    # 1. fraction of objects of every class (n_classes per k -> 3 * 2 = 6)
    # 2. same label streak (1)
    # 3. min dist to objects of each class (n_classes -> 3)
    # 4. min *normalized* dist to objects of each class (n_classes -> 3)
    # 5. dist to Kth neighbor and normalized dist (2 per k -> 2 * 2 = 4)
    # 6. mean dist to neighbors of each class for each K (n_classes per k -> 3 * 2 = 6)
    # Total = 6 + 1 + 3 + 3 + 4 + 6 = 23
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == 23
    for val in result:
        assert not math.isnan(val)
