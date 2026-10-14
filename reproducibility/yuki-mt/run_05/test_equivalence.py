import sys
import os
import numpy as np
import math
import pytest

# Setup sys.path
module_dir = os.environ.get('MODULE_DIR', '.')
sys.path.insert(0, module_dir)

from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    # Setup data
    x = np.array([[1.0, 2.0, 3.0]])
    k_list = [2, 5]
    n_classes = 3
    eps = 1e-6
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    
    # NN_output format: (distances, indices)
    # We need indices to index y_train. len(indices) should be >= max(k_list)
    indices = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    distances = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    NN_output = (distances, indices)
    
    result = nearestneighborsfeats__get_features_for_one(
        x=x,
        NN_output=NN_output,
        eps=eps,
        k_list=k_list,
        n_classes=n_classes,
        y_train=y_train
    )
    
    # Assertions
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) > 0
    
    # Features should not have NaN
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
