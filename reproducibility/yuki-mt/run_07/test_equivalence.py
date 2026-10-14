import sys
import os
import numpy as np

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))
from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([1.0, 2.0, 3.0])
    # NN_output is a tuple of (distances, indices)
    distances = np.array([[0.1, 0.5, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0]])
    indices = np.array([[3, 1, 4, 0, 2, 5, 6, 7]])
    NN_output = (distances, indices)
    eps = 1e-6
    k_list = [2, 5]
    n_classes = 3
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    
    result = nearestneighborsfeats__get_features_for_one(x, NN_output, eps, k_list, n_classes, y_train)
    
    # Assert return is a 1D numpy array
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    
    # Number of features should be consistent based on k_list and n_classes
    # Fraction of objects for each class (for each k in k_list) -> len(k_list) * n_classes
    # Same label streak -> 1
    # Min distance to each class -> n_classes
    # Min normalized distance to each class -> n_classes
    # Dist and norm dist to Kth neighbor (for each k in k_list) -> len(k_list) * 2
    # Mean dist to neighbors of each class for each K -> len(k_list) * n_classes
    expected_len = (len(k_list) * n_classes) + 1 + n_classes + n_classes + (len(k_list) * 2) + (len(k_list) * n_classes)
    assert len(result) == expected_len
    
    # All results should be non-NaN
    assert not np.isnan(result).any()
