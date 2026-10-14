import os
import sys
import numpy as np
import math

# Ensure the source module is in the import path
sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([[1.0, 2.0]])
    neighs_dist = np.array([[0.1, 0.2, 0.3, 0.4]])
    neighs = np.array([[1, 2, 0, 3]])
    NN_output = (neighs_dist, neighs)
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
    
    # Assert result is an array
    assert isinstance(result, np.ndarray)
    
    # Length of features depends on k_list and n_classes
    # fractions (n_classes per k) = 2 * 2 = 4
    # streak = 1
    # min dist = n_classes = 2
    # min norm dist = n_classes = 2
    # dist to kth (2 per k) = 2 * 2 = 4
    # mean dist (n_classes per k) = 2 * 2 = 4
    # Total = 4 + 1 + 2 + 2 + 4 + 4 = 17
    assert len(result) == 17
    
    # Verify no NaN values
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
