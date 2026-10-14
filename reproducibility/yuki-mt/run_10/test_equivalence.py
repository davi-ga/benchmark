import os
import sys
import numpy as np
import math

sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))

from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([[1.0, 2.0]])
    NN_output = (
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        np.array([[0, 1, 2, 3, 4]])
    )
    eps = 1e-6
    k_list = [2, 4]
    n_classes = 3
    y_train = np.array([0, 1, 0, 2, 1])
    
    res = nearestneighborsfeats__get_features_for_one(
        x=x, 
        NN_output=NN_output, 
        eps=eps, 
        k_list=k_list, 
        n_classes=n_classes, 
        y_train=y_train
    )
    
    assert isinstance(res, np.ndarray)
    assert len(res) > 0
    assert not np.any(np.isnan(res))
