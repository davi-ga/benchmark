import sys
import os
import numpy as np
import math
sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))
from src.knn_feature import nearestneighborsfeats__get_features_for_one

def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([[0.5, 0.5]])
    NN_output_0 = np.array([[0.1, 0.2, 0.3, 0.4]])
    NN_output_1 = np.array([[0, 1, 2, 3]])
    NN_output = (NN_output_0, NN_output_1)
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
    
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) > 0
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
