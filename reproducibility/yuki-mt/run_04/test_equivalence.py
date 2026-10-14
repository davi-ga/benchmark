import sys
import os
import math
import numpy as np

sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))

from src.knn_feature import nearestneighborsfeats__get_features_for_one


def test_nearestneighborsfeats__get_features_for_one():
    x = np.array([[0.5]])
    NN_output = (np.array([[0.1, 0.2, 0.5, 0.6, 0.9]]), np.array([[0, 1, 2, 3, 4]]))
    eps = 1e-6
    k_list = [2, 4]
    n_classes = 2
    y_train = np.array([0, 1, 0, 1, 0])

    result = nearestneighborsfeats__get_features_for_one(
        x=x,
        NN_output=NN_output,
        eps=eps,
        k_list=k_list,
        n_classes=n_classes,
        y_train=y_train
    )

    assert result is not None
    # Length should be: len(k_list)*n_classes + 1 + n_classes + n_classes + len(k_list)*2 + len(k_list)*n_classes = 4 + 1 + 2 + 2 + 4 + 4 = 17
    assert len(result) == 17

    for val in result:
        assert not math.isnan(val)
        assert not math.isinf(val)
