import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))
from src.knn_feature import NearestNeighborsFeats


def test_nearest_neighbors_feats_invariants():
    np.random.seed(42)
    X_train = np.random.rand(50, 4)
    y_train = np.random.randint(0, 3, 50)
    
    k_list = [2, 5]
    n_classes = 3
    model = NearestNeighborsFeats(n_jobs=1, k_list=k_list, metric='euclidean', n_classes=n_classes)
    model.fit(X_train, y_train)
    
    X_test = np.random.rand(10, 4)
    res = model.predict(X_test)
    
    # Shape check
    assert res.shape[0] == X_test.shape[0]
    
    expected_feats = (len(k_list) * n_classes) + 1 + n_classes + n_classes + (len(k_list) * 2) + (len(k_list) * n_classes)
    assert res.shape[1] == expected_feats
    
    # Invariant checks for each feature segment
    for row in res:
        for val in row:
            assert not math.isnan(val)
            assert not math.isinf(val)
            
        idx = 0
        
        # 1. Fractions in [0, 1]
        for _ in k_list:
            for _ in range(n_classes):
                assert 0.0 <= row[idx] <= 1.0
                idx += 1
                
        # 2. Same label streak >= 1
        assert row[idx] >= 1.0
        idx += 1
        
        # 3. Minimum distance >= 0
        for _ in range(n_classes):
            assert row[idx] >= 0.0
            idx += 1
            
        # 4. Minimum normalized distance >= 0
        for _ in range(n_classes):
            assert row[idx] >= 0.0
            idx += 1
            
        # 5. Distance to Kth neighbor and normalized distance >= 0
        for _ in k_list:
            assert row[idx] >= 0.0
            assert row[idx+1] >= 0.0
            idx += 2
            
        # 6. Mean distance >= 0
        for _ in k_list:
            for _ in range(n_classes):
                assert row[idx] >= 0.0
                idx += 1
