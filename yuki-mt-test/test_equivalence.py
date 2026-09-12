import os
import sys
import pytest
import numpy as np
import math

# Ensure the module directory is in sys.path
sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

from src.knn_feature import NearestNeighborsFeats

def test_knn_features_invariants():
    # Generate some random data
    np.random.seed(42)
    X_train = np.random.rand(50, 5)
    y_train = np.random.randint(0, 3, 50)  # 3 classes
    
    k_list = [1, 3, 5]
    n_classes = 3
    metric = 'euclidean'
    
    # Initialize and fit
    model = NearestNeighborsFeats(n_jobs=1, k_list=k_list, metric=metric, n_classes=n_classes)
    model.fit(X_train, y_train)
    
    # Predict
    X_test = np.random.rand(10, 5)
    result = model.predict(X_test)
    
    # Expected feature length:
    # 1. Fraction of objects for each class for each k: len(k_list) * n_classes
    # 2. Same label streak: 1
    # 3. Min distance to objects of each class: n_classes
    # 4. Min normalized distance to objects of each class: n_classes
    # 5. Distance to Kth neighbor and normalized distance: len(k_list) * 2
    # 6. Mean distance to neighbors of each class for each K: len(k_list) * n_classes
    expected_feat_len = (len(k_list) * n_classes) + 1 + n_classes + n_classes + (len(k_list) * 2) + (len(k_list) * n_classes)
    
    # Check overall shape
    assert result.shape == (10, expected_feat_len), f"Expected shape (10, {expected_feat_len}), got {result.shape}"
    
    for i in range(result.shape[0]):
        row = result[i]
        idx = 0
        
        # 1. Fraction of objects
        for _ in k_list:
            for c in range(n_classes):
                val = row[idx]
                assert not math.isnan(val), "Fraction should not be NaN"
                assert not math.isinf(val), "Fraction should not be Inf"
                assert 0.0 <= val <= 1.0, "Fraction must be in [0, 1]"
                idx += 1
                
        # 2. Same label streak
        val = row[idx]
        assert not math.isnan(val), "Streak should not be NaN"
        assert val >= 1.0, "Streak must be at least 1"
        idx += 1
        
        # 3. Min distance
        for _ in range(n_classes):
            val = row[idx]
            assert not math.isnan(val), "Min distance should not be NaN"
            assert val >= 0.0, "Distance must be non-negative"
            idx += 1
            
        # 4. Min normalized distance
        for _ in range(n_classes):
            val = row[idx]
            assert not math.isnan(val), "Min normalized distance should not be NaN"
            assert val >= 0.0, "Normalized distance must be non-negative"
            idx += 1
            
        # 5. Distance to Kth neighbor and normalized
        for _ in k_list:
            val1 = row[idx]
            val2 = row[idx + 1]
            assert not math.isnan(val1), "Kth distance should not be NaN"
            assert val1 >= 0.0, "Distance must be non-negative"
            assert not math.isnan(val2), "Kth normalized distance should not be NaN"
            assert val2 >= 0.0, "Normalized distance must be non-negative"
            idx += 2
            
        # 6. Mean distance
        for _ in k_list:
            for _ in range(n_classes):
                val = row[idx]
                assert not math.isnan(val), "Mean distance should not be NaN"
                assert val >= 0.0, "Mean distance must be non-negative"
                idx += 1
