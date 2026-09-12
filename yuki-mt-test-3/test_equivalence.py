import os
import sys
import math
import numpy as np

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

from src.knn_feature import NearestNeighborsFeats

def test_knn_feature_invariants():
    # Setup data
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.rand(10, 5)
    
    k_list = [1, 3, 5]
    n_classes = 3
    n_samples_test = X_test.shape[0]
    
    # Expected feature length calculation based on the _get_features_for_one method
    # 1. Fraction of objects of every class (k_list * n_classes)
    # 2. Same label streak (1)
    # 3. Min distance to objects of each class (n_classes)
    # 4. Min normalized distance to objects of each class (n_classes)
    # 5. Dist to Kth neighbor and normalized dist (k_list * 2)
    # 6. Mean distance to neighbors of each class for each K (k_list * n_classes)
    expected_feature_len = (len(k_list) * n_classes) + 1 + n_classes + n_classes + (len(k_list) * 2) + (len(k_list) * n_classes)

    model = NearestNeighborsFeats(n_jobs=1, k_list=k_list, metric='euclidean', n_classes=n_classes)
    model.fit(X_train, y_train)
    
    # First call (warm-up for Numba or just regular call)
    result1 = model.predict(X_test)
    
    # Second call for determinism check
    result2 = model.predict(X_test)
    
    # INVARIANT: Determinism
    np.testing.assert_allclose(result1, result2, rtol=1e-5, err_msg="Model prediction is not deterministic")
    
    # INVARIANT: Structural (correct shape)
    assert result1.shape == (n_samples_test, expected_feature_len), f"Expected shape ({n_samples_test}, {expected_feature_len}), got {result1.shape}"
    
    # INVARIANT: Non-degenerate
    assert not np.any(np.isnan(result1)), "Result contains NaN values"
    assert not np.any(np.isinf(result1)), "Result contains Inf values"
    
    # INVARIANT: Values check based on known logic
    for row in result1:
        # Check fraction of objects (first len(k_list) * n_classes features)
        idx = 0
        for _ in k_list:
            for c in range(n_classes):
                val = row[idx]
                assert 0.0 <= val <= 1.0, f"Fraction feature out of bounds: {val}"
                idx += 1
                
        # Check streak (next feature)
        streak = row[idx]
        assert streak >= 1, f"Streak feature must be >= 1, got {streak}"
        idx += 1
        
        # Check distances (should be >= 0 or 999)
        for _ in range(n_classes * 2): # Min dist and min normalized dist
            dist_val = row[idx]
            assert dist_val >= 0, f"Distance feature cannot be negative: {dist_val}"
            idx += 1
            
        # Check Kth distances and normalized Kth distances
        for _ in k_list:
            feat_51 = row[idx]
            feat_52 = row[idx+1]
            assert feat_51 >= 0, f"Kth distance cannot be negative: {feat_51}"
            assert feat_52 >= 0, f"Normalized Kth distance cannot be negative: {feat_52}"
            idx += 2
            
        # Check mean distances
        for _ in k_list:
            for _ in range(n_classes):
                mean_dist = row[idx]
                assert mean_dist >= 0, f"Mean distance cannot be negative: {mean_dist}"
                idx += 1
