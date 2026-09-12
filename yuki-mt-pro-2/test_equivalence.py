import sys
import os
sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

import numpy as np
import math
from src.knn_feature import NearestNeighborsFeats

def test_nearest_neighbors_features():
    # Generate a small dataset
    np.random.seed(42)
    X_train = np.random.rand(50, 4)
    y_train = np.random.randint(0, 3, 50)  # 3 classes: 0, 1, 2
    
    X_test = np.random.rand(10, 4)
    
    k_list = [2, 5]
    n_classes = 3
    
    model = NearestNeighborsFeats(n_jobs=1, k_list=k_list, metric='euclidean')
    model.fit(X_train, y_train)
    
    # First call
    preds1 = model.predict(X_test)
    
    # Second call for idempotence
    preds2 = model.predict(X_test)
    
    assert isinstance(preds1, np.ndarray), "Prediction must return a numpy array"
    
    # Idempotence check
    np.testing.assert_array_equal(preds1, preds2, err_msg="Deterministic predict must return identical results")
    
    # Shape check
    expected_features_count = (
        (len(k_list) * n_classes) +    # Fraction of objects
        1 +                            # Same label streak
        n_classes +                    # Minimum distance
        n_classes +                    # Minimum normalized distance
        (len(k_list) * 2) +            # Distance to Kth neighbor and normalized distance
        (len(k_list) * n_classes)      # Mean distance to neighbors of each class
    )
    
    assert preds1.shape[0] == X_test.shape[0], "Output row count must match input row count"
    assert preds1.shape[1] == expected_features_count, "Output feature count is incorrect"
    
    # Data sanity check
    assert not np.isnan(preds1).any(), "Output contains NaNs"
    assert not np.isinf(preds1).any(), "Output contains Infs"
    
    # Specific feature invariants
    for i in range(preds1.shape[0]):
        # First len(k_list) * n_classes are fractions (in [0, 1])
        frac_feats = preds1[i, :len(k_list)*n_classes]
        assert np.all(frac_feats >= 0) and np.all(frac_feats <= 1), "Fractions must be between 0 and 1"
        
        # Streak feature is at len(k_list)*n_classes, must be >= 1
        streak = preds1[i, len(k_list)*n_classes]
        assert streak >= 1, "Streak must be at least 1"
