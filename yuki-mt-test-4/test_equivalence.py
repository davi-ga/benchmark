import os
import sys
import pytest
import numpy as np
import math

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

from src.knn_feature import NearestNeighborsFeats


def test_knn_feature_invariants():
    # Generate consistent random data for testing invariants
    np.random.seed(42)
    X_train = np.random.rand(50, 5)
    y_train = np.random.randint(0, 3, 50)
    X_test = np.random.rand(10, 5)

    k_list = [2, 5]
    n_classes = 3

    model = NearestNeighborsFeats(n_jobs=1, k_list=k_list, metric='euclidean', n_classes=n_classes)
    model.fit(X_train, y_train)
    
    # First call
    preds1 = model.predict(X_test)
    
    # Second call for determinism
    preds2 = model.predict(X_test)
    
    # Check determinism
    np.testing.assert_array_equal(preds1, preds2)
    
    # Check structural invariant (shape)
    expected_feat_len = 0
    # 1. fraction of objects for each K
    expected_feat_len += len(k_list) * n_classes
    # 2. same label streak
    expected_feat_len += 1
    # 3. min distance to objects of each class
    expected_feat_len += n_classes
    # 4. min normalized distance
    expected_feat_len += n_classes
    # 5. distance to Kth neighbor + normalized dist
    expected_feat_len += len(k_list) * 2
    # 6. mean distance to neighbors of each class for each K
    expected_feat_len += len(k_list) * n_classes
    
    assert preds1.shape == (X_test.shape[0], expected_feat_len)
    
    # Assert no NaNs or Infs
    assert not np.isnan(preds1).any()
    assert not np.isinf(preds1).any()
    
    # Fraction of objects features should be between 0 and 1
    # They are the first len(k_list) * n_classes features
    frac_feats = preds1[:, :len(k_list) * n_classes]
    assert (frac_feats >= 0).all() and (frac_feats <= 1).all()

    # Same label streak is a positive integer
    streak_feats = preds1[:, len(k_list) * n_classes]
    assert (streak_feats >= 1).all()
    
    # Minimum distances and normalized distances are >= 0
    dist_start_idx = len(k_list) * n_classes + 1
    dist_feats = preds1[:, dist_start_idx:]
    assert (dist_feats >= 0).all()
