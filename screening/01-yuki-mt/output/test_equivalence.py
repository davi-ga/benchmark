import os
import sys
sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

import numpy as np
import math
from knn_feature import (
    compute_knn_features,
    compute_class_fractions,
    compute_min_distances,
    compute_normalized_distances,
    compute_mean_distances
)

def test_compute_knn_features_invariants():
    np.random.seed(42)
    k_list = np.array([2, 5, 10], dtype=np.int32)
    n_classes = 3
    n_neighbors = 15
    neighs_dist = np.sort(np.random.rand(n_neighbors).astype(np.float64))
    neighs_y = np.random.randint(0, n_classes, size=n_neighbors, dtype=np.int32)
    eps = 1e-6

    result = compute_knn_features(neighs_dist, neighs_y, k_list, n_classes, eps)
    
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    
    n_k = len(k_list)
    expected_len = n_k * n_classes + 1 + n_classes + n_classes + n_k * 2 + n_k * n_classes
    assert len(result) == expected_len
    
    # Invariants for computed feature values
    for v in result:
        assert not math.isnan(v)
        assert not math.isinf(v)
        assert v >= 0.0

def test_compute_class_fractions_invariants():
    np.random.seed(42)
    k = 5
    n_classes = 4
    neighs_y = np.random.randint(0, n_classes, size=10, dtype=np.int32)
    
    result = compute_class_fractions(neighs_y, k, n_classes)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == n_classes
    
    # Fractions should be in [0, 1] and sum to <= 1.0 (some labels might be >= n_classes, though not here)
    for v in result:
        assert not math.isnan(v)
        assert not math.isinf(v)
        assert 0.0 <= v <= 1.0
    assert np.sum(result) <= 1.0001

def test_compute_min_distances_invariants():
    np.random.seed(42)
    n_classes = 3
    n_neighbors = 10
    neighs_dist = np.sort(np.random.rand(n_neighbors).astype(np.float64))
    neighs_y = np.random.randint(0, n_classes, size=n_neighbors, dtype=np.int32)

    result = compute_min_distances(neighs_dist, neighs_y, n_classes)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == n_classes
    
    for v in result:
        assert not math.isnan(v)
        assert not math.isinf(v)
        assert v >= 0.0

def test_compute_normalized_distances_invariants():
    np.random.seed(42)
    n_classes = 3
    n_neighbors = 10
    neighs_dist = np.sort(np.random.rand(n_neighbors).astype(np.float64))
    neighs_y = np.random.randint(0, n_classes, size=n_neighbors, dtype=np.int32)
    eps = 1e-6

    result = compute_normalized_distances(neighs_dist, neighs_y, n_classes, eps)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == n_classes
    
    for v in result:
        assert not math.isnan(v)
        assert not math.isinf(v)
        assert v >= 0.0

def test_compute_mean_distances_invariants():
    np.random.seed(42)
    k = 5
    n_classes = 3
    n_neighbors = 10
    neighs_dist = np.sort(np.random.rand(n_neighbors).astype(np.float64))
    neighs_y = np.random.randint(0, n_classes, size=n_neighbors, dtype=np.int32)
    eps = 1e-6

    result = compute_mean_distances(neighs_dist, neighs_y, k, n_classes, eps)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == n_classes
    
    for v in result:
        assert not math.isnan(v)
        assert not math.isinf(v)
        assert v >= 0.0
