import os
import sys
import math
import numpy as np
import pytest

module_dir = os.environ.get("MODULE_DIR", ".")
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
src_dir = os.path.join(module_dir, "src")
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import knn_feature
import app as flask_app


def test_knn_feature_predict_shape_and_values():
    np.random.seed(42)
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)

    model = knn_feature.NearestNeighborsFeats(n_jobs=1, k_list=[1, 3], metric="euclidean")
    model.fit(X, y)

    feats = model.predict(X[:4])
    assert feats.ndim == 2
    assert feats.shape[0] == 4
    assert feats.shape[1] > 0
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_knn_feature_cosine_metric():
    np.random.seed(42)
    X = np.random.rand(15, 4)
    y = np.random.randint(0, 2, 15)

    model = knn_feature.NearestNeighborsFeats(n_jobs=1, k_list=[2], metric="cosine")
    model.fit(X, y)

    feats = model.predict(X[:3])
    assert feats.shape[0] == 3
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_flask_health_endpoint():
    client = flask_app.app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data.get("status") == "ok"


def test_flask_predict_endpoint():
    client = flask_app.app.test_client()
    sample_data = np.random.rand(3, 10).tolist()
    response = client.post("/predict", json={"X": sample_data})
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert "result" in data
    assert len(data["result"]) == 3
    assert "time_ms" in data
    assert data["time_ms"] >= 0
