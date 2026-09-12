import os
import sys
import numpy as np
import pytest

MODULE_DIR = os.environ.get("MODULE_DIR", ".")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

SRC_DIR = os.path.join(MODULE_DIR, "src")
if os.path.exists(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from knn_feature import NearestNeighborsFeats
from app import app


def test_knn_feature_fit_predict():
    X_train = np.random.rand(30, 4)
    y_train = np.array([0, 1] * 15)
    X_test = np.random.rand(5, 4)

    k_list = [1, 3]
    nn_feats = NearestNeighborsFeats(n_jobs=1, k_list=k_list, metric="minkowski")
    nn_feats.fit(X_train, y_train)

    res = nn_feats.predict(X_test)

    assert isinstance(res, np.ndarray)
    assert res.shape[0] == len(X_test)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()


def test_knn_feature_get_features_for_one():
    X_train = np.random.rand(20, 5)
    y_train = np.random.randint(0, 2, 20)

    nn_feats = NearestNeighborsFeats(n_jobs=1, k_list=[1, 3], metric="euclidean")
    nn_feats.fit(X_train, y_train)

    one_sample = X_train[0:1]
    feats = nn_feats._get_features_for_one(one_sample)

    assert isinstance(feats, np.ndarray)
    assert feats.ndim == 1
    assert len(feats) > 0
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_flask_health_endpoint():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("status") == "ok"


def test_flask_predict_endpoint():
    client = app.test_client()
    sample_x = np.random.rand(3, 10).tolist()
    response = client.post("/predict", json={"X": sample_x})
    assert response.status_code == 200
    data = response.get_json()
    assert "result" in data
    assert len(data["result"]) == 3
    assert "time_ms" in data
    assert data["time_ms"] >= 0
