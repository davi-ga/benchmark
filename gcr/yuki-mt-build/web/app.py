import os
import sys
import time

import numpy as np
from flask import Flask, g, jsonify, request

# A pasta de funções é o artefato produzido/consumido pelo forge.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "functions"))
import knn_feature  # noqa: E402

from sklearn.neighbors import NearestNeighbors  # noqa: E402

app = Flask(__name__)
VERSION = os.getenv("VERSION", "unknown")

X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
k_list = np.array([1, 3, 5, 7, 9], dtype=np.int64)
n_classes = 2
eps = 1e-6

NN = NearestNeighbors(n_neighbors=int(max(k_list)), metric="euclidean", n_jobs=1)
NN.fit(X_train)

# Warm-up: absorve a compilação JIT (quando a variante for a do forge).
_wx = X_train[0:1]
_wd, _wi = NN.kneighbors(_wx)
_warm = knn_feature.get_features_for_one((_wd, _wi), eps, k_list, n_classes, y_train)
print("[warmup] ok", np.asarray(_warm).shape)


@app.before_request
def _mark_request_start():
    g.t0 = time.perf_counter()


@app.after_request
def _add_billed_header(resp):
    if hasattr(g, "t0"):
        resp.headers["X-Billed-Ms"] = f"{(time.perf_counter() - g.t0) * 1000:.3f}"
    return resp


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    X_test = np.array(data["X"], dtype=np.float64)

    start = time.perf_counter()
    dist_all, idx_all = NN.kneighbors(X_test)
    results = [
        knn_feature.get_features_for_one(
            (dist_all[i : i + 1], idx_all[i : i + 1]), eps, k_list, n_classes, y_train
        )
        for i in range(len(X_test))
    ]
    result = np.array(results)
    elapsed = time.perf_counter() - start
    billed_ms = (time.perf_counter() - g.t0) * 1000.0

    return jsonify(
        {
            "result": result.tolist(),
            "time_ms": elapsed * 1000.0,
            "billed_ms": billed_ms,
            "version": VERSION,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": VERSION})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
