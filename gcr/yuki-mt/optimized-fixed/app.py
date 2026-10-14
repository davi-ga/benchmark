from flask import Flask, request, jsonify
import time
import sys

import numpy as np

sys.path.insert(0, "optimized")
import knn_feature  # noqa: E402

from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sentinel import Middleware, ResourceMonitor  # noqa: E402

app = Flask(__name__)

VERSION = "optimized-fixed"

X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
k_list = np.array([1, 3, 5, 7, 9], dtype=np.int64)
n_classes = 2
eps = 1e-6

NN = NearestNeighbors(n_neighbors=int(max(k_list)), metric="euclidean", n_jobs=1)
NN.fit(X_train)

mw = Middleware(modules=[knn_feature])

_wx = X_train[0:1]
_wd, _wi = NN.kneighbors(_wx)
WARMUP_RESULT = mw.warmup(
    inputs={
        "nearestneighborsfeats__get_features_for_one": (
            _wx,
            (_wd, _wi),
            eps,
            k_list,
            n_classes,
            y_train,
        )
    }
)
print(f"[warmup] { {k: v['status'] for k, v in WARMUP_RESULT.items()} }")


def _predict_one(x_i):
    nd, ni = NN.kneighbors(x_i)
    return knn_feature.nearestneighborsfeats__get_features_for_one(
        x_i, (nd, ni), eps, k_list, n_classes, y_train
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X_test = np.array(data["X"])

    start = time.perf_counter()
    dist_all, idx_all = NN.kneighbors(X_test)
    results = [
        knn_feature.nearestneighborsfeats__get_features_for_one(
            X_test[i : i + 1], (dist_all[i : i + 1], idx_all[i : i + 1]),
            eps, k_list, n_classes, y_train,
        )
        for i in range(len(X_test))
    ]
    result = np.array(results)
    elapsed = time.perf_counter() - start
    resources = {}

    return jsonify(
        {
            "result": result.tolist(),
            "time_ms": elapsed * 1000,
            "version": VERSION,
            **resources,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": VERSION, "warmup": {k: v["status"] for k, v in WARMUP_RESULT.items()}})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
