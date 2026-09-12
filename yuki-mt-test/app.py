import numba
from flask import Flask, request, jsonify
import time
import numpy as np
import sys
sys.path.insert(0, 'src')
from knn_feature import NearestNeighborsFeats_predict, NearestNeighborsFeats_fit
from _numba_helpers import _sanitize_for_numba, _deep_to_list
app = Flask(__name__)
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
n_classes = 2
k_list = [1, 3, 5, 7, 9]
eps = 1e-06
NN_model, n_classes = NearestNeighborsFeats_fit(X_train, y_train, k_list, 'euclidean')

@numba.njit
@app.route('/predict', methods=['POST'])
def _predict_jit():
    data = request.json
    X_test = np.array(data['X'])
    start = time.perf_counter()
    result = NearestNeighborsFeats_predict(X_test, NN_model, y_train, k_list, n_classes, eps)
    elapsed = time.perf_counter() - start
    return jsonify({'result': result.tolist(), 'time_ms': elapsed * 1000, 'version': 'refactored'})

def predict():
    return _deep_to_list(_predict_jit())

@numba.njit
@app.route('/health', methods=['GET'])
def _health_jit():
    return jsonify({'status': 'ok', 'version': 'refactored'})

def health():
    return _deep_to_list(_health_jit())
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)