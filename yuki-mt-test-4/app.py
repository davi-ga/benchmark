import numba
import sys
import time
import numpy as np
from flask import Flask, request, jsonify
sys.path.insert(0, 'src')
from knn_feature import get_test_feature
from _numba_helpers import _sanitize_for_numba, _deep_to_list
app = Flask(__name__)
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)

@numba.njit
@app.route('/predict', methods=['POST'])
def _predict_jit():
    data = request.json
    X_test = np.array(data['X'])
    start = time.perf_counter()
    result = X_test
    elapsed = time.perf_counter() - start
    return jsonify({'result': result.tolist(), 'time_ms': elapsed * 1000, 'version': 'original'})

def predict():
    return _deep_to_list(_predict_jit())

@numba.njit
@app.route('/health', methods=['GET'])
def _health_jit():
    return jsonify({'status': 'ok', 'version': 'original'})

def health():
    return _deep_to_list(_health_jit())
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)