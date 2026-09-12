import numba
import numpy as np
from flask import Flask, request, jsonify
import time
import sys
sys.path.insert(0, 'src')
from knn_feature import nearestneighborsfeats_predict, nearestneighborsfeats_fit
from _numba_helpers import _sanitize_for_numba, _deep_to_list
app = Flask(__name__)
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
model_state = {}
model_state['k_list'] = np.array([1, 3, 5, 7, 9], dtype=np.int32)
model_state['metric'] = 'euclidean'
model_state['eps'] = 1e-06
model_state['n_classes'] = 2
nearestneighborsfeats_fit(model_state, X_train, y_train)

@numba.njit
@app.route('/predict', methods=['POST'])
def _predict_jit():
    data = request.json
    X_test = np.array(data['X'])
    start = time.perf_counter()
    result = nearestneighborsfeats_predict(model_state, X_test)
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