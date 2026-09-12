import numba
import sys
import numpy as np
from flask import Flask, jsonify
sys.path.insert(0, 'src')
from knn_feature import fit_knn
from _numba_helpers import _sanitize_for_numba, _deep_to_list
app = Flask(__name__)
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
model_data = fit_knn(X_train, y_train)

@numba.njit
@app.route('/health', methods=['GET'])
def _health_jit():
    return jsonify({'status': 'ok', 'version': 'original'})

def health():
    return _deep_to_list(_health_jit())
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)