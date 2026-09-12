from flask import Flask, request, jsonify
import time
import numpy as np
import sys

sys.path.insert(0, 'optimized/src')
from knn_feature import compute_knn_features_numba
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

print("🔥 Inicializando modelo sklearn + Numba JIT...")
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
k_list = np.array([1, 3, 5, 7, 9], dtype=np.int64)
n_classes = 2
eps = 1e-6

NN = NearestNeighbors(n_neighbors=max(k_list), metric='euclidean', n_jobs=1)
NN.fit(X_train)
print("✅ Modelo sklearn treinado")

X_warmup = np.random.rand(1, 10)
neighs_dist, neighs_idx = NN.kneighbors(X_warmup)
neighs_y = y_train[neighs_idx]
try:
    _ = compute_knn_features_numba(neighs_dist, neighs_y, k_list, n_classes, eps)
    print("✅ Warmup Numba OK")
except Exception as e:
    print(f"❌ Warmup Numba FALHOU: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_test = np.array(data['X'])
    
    start = time.perf_counter()
    neighs_dist, neighs_idx = NN.kneighbors(X_test)
    neighs_y = y_train[neighs_idx]
    result = compute_knn_features_numba(neighs_dist, neighs_y, k_list, n_classes, eps)
    elapsed = time.perf_counter() - start
    
    return jsonify({
        'result': result.tolist(),
        'time_ms': elapsed * 1000,
        'version': 'optimized-flash-35'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'optimized-flash-35'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
