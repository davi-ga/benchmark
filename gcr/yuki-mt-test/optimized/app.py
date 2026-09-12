from flask import Flask, request, jsonify
import time
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, 'optimized/src')
from knn_feature import nearestneighborsfeats__get_features_for_one

app = Flask(__name__)

X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
k_list = np.array([1, 3, 5, 7, 9])
n_classes = 2
eps = 1e-6

NN = NearestNeighbors(n_neighbors=9, n_jobs=1)
NN.fit(X_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_test = np.array(data['X'])
    
    neighs_dist, neighs_idx = NN.kneighbors(X_test)
    
    start = time.perf_counter()
    results = []
    for i in range(len(X_test)):
        neighs_y = y_train[neighs_idx[i]]
        features = nearestneighborsfeats__get_features_for_one(
            neighs_dist[i], neighs_y, k_list, n_classes, eps
        )
        results.append(features)
    result = np.array(results)
    elapsed = time.perf_counter() - start
    
    return jsonify({
        'result': result.tolist(),
        'time_ms': elapsed * 1000,
        'version': 'optimized-test'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'optimized-test'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
