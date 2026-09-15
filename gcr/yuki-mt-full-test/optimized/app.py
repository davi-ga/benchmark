from flask import Flask, request, jsonify
import time
import numpy as np
import knn_feature

app = Flask(__name__)

from sklearn.neighbors import NearestNeighbors

print("Inicializando modelo sklearn...")

X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
k_list = np.array([1, 3, 5, 7, 9], dtype=np.int64)
n_classes = 2
eps = 1e-6

NN = NearestNeighbors(n_neighbors=max(k_list), metric='euclidean', n_jobs=1)
NN.fit(X_train)
print("Modelo sklearn treinado")

X_warmup = np.random.rand(1, 10)
neighs_dist, neighs_idx = NN.kneighbors(X_warmup)
try:
    _ = knn_feature.nearestneighborsfeats__get_features_for_one(
        X_warmup[0], 
        (neighs_dist, neighs_idx), 
        eps, 
        k_list, 
        n_classes, 
        y_train
    )
    print("Warmup Numba OK")
except Exception as e:
    print(f"Warmup Numba FALHOU: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_test = np.array(data['X'])
    
    start = time.perf_counter()
    neighs_dist, neighs_idx = NN.kneighbors(X_test)
    
    results = []
    for i in range(len(X_test)):
        result = knn_feature.nearestneighborsfeats__get_features_for_one(
            X_test[i],
            (neighs_dist[i:i+1], neighs_idx[i:i+1]),
            eps,
            k_list,
            n_classes,
            y_train
        )
        results.append(result)
    result = np.array(results)
    
    elapsed = time.perf_counter() - start
    
    return jsonify({
        'result': result.tolist(),
        'time_ms': elapsed * 1000,
        'version': 'optimized-full-test'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'optimized-full-test'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
