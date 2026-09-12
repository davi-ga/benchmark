from flask import Flask, request, jsonify
import time
import numpy as np
import sys

sys.path.insert(0, 'src')
from knn_feature import NearestNeighborsFeats

app = Flask(__name__)

model = NearestNeighborsFeats(n_jobs=1, k_list=[1, 3, 5, 7, 9], metric='euclidean')
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_test = np.array(data['X'])
    
    start = time.perf_counter()
    result = model.predict(X_test)
    elapsed = time.perf_counter() - start
    
    return jsonify({
        'result': result.tolist(),
        'time_ms': elapsed * 1000,
        'version': 'original'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'original'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
