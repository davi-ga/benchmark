from flask import Flask, request, jsonify
import time
import numpy as np
import sys

sys.path.insert(0, 'src')
from knn_feature import DummyNNF

app = Flask(__name__)

print("🔥 Inicializando modelo DummyNNF...")
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
k_list = [1, 3, 5, 7, 9]
n_classes = 2

model = DummyNNF(k_list=k_list, metric='euclidean', n_classes=n_classes)
model.fit(X_train, y_train)
print("✅ Modelo treinado")

X_warmup = np.random.rand(1, 10)
try:
    _ = model.predict(X_warmup)
    print("✅ Warmup predict OK")
except Exception as e:
    print(f"❌ Warmup predict FALHOU: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_test = np.array(data['X'])
    
    start = time.perf_counter()
    result = model.predict(X_test)
    elapsed = time.perf_counter() - start
    
    return jsonify({
        'result': np.asarray(result).tolist(),
        'time_ms': elapsed * 1000,
        'version': 'optimized-pro-2'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'optimized-pro-2'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
