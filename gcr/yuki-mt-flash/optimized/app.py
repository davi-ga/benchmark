from flask import Flask, request, jsonify
import time
import numpy as np
import sys

sys.path.insert(0, 'optimized/src')
import knn_feature

app = Flask(__name__)

print("🔥 Inicializando modelo (sem middleware)...")
print(f"📋 Funções disponíveis: {[f for f in dir(knn_feature) if not f.startswith('_') and callable(getattr(knn_feature, f, None))]}")

X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
k_list = np.array([1, 3, 5, 7, 9], dtype=np.int64)

model_state = knn_feature.nearestneighborsfeats_fit(X_train, y_train, k_list=k_list, metric='euclidean')
print(f"✅ Modelo treinado. Estado: {type(model_state)}")

X_warmup = np.random.rand(1, 10)
try:
    _ = knn_feature.nearestneighborsfeats_predict(model_state, X_warmup)
    print("✅ Warmup predict OK")
except Exception as e:
    print(f"❌ Warmup predict FALHOU: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_test = np.array(data['X'])

    start = time.perf_counter()
    result = knn_feature.nearestneighborsfeats_predict(model_state, X_test)
    elapsed = time.perf_counter() - start

    return jsonify({
        'result': np.asarray(result).tolist(),
        'time_ms': elapsed * 1000,
        'version': 'optimized-flash'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'optimized-flash'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
