from flask import Flask, request, jsonify
import time
import numpy as np
import sys

sys.path.insert(0, 'optimized')
from processing import extract_qa_bits, compute_band_index_expression

app = Flask(__name__)

@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    band_data = np.array(data['band_data'], dtype=np.uint16)
    bit_location = int(data['bit_location'])  # Escalar, não array
    
    start = time.perf_counter()
    result = extract_qa_bits(band_data, bit_location)
    elapsed = time.perf_counter() - start
    
    return jsonify({
        'result': result.tolist() if hasattr(result, 'tolist') else list(result),
        'time_ms': elapsed * 1000,
        'version': 'optimized'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'optimized'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
