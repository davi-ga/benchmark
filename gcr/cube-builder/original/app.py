from flask import Flask, request, jsonify
import time
import numpy as np

app = Flask(__name__)

def extract_qa_bits(band_data, bit_location):
    """Exact copy of processing.py:extract_qa_bits for fair benchmark."""
    return band_data & (1 << bit_location)

@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    band_data = np.array(data['band_data'], dtype=np.uint16)
    bit_location = int(data['bit_location'])
    
    start = time.perf_counter()
    result = extract_qa_bits(band_data, bit_location)
    elapsed = time.perf_counter() - start
    
    return jsonify({
        'result': result.tolist() if hasattr(result, 'tolist') else list(result),
        'time_ms': elapsed * 1000,
        'version': 'original'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'original'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
