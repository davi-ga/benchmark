import numba
import numpy as np
import sys
import time
import requests
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
sys.path.insert(0, 'src')
from image import get_palette
from _numba_helpers import _sanitize_for_numba, _deep_to_list
app = Flask(__name__)

@app.route('/palette', methods=['POST'])
def palette():
    data = request.json
    image_url = data.get('image_url')
    k = data.get('n_colors', 5)
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    start = time.perf_counter()
    im_output, hex_colors = get_palette(img, k)
    elapsed = time.perf_counter() - start
    return jsonify({'palette': hex_colors.tolist() if hasattr(hex_colors, 'tolist') else list(hex_colors), 'time_ms': elapsed * 1000, 'version': 'original'})

@app.route('/palette-local', methods=['POST'])
def palette_local():
    """Test with a generated image instead of downloading"""
    data = request.json
    k = data.get('n_colors', 5)
    size = data.get('size', 100)
    img_array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    start = time.perf_counter()
    im_output, hex_colors = get_palette(img, k)
    elapsed = time.perf_counter() - start
    return jsonify({'palette': hex_colors.tolist() if hasattr(hex_colors, 'tolist') else list(hex_colors), 'time_ms': elapsed * 1000, 'version': 'original'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'original'})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)