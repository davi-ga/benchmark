from flask import Flask, request, jsonify
import time
import sys
from io import BytesIO
from PIL import Image
import requests
import numpy as np

sys.path.insert(0, 'optimized')
from image import get_palette

app = Flask(__name__)

@app.route('/palette', methods=['POST'])
def palette():
    data = request.json
    image_url = data.get('image_url')
    k = data.get('n_colors', 5)
    
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        return jsonify({'error': f'Failed to load image: {str(e)}'}), 400
    
    start = time.perf_counter()
    im_output, hex_colors = get_palette(img, k)
    elapsed = time.perf_counter() - start
    
    return jsonify({
        'palette': hex_colors.tolist() if hasattr(hex_colors, 'tolist') else list(hex_colors),
        'time_ms': elapsed * 1000,
        'version': 'optimized'
    })

@app.route('/palette-local', methods=['POST'])
def palette_local():
    """Test with a generated image instead of downloading"""
    data = request.json
    k = data.get('n_colors', 5)
    size = data.get('size', 100)
    
    # Generate a random image
    img_array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    
    start = time.perf_counter()
    im_output, hex_colors = get_palette(img, k)
    elapsed = time.perf_counter() - start
    
    return jsonify({
        'palette': hex_colors.tolist() if hasattr(hex_colors, 'tolist') else list(hex_colors),
        'time_ms': elapsed * 1000,
        'version': 'optimized'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'optimized'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
