import numba
import sys
import time
import numpy as np
from flask import Flask, request, jsonify
from knn_feature import get_test_feature
from _numba_helpers import _sanitize_for_numba, _deep_to_list
sys.path.insert(0, 'src')
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': 'refactored'})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)