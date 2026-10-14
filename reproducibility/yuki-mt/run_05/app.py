import numba
import sys
import numpy as np
from _numba_helpers import _sanitize_for_numba, _deep_to_list
sys.path.insert(0, 'src')
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
try:
    model.fit(X_train, y_train)
except NameError:
    pass
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8080)
    except NameError:
        pass