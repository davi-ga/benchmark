import numba
import numpy as np
import sys
from _numba_helpers import _sanitize_for_numba, _deep_to_list
sys.path.insert(0, 'src')
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
if __name__ == '__main__':
    pass