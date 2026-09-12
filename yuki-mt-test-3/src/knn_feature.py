import numba
import numpy as np
from typing import Tuple
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _fit_knn_jit(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (X_train, y_train)

def fit_knn(X_train: np.ndarray, y_train: np.ndarray):
    return _deep_to_list(_fit_knn_jit(_sanitize_for_numba(X_train), _sanitize_for_numba(y_train)))