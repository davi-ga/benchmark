import numba
import numpy as np
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _extract_qa_bits_jit(band_data, bit_location):
    """Exact copy of processing.py:extract_qa_bits for fair benchmark."""
    return band_data & 1 << bit_location

def extract_qa_bits(band_data, bit_location):
    return _deep_to_list(_extract_qa_bits_jit(_sanitize_for_numba(np.array(band_data) if isinstance(band_data, list) else band_data), _sanitize_for_numba(np.array(bit_location) if isinstance(bit_location, list) else bit_location)))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)