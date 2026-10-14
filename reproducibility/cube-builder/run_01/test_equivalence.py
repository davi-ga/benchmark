import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))

from app import extract_qa_bits

def test_extract_qa_bits():
    band_data = np.array([5, 10, 15], dtype=np.int32)
    bit_location = 1
    result = extract_qa_bits(band_data, bit_location)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    for val in result:
        assert not math.isnan(val)
        assert not math.isinf(val)
