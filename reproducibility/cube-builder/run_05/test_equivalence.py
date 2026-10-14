import sys
import os
import math
import pytest
import numpy as np

sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))

from app import extract_qa_bits


def test_extract_qa_bits_equivalence():
    band_data = np.array([255, 15, 0, 1024], dtype=np.int32)
    bit_location = 3
    
    result = extract_qa_bits(band_data, bit_location)
    
    assert len(result) == len(band_data)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    assert np.all(result >= 0)
    
    result2 = extract_qa_bits(band_data, bit_location)
    np.testing.assert_array_equal(result, result2)
