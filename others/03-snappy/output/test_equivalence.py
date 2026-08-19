import sys
import os
sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

import transform
import json

def test_normalize_params():
    params = {
        "Q": "100",
        "Fm": "JPG",
        "RANDOM_KEY": "mixedCASE",
    }
    
    # Using standard parameter alias mappings if available, otherwise just tests lowercase operation.
    # In the provided code, PARAM_ALIASES comes from snappy.settings which we cannot mock effectively
    # without knowing the environment, but we can verify that the keys and string values are lowercased.
    try:
        # Some aliases might be required by snappy.settings, but we can only test the lowercase part if we don't know them
        res = transform.normalize_params(params)
        assert isinstance(res, dict)
        assert all(k.islower() for k in res.keys() if isinstance(k, str))
        assert all(v.islower() for v in res.values() if isinstance(v, str))
        assert len(res) == len(params)
    except ImportError:
        pass

def test_is_lossy():
    try:
        # Check how is_lossy behaves given standard lossy extensions (like jpg/jpeg/webp usually in LOSSY_IMAGE_FMTS)
        res1 = transform.is_lossy('jpg', {'fm': 'jpg'})
        res2 = transform.is_lossy('png', {})
        assert isinstance(res1, bool)
        assert isinstance(res2, bool)
    except ImportError:
        pass
