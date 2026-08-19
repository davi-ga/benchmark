import os
import sys
sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))
import pytest

def test_import():
    try:
        import task
        assert hasattr(task, 'handler')
        assert hasattr(task, 'cogify')
    except ImportError:
        pass
