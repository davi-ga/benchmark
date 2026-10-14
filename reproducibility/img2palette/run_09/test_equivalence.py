import os
import sys
sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))

import math
import numpy as np
from src.image import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb, rgb2lab, lab2rgb

def test_rgb2xyz():
    rgb = np.array([[0.1, 0.5, 0.9], [0.0, 1.0, 0.0]])
    result1 = rgb2xyz(rgb)
    result2 = rgb2xyz(rgb)
    assert np.allclose(result1, result2)
    assert result1.shape == (2, 3)
    assert not np.any(np.isnan(result1))
    assert not np.any(np.isinf(result1))

def test_xyz2lab():
    xyz = np.array([[0.2, 0.3, 0.4], [0.9, 1.0, 1.1]])
    result1 = xyz2lab(xyz)
    result2 = xyz2lab(xyz)
    assert np.allclose(result1, result2)
    assert result1.shape == (2, 3)
    assert not np.any(np.isnan(result1))
    assert not np.any(np.isinf(result1))

def test_lab2xyz():
    lab = np.array([[50.0, 10.0, -10.0], [100.0, 0.0, 0.0]])
    result1 = lab2xyz(lab)
    result2 = lab2xyz(lab)
    assert np.allclose(result1, result2)
    assert result1.shape == (2, 3)
    assert not np.any(np.isnan(result1))
    assert not np.any(np.isinf(result1))

def test_xyz2rgb():
    xyz = np.array([[0.2, 0.3, 0.4], [0.9, 1.0, 1.1]])
    result1 = xyz2rgb(xyz)
    result2 = xyz2rgb(xyz)
    assert np.allclose(result1, result2)
    assert result1.shape == (2, 3)
    assert not np.any(np.isnan(result1))
    assert not np.any(np.isinf(result1))
    assert np.all(result1 >= 0.0)
    assert np.all(result1 <= 1.0)

def test_rgb2lab():
    rgb = np.array([[0.1, 0.5, 0.9], [0.0, 1.0, 0.0]])
    result1 = rgb2lab(rgb)
    result2 = rgb2lab(rgb)
    assert np.allclose(result1, result2)
    assert result1.shape == (2, 3)
    assert not np.any(np.isnan(result1))
    assert not np.any(np.isinf(result1))

def test_lab2rgb():
    lab = np.array([[50.0, 10.0, -10.0], [100.0, 0.0, 0.0]])
    result1 = lab2rgb(lab)
    result2 = lab2rgb(lab)
    assert np.allclose(result1, result2)
    assert result1.shape == (2, 3)
    assert not np.any(np.isnan(result1))
    assert not np.any(np.isinf(result1))
    assert np.all(result1 >= 0.0)
    assert np.all(result1 <= 1.0)
