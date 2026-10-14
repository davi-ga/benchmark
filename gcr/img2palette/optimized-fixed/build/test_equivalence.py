import os
import sys
sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))

import numpy as np
import math

from src.image import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb, rgb2lab, lab2rgb

def test_rgb2xyz():
    rgb_arr = np.array([[0.5, 0.2, 0.8], [0.1, 0.9, 0.3]], dtype=np.float64)
    result = rgb2xyz(rgb_arr)
    assert result.shape == (2, 3)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
    assert np.all(result >= 0.0)

def test_xyz2lab():
    xyz_arr = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.3]], dtype=np.float64)
    result = xyz2lab(xyz_arr)
    assert result.shape == (2, 3)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()

def test_lab2xyz():
    lab_arr = np.array([[50.0, 10.0, -10.0], [90.0, -20.0, 30.0]], dtype=np.float64)
    result = lab2xyz(lab_arr)
    assert result.shape == (2, 3)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
    assert np.all(result >= 0.0)

def test_xyz2rgb():
    xyz_arr = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.3]], dtype=np.float64)
    result = xyz2rgb(xyz_arr)
    assert result.shape == (2, 3)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)

def test_rgb2lab():
    rgb_arr = np.array([[0.5, 0.2, 0.8], [0.1, 0.9, 0.3]], dtype=np.float64)
    result = rgb2lab(rgb_arr)
    assert result.shape == (2, 3)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()

def test_lab2rgb():
    lab_arr = np.array([[50.0, 10.0, -10.0], [90.0, -20.0, 30.0]], dtype=np.float64)
    result = lab2rgb(lab_arr)
    assert result.shape == (2, 3)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
