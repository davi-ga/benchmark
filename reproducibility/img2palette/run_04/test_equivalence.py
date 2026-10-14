import os
import sys
sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))

import math
import numpy as np
from src.image import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb, rgb2lab, lab2rgb

def test_rgb2xyz():
    rgb = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]], dtype=np.float64)
    xyz = rgb2xyz(rgb)
    assert xyz.shape == (2, 3)
    assert not np.any(np.isnan(xyz))
    assert not np.any(np.isinf(xyz))

def test_xyz2lab():
    xyz = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]], dtype=np.float64)
    lab = xyz2lab(xyz)
    assert lab.shape == (2, 3)
    assert not np.any(np.isnan(lab))
    assert not np.any(np.isinf(lab))

def test_lab2xyz():
    lab = np.array([[50.0, 10.0, -10.0], [80.0, 5.0, 20.0]], dtype=np.float64)
    xyz = lab2xyz(lab)
    assert xyz.shape == (2, 3)
    assert not np.any(np.isnan(xyz))
    assert not np.any(np.isinf(xyz))

def test_xyz2rgb():
    xyz = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]], dtype=np.float64)
    rgb = xyz2rgb(xyz)
    assert rgb.shape == (2, 3)
    assert not np.any(np.isnan(rgb))
    assert not np.any(np.isinf(rgb))
    assert np.all(rgb >= 0)
    assert np.all(rgb <= 1)

def test_rgb2lab():
    rgb = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]], dtype=np.float64)
    lab = rgb2lab(rgb)
    assert lab.shape == (2, 3)
    assert not np.any(np.isnan(lab))
    assert not np.any(np.isinf(lab))

def test_lab2rgb():
    lab = np.array([[50.0, 10.0, -10.0], [80.0, 5.0, 20.0]], dtype=np.float64)
    rgb = lab2rgb(lab)
    assert rgb.shape == (2, 3)
    assert not np.any(np.isnan(rgb))
    assert not np.any(np.isinf(rgb))
    assert np.all(rgb >= 0)
    assert np.all(rgb <= 1)
