import os
import sys
sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))
import numpy as np
import math
from src.image import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb, rgb2lab, lab2rgb

def test_rgb2xyz():
    rgb = np.array([[0.1, 0.5, 0.9], [0.8, 0.2, 0.4]])
    res = rgb2xyz(rgb)
    assert res.shape == (2, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()

def test_xyz2lab():
    xyz = np.array([[0.1, 0.5, 0.9], [0.8, 0.2, 0.4]])
    res = xyz2lab(xyz)
    assert res.shape == (2, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()

def test_lab2xyz():
    lab = np.array([[50.0, 10.0, -10.0], [80.0, -20.0, 30.0]])
    res = lab2xyz(lab)
    assert res.shape == (2, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()

def test_xyz2rgb():
    xyz = np.array([[0.1, 0.5, 0.9], [0.8, 0.2, 0.4]])
    res = xyz2rgb(xyz)
    assert res.shape == (2, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()

def test_rgb2lab():
    rgb = np.array([[0.1, 0.5, 0.9], [0.8, 0.2, 0.4]])
    res = rgb2lab(rgb)
    assert res.shape == (2, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()

def test_lab2rgb():
    lab = np.array([[50.0, 10.0, -10.0], [80.0, -20.0, 30.0]])
    res = lab2rgb(lab)
    assert res.shape == (2, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
