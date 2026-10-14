import sys
import os
import numpy as np
import math

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

from src.image import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb, rgb2lab, lab2rgb


def test_rgb2xyz():
    rgb_arr = np.array([[0.1, 0.5, 0.9], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    res = rgb2xyz(rgb_arr)
    assert res.shape == (3, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()

def test_xyz2lab():
    xyz_arr = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.95047, 1.0, 1.08883]])
    res = xyz2lab(xyz_arr)
    assert res.shape == (3, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()

def test_lab2xyz():
    lab_arr = np.array([[50.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    res = lab2xyz(lab_arr)
    assert res.shape == (3, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()

def test_xyz2rgb():
    xyz_arr = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.95047, 1.0, 1.08883]])
    res = xyz2rgb(xyz_arr)
    assert res.shape == (3, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()

def test_rgb2lab():
    rgb_arr = np.array([[0.1, 0.5, 0.9], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    res = rgb2lab(rgb_arr)
    assert res.shape == (3, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()

def test_lab2rgb():
    lab_arr = np.array([[50.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    res = lab2rgb(lab_arr)
    assert res.shape == (3, 3)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
