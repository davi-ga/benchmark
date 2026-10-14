import sys
import os
import numpy as np
import pytest
import math

# Ensure the module can be imported
sys.path.insert(0, os.environ.get('MODULE_DIR', '.'))

from src.image import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb, rgb2lab, lab2rgb

def test_rgb2xyz():
    rgb_arr = np.array([[0.1, 0.5, 0.9], [0.8, 0.2, 0.1]])
    # Warm-up
    res_wu = rgb2xyz(rgb_arr)
    res = rgb2xyz(rgb_arr)
    
    assert res.shape == rgb_arr.shape
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
    # Test idempotence
    np.testing.assert_allclose(res_wu, res)

def test_xyz2lab():
    xyz_arr = np.array([[0.2, 0.3, 0.4], [0.8, 0.7, 0.9]])
    res_wu = xyz2lab(xyz_arr)
    res = xyz2lab(xyz_arr)
    
    assert res.shape == xyz_arr.shape
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
    np.testing.assert_allclose(res_wu, res)

def test_lab2xyz():
    lab_arr = np.array([[50.0, 10.0, -10.0], [90.0, -5.0, 5.0]])
    res_wu = lab2xyz(lab_arr)
    res = lab2xyz(lab_arr)
    
    assert res.shape == lab_arr.shape
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
    np.testing.assert_allclose(res_wu, res)

def test_xyz2rgb():
    xyz_arr = np.array([[0.2, 0.3, 0.4], [0.8, 0.7, 0.9]])
    res_wu = xyz2rgb(xyz_arr)
    res = xyz2rgb(xyz_arr)
    
    assert res.shape == xyz_arr.shape
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
    assert np.all(res >= 0) and np.all(res <= 1)
    np.testing.assert_allclose(res_wu, res)

def test_rgb2lab():
    rgb_arr = np.array([[0.1, 0.5, 0.9], [0.8, 0.2, 0.1]])
    res_wu = rgb2lab(rgb_arr)
    res = rgb2lab(rgb_arr)
    
    assert res.shape == rgb_arr.shape
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
    np.testing.assert_allclose(res_wu, res)

def test_lab2rgb():
    lab_arr = np.array([[50.0, 10.0, -10.0], [90.0, -5.0, 5.0]])
    res_wu = lab2rgb(lab_arr)
    res = lab2rgb(lab_arr)
    
    assert res.shape == lab_arr.shape
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
    assert np.all(res >= 0) and np.all(res <= 1)
    np.testing.assert_allclose(res_wu, res)
