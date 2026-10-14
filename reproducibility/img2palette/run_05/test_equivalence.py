import os
import sys
import math
import numpy as np
import pytest

# Configure sys.path for direct imports
module_dir = os.environ.get('MODULE_DIR', '.')
sys.path.insert(0, module_dir)

from src.image import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb, rgb2lab, lab2rgb, get_cluster

def test_rgb2xyz():
    rgb_arr = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.2]])
    out = rgb2xyz(rgb_arr)
    assert out.shape == (2, 3)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.all(out >= 0)

def test_xyz2lab():
    xyz_arr = np.array([[0.2, 0.3, 0.4], [0.8, 0.9, 1.0]])
    out = xyz2lab(xyz_arr)
    assert out.shape == (2, 3)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

def test_lab2xyz():
    lab_arr = np.array([[50.0, 10.0, -10.0], [80.0, 0.0, 0.0]])
    out = lab2xyz(lab_arr)
    assert out.shape == (2, 3)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.all(out >= 0)

def test_xyz2rgb():
    xyz_arr = np.array([[0.2, 0.3, 0.4], [0.8, 0.9, 1.0]])
    out = xyz2rgb(xyz_arr)
    assert out.shape == (2, 3)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.all(out >= 0)
    assert np.all(out <= 1)

def test_rgb2lab():
    rgb_arr = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.2]])
    out = rgb2lab(rgb_arr)
    assert out.shape == (2, 3)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

def test_lab2rgb():
    lab_arr = np.array([[50.0, 10.0, -10.0], [80.0, 0.0, 0.0]])
    out = lab2rgb(lab_arr)
    assert out.shape == (2, 3)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.all(out >= 0)
    assert np.all(out <= 1)

def test_get_cluster():
    centers = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    labels = np.array([0, 1, 1, 2, 2, 2])
    sort_centers, sort_labels, sort_counts = get_cluster(centers, labels)
    assert sort_centers.shape == (3, 3)
    assert sort_labels.shape == (6,)
    assert sort_counts.shape == (3,)
    assert sort_counts[0] == 3
    assert sort_counts[1] == 2
    assert sort_counts[2] == 1
