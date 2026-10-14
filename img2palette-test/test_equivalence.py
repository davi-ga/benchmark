import sys
import os
import numpy as np
import math

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))
from src.image import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb, rgb2lab, lab2rgb, get_cluster


def test_rgb2xyz():
    rgb_arr = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float64)
    result = rgb2xyz(rgb_arr)
    assert result.shape == rgb_arr.shape
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_xyz2lab():
    xyz_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    result = xyz2lab(xyz_arr)
    assert result.shape == xyz_arr.shape
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_lab2xyz():
    lab_arr = np.array([[50.0, 10.0, -10.0], [70.0, -20.0, 30.0]], dtype=np.float64)
    result = lab2xyz(lab_arr)
    assert result.shape == lab_arr.shape
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_xyz2rgb():
    xyz_arr = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float64)
    result = xyz2rgb(xyz_arr)
    assert result.shape == xyz_arr.shape
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
    assert np.all(result >= 0) and np.all(result <= 1)


def test_rgb2lab():
    rgb_arr = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float64)
    result = rgb2lab(rgb_arr)
    assert result.shape == rgb_arr.shape
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_lab2rgb():
    lab_arr = np.array([[50.0, 10.0, -10.0], [70.0, -20.0, 30.0]], dtype=np.float64)
    result = lab2rgb(lab_arr)
    assert result.shape == lab_arr.shape
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
    assert np.all(result >= 0) and np.all(result <= 1)


def test_get_cluster():
    centers = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    labels = np.array([0, 1, 0, 1, 1], dtype=np.int32)
    result = get_cluster(centers, labels)
    assert len(result) == 3
    sorted_centers, sorted_labels, sorted_counts = result
    assert sorted_centers.shape == centers.shape
    assert len(sorted_counts) == len(centers)
    # Check monotonic counts descending
    assert sorted_counts[0] >= sorted_counts[-1]
