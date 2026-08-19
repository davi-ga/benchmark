import sys
import os
import numpy as np
import math

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

import image

def test_rgb2xyz():
    rgb_arr = np.array([[0.5, 0.2, 0.8], [0.1, 0.9, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    xyz = image.rgb2xyz(rgb_arr)
    assert xyz.shape == (4, 3)
    assert not np.isnan(xyz).any()
    assert not np.isinf(xyz).any()
    # test idempotence/determinism
    xyz2 = image.rgb2xyz(rgb_arr)
    np.testing.assert_array_equal(xyz, xyz2)

def test_xyz2lab():
    xyz_arr = np.array([[0.5, 0.5, 0.5], [0.1, 0.8, 0.2], [0.95047, 1.0, 1.08883], [0.0, 0.0, 0.0]])
    lab = image.xyz2lab(xyz_arr)
    assert lab.shape == (4, 3)
    assert not np.isnan(lab).any()
    assert not np.isinf(lab).any()
    lab2 = image.xyz2lab(xyz_arr)
    np.testing.assert_array_equal(lab, lab2)

def test_lab2xyz():
    lab_arr = np.array([[50.0, 20.0, -10.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    xyz = image.lab2xyz(lab_arr)
    assert xyz.shape == (3, 3)
    assert not np.isnan(xyz).any()
    assert not np.isinf(xyz).any()
    xyz2 = image.lab2xyz(lab_arr)
    np.testing.assert_array_equal(xyz, xyz2)

def test_xyz2rgb():
    xyz_arr = np.array([[0.5, 0.5, 0.5], [0.1, 0.8, 0.2], [0.95047, 1.0, 1.08883], [0.0, 0.0, 0.0]])
    rgb = image.xyz2rgb(xyz_arr)
    assert rgb.shape == (4, 3)
    assert not np.isnan(rgb).any()
    assert not np.isinf(rgb).any()
    assert np.all(rgb >= 0.0)
    assert np.all(rgb <= 1.0)
    rgb2 = image.xyz2rgb(xyz_arr)
    np.testing.assert_array_equal(rgb, rgb2)

def test_rgb2lab():
    rgb_arr = np.array([[0.5, 0.2, 0.8], [0.1, 0.9, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    lab = image.rgb2lab(rgb_arr)
    assert lab.shape == (4, 3)
    assert not np.isnan(lab).any()
    assert not np.isinf(lab).any()

def test_lab2rgb():
    lab_arr = np.array([[50.0, 20.0, -10.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rgb = image.lab2rgb(lab_arr)
    assert rgb.shape == (3, 3)
    assert not np.isnan(rgb).any()
    assert not np.isinf(rgb).any()
    assert np.all(rgb >= 0.0)
    assert np.all(rgb <= 1.0)

def test_get_hex_string():
    rgb_arr = np.array([255, 0, 128])
    hex_str = image.get_hex_string(rgb_arr)
    assert isinstance(hex_str, str)
    assert len(hex_str) == 6
    hex_str2 = image.get_hex_string(rgb_arr)
    assert hex_str == hex_str2

def test_compute_medoid():
    data = np.array([[1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [10.0, 10.0]])
    medoid = image.compute_medoid(data)
    assert medoid.shape == (2,)
    assert not np.isnan(medoid).any()
    assert not np.isinf(medoid).any()
    medoid2 = image.compute_medoid(data)
    np.testing.assert_array_equal(medoid, medoid2)

def test_get_cluster():
    centers = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels = np.array([0, 0, 1, 0, 2, 2, 0])
    sort_centers, sort_labels, sort_counts = image.get_cluster(centers, labels)
    assert len(sort_centers) == 3
    assert len(sort_counts) == 3
    assert len(sort_labels) == 7
    assert sort_counts[0] >= sort_counts[1] >= sort_counts[2]
    assert np.sum(sort_counts) == 7
