import os
import sys
sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

import numpy as np
from image import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb, rgb2lab, lab2rgb, cluster_kmeans, compute_medoid, get_cluster, get_hex_string

def test_rgb_to_xyz():
    rgb = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.9]])
    xyz = rgb2xyz(rgb)
    assert xyz.shape == (2, 3)
    assert not np.any(np.isnan(xyz))
    assert not np.any(np.isinf(xyz))

def test_xyz_to_lab():
    xyz = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.9]])
    lab = xyz2lab(xyz)
    assert lab.shape == (2, 3)
    assert not np.any(np.isnan(lab))

def test_lab_to_xyz():
    lab = np.array([[50.0, 10.0, -10.0], [90.0, -5.0, 5.0]])
    xyz = lab2xyz(lab)
    assert xyz.shape == (2, 3)
    assert not np.any(np.isnan(xyz))

def test_xyz_to_rgb():
    xyz = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.9]])
    rgb = xyz2rgb(xyz)
    assert rgb.shape == (2, 3)
    assert np.all(rgb >= 0.0)
    assert np.all(rgb <= 1.0)

def test_rgb_to_lab():
    rgb = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.9]])
    lab = rgb2lab(rgb)
    assert lab.shape == (2, 3)

def test_lab_to_rgb():
    lab = np.array([[50.0, 10.0, -10.0], [90.0, -5.0, 5.0]])
    rgb = lab2rgb(lab)
    assert rgb.shape == (2, 3)
    assert np.all(rgb >= 0.0)
    assert np.all(rgb <= 1.0)

def test_cluster_kmeans():
    data = np.random.rand(20, 3)
    n_clusters = 3
    centers, labels = cluster_kmeans(data, n_clusters)
    assert centers.shape == (n_clusters, 3)
    assert len(labels) == 20
    assert np.all(labels >= 0)
    assert np.all(labels < n_clusters)

def test_compute_medoid():
    data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
    medoid = compute_medoid(data)
    assert medoid.shape == (3,)

def test_get_cluster():
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    labels = np.array([0, 0, 0, 1, 1, 2])
    sort_centers, sort_labels, sort_counts = get_cluster(centers, labels)
    assert sort_centers.shape == (3, 3)
    assert len(sort_labels) == 6
    assert len(sort_counts) == 3
    assert sort_counts[0] >= sort_counts[1]
    assert sort_counts[1] >= sort_counts[2]

def test_get_hex_string():
    rgb = np.array([255, 128, 0])
    hex_str = get_hex_string(rgb)
    assert len(hex_str) == 6
    assert type(hex_str) is str
