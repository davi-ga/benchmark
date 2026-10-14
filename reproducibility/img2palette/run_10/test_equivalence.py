import sys
import os
import numpy as np
import math
import pytest

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))
from src.image import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb, rgb2lab, lab2rgb

def test_rgb2xyz():
    rgb = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.9]])
    out = rgb2xyz(rgb)
    assert out.shape == rgb.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

def test_xyz2lab():
    xyz = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.9]])
    out = xyz2lab(xyz)
    assert out.shape == xyz.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

def test_lab2xyz():
    lab = np.array([[50.0, 10.0, -10.0], [20.0, 0.0, 0.0]])
    out = lab2xyz(lab)
    assert out.shape == lab.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

def test_xyz2rgb():
    xyz = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.9]])
    out = xyz2rgb(xyz)
    assert out.shape == xyz.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert (out >= 0.0).all() and (out <= 1.0).all()

def test_rgb2lab():
    rgb = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.9]])
    out = rgb2lab(rgb)
    assert out.shape == rgb.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

def test_lab2rgb():
    lab = np.array([[50.0, 10.0, -10.0], [20.0, 0.0, 0.0]])
    out = lab2rgb(lab)
    assert out.shape == lab.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert (out >= 0.0).all() and (out <= 1.0).all()
