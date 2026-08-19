import os
import sys
sys.path.insert(0, os.environ.get("MODULE_DIR", "."))
import numpy as np
import pytest
import math
from maestro import *
from processing import *

def test_maestro_blend_stack_window():
    stack_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    stack_mask = np.array([[[False, True], [False, True]], [[True, False], [True, False]]], dtype=np.bool_)
    nodata = -9999
    res = blend_stack_window(stack_data, stack_mask, nodata, 2, 2, 2)
    assert res.shape == (2, 2)

def test_maestro_blend_median_window():
    stack_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    stack_mask = np.array([[[False, False], [False, False]], [[False, False], [False, False]]], dtype=np.bool_)
    nodata = -9999
    res = blend_median_window(stack_data, stack_mask, nodata, 2, 2, 2)
    assert res.shape == (2, 2)

def test_maestro_blend_clear_observation():
    stack_mask = np.array([[[False, True], [False, True]], [[True, False], [True, False]]], dtype=np.bool_)
    res = blend_clear_observation(stack_mask, 2, 2, 2)
    assert res.shape == (2, 2)
    assert np.all(res >= 0)

def test_processing_extract_qa_bits():
    band_data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    res = extract_qa_bits(band_data, 1)
    assert res.shape == (2, 2)

def test_processing_qa_cloud_confidence_apply():
    data = np.array([[1024, 2048], [4096, 8192]], dtype=np.int32)
    landsat_mask = np.array([[True, True], [True, True]], dtype=np.bool_)
    res = qa_cloud_confidence_apply(data, 1, 1, 1, 1, landsat_mask)
    assert res.shape == (2, 2)

def test_processing_compute_band_index_expression():
    band_arrays = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    expr_ops = np.array([0], dtype=np.int32)
    res = compute_band_index_expression(band_arrays, None, expr_ops, 0, 10000, -9999)
    assert res.shape == (2, 2)

def test_processing_compute_ndvi():
    nir = np.array([[500, 600], [700, 800]], dtype=np.int16)
    red = np.array([[100, 200], [300, 400]], dtype=np.int16)
    res = compute_ndvi(nir, red, -9999)
    assert res.shape == (2, 2)

def test_processing_blend_pixel_stack():
    stack_data = np.array([[[1, 2], [3, 4]]], dtype=np.int16)
    stack_mask = np.array([[[False, False], [False, False]]], dtype=np.bool_)
    stk, med, clear = blend_pixel_stack(stack_data, stack_mask, -9999, 1)
    assert stk.shape == (1, 2)
    assert med.shape == (1, 2)
    assert clear.shape == (1, 2)
