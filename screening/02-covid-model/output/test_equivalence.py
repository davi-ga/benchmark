import os
import sys
import math
import numpy as np
import pytest

# Ensure the correct directory is in sys.path
sys.path.insert(0, os.environ.get("MODULE_DIR", "."))

import infer_rt
import seir_model


def test_normal_pdf():
    val = infer_rt.normal_pdf(0.0, 0.0, 1.0)
    assert not math.isnan(val)
    assert val > 0
    assert val < 1.0  # ~0.3989


def test_pdf_vector():
    x = np.array([0.0, 1.0, 2.0])
    loc = np.array([0.0, 1.0])
    scale = 1.0
    result = infer_rt.pdf_vector(x, loc, scale)
    assert result.shape == (3, 2)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_poisson_pmf_vectorized():
    k = np.array([[0, 1, 2]])
    lam = np.array([[1.0, 2.0, 3.0]])
    result = infer_rt.poisson_pmf_vectorized(k, lam)
    assert result.shape == (1, 3)
    assert np.all(result >= 0)
    assert np.all(result <= 1.0)
    assert not np.any(np.isnan(result))


def test_compute_likelihoods():
    ts = np.array([10.0, 15.0, 20.0])
    r_list = np.array([0.5, 1.0, 1.5])
    result = infer_rt.compute_likelihoods(ts, r_list, 4.0)
    assert result.shape == (2, 3)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_make_process_matrix():
    r_list = np.linspace(0.5, 2.0, 10)
    use_sigma, matrix = infer_rt.make_process_matrix(r_list, 100, 0.1, 0.05, 50)
    assert use_sigma > 0
    assert matrix.shape == (10, 10)
    assert np.all(matrix >= 0)
    # Each row should sum to 1
    row_sums = matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_highest_density_interval():
    posteriors = np.array([[0.1, 0.8, 0.1], [0.2, 0.6, 0.2]]).T
    r_list = np.array([0.5, 1.0, 1.5])
    low, high = infer_rt.highest_density_interval(posteriors, r_list, 0.9)
    assert len(low) == 2
    assert len(high) == 2
    assert np.all(low <= high)


def test_get_posteriors():
    ts = np.array([10.0, 12.0, 15.0])
    r_list = np.linspace(0.5, 2.0, 10)
    post, log_lik, _ = infer_rt.get_posteriors(ts, r_list, 4.0, 0.1, 0.05, 50)
    assert post.shape == (10, 3)
    assert np.all(post >= 0)
    assert not np.isnan(log_lik)


def test_evaluate_head_tail_suppression():
    result = infer_rt.evaluate_head_tail_suppression(10, 2.0)
    assert len(result) == 5
    assert np.all(result <= 1.0)
    assert not np.any(np.isnan(result))


def test_apply_tail_suppression():
    rt_map = np.array([1.5, 2.0])
    suppression = np.array([0.9, 0.8])
    result = infer_rt.apply_tail_suppression(rt_map, suppression, 1.0)
    assert result.shape == (2,)
    assert not np.any(np.isnan(result))


def test_derivative():
    t = np.array([1.0, 3.0, 6.0])
    res = seir_model.derivative(t)
    assert len(res) == 3
    assert res[0] == 0.0
    assert res[1] == 2.0
    assert res[2] == 3.0


def test_build_params_array():
    params = seir_model.build_params_array(
        1000.0, 2.5, 1.5, 0.2, 0.1, 0.1, 0.5, 0.5,
        0.1, 0.05, 5.0, 8.0, 10.0, 14.0, 0.5,
        0.4, 0.2, 0.8, 0.6, 0.9, 100, 20
    )
    assert len(params) == seir_model.N_PARAMS
    assert params[seir_model.IDX_N] == 1000.0
    assert not np.any(np.isnan(params))


def test_seir_time_step():
    params = np.ones(seir_model.N_PARAMS)
    y = np.ones(12)
    res = seir_model.seir_time_step(y, 0.0, 1.0, params)
    assert len(res) == 12
    assert not np.any(np.isnan(res))


def test_run_seir():
    t_list = np.linspace(0, 10, 11)
    y0 = np.zeros(12)
    y0[0] = 1000.0
    y0[3] = 10.0
    params = np.ones(seir_model.N_PARAMS)
    res = seir_model.run_seir(t_list, y0, 1.0, params)
    assert res.shape == (17, 11)
    assert not np.any(np.isnan(res))


def test_steady_state_ratios():
    params = np.ones(seir_model.N_PARAMS)
    res = seir_model.steady_state_ratios(1.5, 1.0, 10000.0, params)
    assert len(res) == 7
    assert res[1] == 1.0
    assert not np.any(np.isnan(res))
