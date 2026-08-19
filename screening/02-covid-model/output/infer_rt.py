import math
import numpy as np
import numba
from scipy import stats as sps
from scipy.special import gammaln
from _numba_helpers import _sanitize_for_numba, _deep_to_list
SQRT2PI = math.sqrt(2.0 * math.pi)

@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)], fastmath=True)
def normal_pdf(x, mean, std_deviation):
    u = (x - mean) / std_deviation
    return math.exp(-0.5 * u ** 2) / (SQRT2PI * std_deviation)

@numba.njit
def pdf_vector(x, loc, scale):
    array = np.empty((x.size, loc.size))
    for i, a in enumerate(x):
        for j, b in enumerate(loc):
            array[i, j] = normal_pdf(a, b, scale)
    return array

@numba.njit
def poisson_pmf_vectorized(k, lam):
    k_safe = np.maximum(k, 1e-10)
    lam_safe = np.maximum(lam, 1e-10)
    log_pmf = k * np.log(lam_safe) - lam_safe - gammaln(k + 1.0)
    result = np.exp(log_pmf)
    for i in range(lam.shape[0]):
        for j in range(lam.shape[1]):
            if lam[i, j] < 1e-10:
                if k[0, j] == 0:
                    result[i, j] = 1.0
                else:
                    result[i, j] = 0.0
    return result

@numba.njit
def compute_likelihoods(ts_values, r_list, serial_period):
    n = len(ts_values)
    m = len(r_list)
    ts_minus_1 = ts_values[:-1]
    lam = np.empty((m, n - 1))
    for i in range(m):
        for j in range(n - 1):
            lam[i, j] = ts_minus_1[j] * math.exp((r_list[i] - 1.0) / serial_period)
    ts_floor = np.floor(ts_values).astype(np.int64)
    ts_ceil = np.ceil(ts_values).astype(np.int64)
    ts_frac = ts_values - ts_floor
    k_floor = np.empty((1, n - 1), dtype=np.float64)
    k_ceil = np.empty((1, n - 1), dtype=np.float64)
    for j in range(n - 1):
        k_floor[0, j] = ts_floor[j + 1]
        k_ceil[0, j] = ts_ceil[j + 1]
    pmf_floor = poisson_pmf_vectorized(k_floor, lam)
    pmf_ceil = poisson_pmf_vectorized(k_ceil, lam)
    likelihoods = np.empty((m, n - 1))
    for i in range(m):
        for j in range(n - 1):
            frac = ts_frac[j + 1]
            likelihoods[i, j] = frac * pmf_ceil[i, j] + (1.0 - frac) * pmf_floor[i, j]
    return likelihoods

@numba.njit
def make_process_matrix(r_list, timeseries_scale, max_scaling_sigma, default_process_sigma, scale_sigma_from_count):
    a = max_scaling_sigma
    if timeseries_scale == 0:
        b = 1.0
    else:
        b = max(1.0, math.sqrt(scale_sigma_from_count / timeseries_scale))
    use_sigma = min(a, b) * default_process_sigma
    process_matrix = pdf_vector(r_list, r_list, use_sigma)
    sz = len(r_list)
    half = (sz - 1) / 2.0
    for row in range(sz):
        if row < half:
            for col in range(2 * row + 1, sz):
                process_matrix[row, col] = 0.0
        elif row > half:
            for col in range(0, sz - 2 * (sz - row)):
                process_matrix[row, col] = 0.0
    row_sums = np.empty(sz)
    for row in range(sz):
        s = 0.0
        for col in range(sz):
            s += process_matrix[row, col]
        row_sums[row] = s
    for row in range(sz):
        for col in range(sz):
            if row_sums[row] > 0:
                process_matrix[row, col] = process_matrix[row, col] / row_sums[row]
    return (use_sigma, process_matrix)

@numba.njit
def _highest_density_interval_jit(posteriors, r_list, ci):
    n_rt = posteriors.shape[0]
    n = posteriors.shape[1]
    ci_low = np.zeros(n)
    ci_high = np.zeros(n)
    for t in range(n):
        s = 0.0
        low_idx = 0
        high_idx = 0
        min_diff_low = 1000000000.0
        min_diff_high = 1000000000.0
        for r_i in range(n_rt):
            s += posteriors[r_i, t]
            diff_low = abs(s - (1.0 - ci))
            if diff_low < min_diff_low:
                min_diff_low = diff_low
                low_idx = r_i
            diff_high = abs(s - ci)
            if diff_high < min_diff_high:
                min_diff_high = diff_high
                high_idx = r_i
        ci_low[t] = r_list[low_idx]
        ci_high[t] = r_list[high_idx]
    return (ci_low, ci_high)

def highest_density_interval(posteriors, r_list, ci):
    return _deep_to_list(_highest_density_interval_jit(_sanitize_for_numba(posteriors), _sanitize_for_numba(r_list), _sanitize_for_numba(ci)))

@numba.njit
def _get_posteriors_jit(ts_values, r_list, serial_period, max_scaling_sigma, default_process_sigma, scale_sigma_from_count, prior0, reinit_prior):
    n = len(ts_values)
    if n == 0:
        return (np.empty((0, 0)), 0.0, 0)
    n_rt = len(r_list)
    likelihoods = compute_likelihoods(ts_values, r_list, serial_period)
    scale = np.median(ts_values)
    current_sigma, process_matrix = make_process_matrix(r_list, scale, max_scaling_sigma, default_process_sigma, scale_sigma_from_count)
    col_sums = np.zeros(n_rt)
    for i in range(n_rt):
        for j in range(n_rt):
            col_sums[j] += process_matrix[i, j]
    for i in range(n_rt):
        for j in range(n_rt):
            if col_sums[j] > 0:
                process_matrix[i, j] = process_matrix[i, j] / col_sums[j]
    posteriors = np.zeros((n_rt, n))
    for i in range(n_rt):
        posteriors[i, 0] = prior0[i]
    log_likelihood = 0.0
    scale = ts_values[0]
    for t in range(1, n):
        scale = 0.9 * scale + 0.1 * ts_values[t]
        current_sigma, process_matrix = make_process_matrix(r_list, scale, max_scaling_sigma, default_process_sigma, scale_sigma_from_count)
        col_sum = np.zeros(n_rt)
        for i in range(n_rt):
            for j in range(n_rt):
                col_sum[j] += process_matrix[i, j]
        for i in range(n_rt):
            for j in range(n_rt):
                if col_sum[j] > 0:
                    process_matrix[i, j] = process_matrix[i, j] / col_sum[j]
        current_prior = np.zeros(n_rt)
        for i in range(n_rt):
            for j in range(n_rt):
                current_prior[i] += process_matrix[i, j] * posteriors[j, t - 1]
        denominator = 0.0
        numerator = np.zeros(n_rt)
        for i in range(n_rt):
            val = likelihoods[i, t - 1] * current_prior[i]
            numerator[i] = val
            denominator += val
        if denominator == 0:
            for i in range(n_rt):
                posteriors[i, t] = reinit_prior[i]
        elif ts_values[t] <= 0.4:
            for i in range(n_rt):
                posteriors[i, t] = reinit_prior[i]
        else:
            for i in range(n_rt):
                posteriors[i, t] = numerator[i] / denominator
        if denominator > 0:
            log_likelihood += math.log(denominator)
    return (posteriors, log_likelihood, 0)

def get_posteriors(ts_values, r_list, serial_period, max_scaling_sigma, default_process_sigma, scale_sigma_from_count, prior0, reinit_prior):
    return _deep_to_list(_get_posteriors_jit(_sanitize_for_numba(ts_values), _sanitize_for_numba(r_list), _sanitize_for_numba(serial_period), _sanitize_for_numba(max_scaling_sigma), _sanitize_for_numba(default_process_sigma), _sanitize_for_numba(scale_sigma_from_count), _sanitize_for_numba(prior0), _sanitize_for_numba(reinit_prior)))

@numba.njit
def _evaluate_head_tail_suppression_jit(window_size, kernel_std):
    n = 2 * window_size
    ts = np.arange(0, n, dtype=np.float64)
    half_win = window_size // 2
    weights = np.empty(window_size)
    w_sum = 0.0
    for i in range(window_size):
        val = math.exp(-0.5 * (i - half_win) ** 2 / kernel_std ** 2)
        weights[i] = val
        w_sum += val
    for i in range(window_size):
        weights[i] /= w_sum
    smoothed = np.convolve(ts, weights)
    len_sm = len(smoothed)
    delta = np.zeros(len_sm - 1)
    for i in range(len_sm - 1):
        delta[i] = smoothed[i + 1] - smoothed[i]
    tail_len = window_size // 2 + window_size % 2
    tail_start = len(delta) - tail_len
    result = np.zeros(tail_len)
    for i in range(tail_len):
        val = delta[tail_start + i]
        if val > 1.0:
            result[i] = 1.0
        else:
            result[i] = val
    return result

def evaluate_head_tail_suppression(window_size, kernel_std):
    return _deep_to_list(_evaluate_head_tail_suppression_jit(_sanitize_for_numba(window_size), _sanitize_for_numba(kernel_std)))

@numba.njit
def _apply_tail_suppression_jit(rt_map, suppression_values, tail_suppression_correction):
    if tail_suppression_correction <= 0.0:
        return rt_map
    n = len(rt_map)
    res = np.zeros(n)
    for i in range(n):
        power = math.pow(suppression_values[i], tail_suppression_correction)
        res[i] = (rt_map[i] - 1.0) / power + 1.0
    return res

def apply_tail_suppression(rt_map, suppression_values, tail_suppression_correction):
    return _deep_to_list(_apply_tail_suppression_jit(_sanitize_for_numba(rt_map), _sanitize_for_numba(suppression_values), _sanitize_for_numba(tail_suppression_correction)))