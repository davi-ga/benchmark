import math
import numpy as np
import numba
from scipy import stats as sps
from scipy.special import gammaln


SQRT2PI = math.sqrt(2.0 * math.pi)


@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)], fastmath=True)
def normal_pdf(x, mean, std_deviation):
    u = (x - mean) / std_deviation
    return math.exp(-0.5 * u ** 2) / (SQRT2PI * std_deviation)


@numba.njit(fastmath=True)
def pdf_vector(x, loc, scale):
    array = np.empty((x.size, loc.size))
    for i, a in enumerate(x):
        for j, b in enumerate(loc):
            array[i, j] = normal_pdf(a, b, scale)
    return array


def poisson_pmf_vectorized(k, lam):
    k_safe = np.maximum(k, 1e-10)
    lam_safe = np.maximum(lam, 1e-10)
    log_pmf = k * np.log(lam_safe) - lam_safe - gammaln(k + 1.0)
    result = np.exp(log_pmf)
    result = np.where(lam < 1e-10, np.where(k == 0, 1.0, 0.0), result)
    return result


def compute_likelihoods(ts_values, r_list, serial_period):
    ts_minus_1 = ts_values[:-1]
    lam = ts_minus_1[np.newaxis, :] * np.exp((r_list[:, np.newaxis] - 1.0) / serial_period)
    ts_floor = np.floor(ts_values).astype(np.int64)
    ts_ceil = np.ceil(ts_values).astype(np.int64)
    ts_frac = ts_values - ts_floor
    k_floor = ts_floor[1:]
    k_ceil = ts_ceil[1:]
    frac = ts_frac[1:]
    pmf_floor = poisson_pmf_vectorized(k_floor[np.newaxis, :], lam)
    pmf_ceil = poisson_pmf_vectorized(k_ceil[np.newaxis, :], lam)
    likelihoods = frac[np.newaxis, :] * pmf_ceil + (1.0 - frac[np.newaxis, :]) * pmf_floor
    return likelihoods


def make_process_matrix(r_list, timeseries_scale, max_scaling_sigma,
                        default_process_sigma, scale_sigma_from_count):
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
            process_matrix[row, 2 * row + 1:sz] = 0.0
        elif row > half:
            process_matrix[row, 0:sz - 2 * (sz - row)] = 0.0
    row_sums = process_matrix.sum(axis=1)
    for row in range(sz):
        process_matrix[row] = process_matrix[row] / row_sums[row]
    return use_sigma, process_matrix


def highest_density_interval(posteriors, r_list, ci):
    posterior_cdfs = np.cumsum(posteriors, axis=0)
    low_idx_list = np.argmin(np.abs(posterior_cdfs - (1.0 - ci)), axis=0)
    high_idx_list = np.argmin(np.abs(posterior_cdfs - ci), axis=0)
    ci_low = r_list[low_idx_list]
    ci_high = r_list[high_idx_list]
    return ci_low, ci_high


def get_posteriors(ts_values, r_list, serial_period, max_scaling_sigma,
                   default_process_sigma, scale_sigma_from_count):
    n = len(ts_values)
    if n == 0:
        return np.empty((0, 0)), 0.0, 0
    n_rt = len(r_list)
    likelihoods = compute_likelihoods(ts_values, r_list, serial_period)
    median_val = np.median(ts_values)
    current_sigma, process_matrix = make_process_matrix(
        r_list, median_val, max_scaling_sigma, default_process_sigma, scale_sigma_from_count
    )
    col_sums = process_matrix.sum(axis=0)
    process_matrix = process_matrix / col_sums[np.newaxis, :]
    prior0 = sps.gamma(a=2).pdf(r_list)
    prior0 = prior0 / prior0.sum()
    reinit_prior = sps.gamma(a=2).pdf(r_list)
    reinit_prior = reinit_prior / reinit_prior.sum()
    posteriors = np.zeros((n_rt, n))
    posteriors[:, 0] = prior0
    log_likelihood = 0.0
    scale = ts_values[0]
    for t in range(1, n):
        scale = 0.9 * scale + 0.1 * ts_values[t]
        current_sigma, process_matrix = make_process_matrix(
            r_list, scale, max_scaling_sigma, default_process_sigma, scale_sigma_from_count
        )
        col_sum = process_matrix.sum(axis=0)
        process_matrix = process_matrix / col_sum[np.newaxis, :]
        current_prior = process_matrix @ posteriors[:, t - 1]
        numerator = likelihoods[:, t - 1] * current_prior
        denominator = np.sum(numerator)
        if denominator == 0:
            posteriors[:, t] = reinit_prior
        elif ts_values[t] <= 0.4:
            posteriors[:, t] = reinit_prior
        else:
            posteriors[:, t] = numerator / denominator
        log_likelihood += np.log(denominator)
    return posteriors, log_likelihood, 0


def evaluate_head_tail_suppression(window_size, kernel_std):
    n = 2 * window_size
    ts = np.arange(0, n, dtype=np.float64)
    half_win = window_size // 2
    weights = np.exp(-0.5 * (np.arange(window_size) - half_win) ** 2 / kernel_std ** 2)
    weights = weights / weights.sum()
    smoothed = np.convolve(ts, weights, mode='same')
    delta = np.diff(smoothed)
    tail_len = math.ceil(window_size / 2)
    tail_start = len(delta) - tail_len
    tail = delta[tail_start:]
    result = np.minimum(tail, 1.0)
    return result


def apply_tail_suppression(rt_map, suppression_values, tail_suppression_correction):
    if tail_suppression_correction <= 0.0:
        return rt_map
    power = np.power(suppression_values, tail_suppression_correction)
    return (rt_map - 1.0) / power + 1.0
