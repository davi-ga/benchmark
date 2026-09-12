import numba
import numpy as np
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _extract_qa_bits_jit(band_data, bit_location):
    return band_data & 1 << bit_location

def extract_qa_bits(band_data, bit_location):
    return _deep_to_list(_extract_qa_bits_jit(_sanitize_for_numba(band_data), _sanitize_for_numba(bit_location)))

@numba.njit
def _extract_qa_bits_range_jit(qa_data, start_offset, end_offset):
    return (qa_data >> start_offset) - (qa_data >> end_offset << 2)

def extract_qa_bits_range(qa_data, start_offset, end_offset):
    return _deep_to_list(_extract_qa_bits_range_jit(_sanitize_for_numba(qa_data), _sanitize_for_numba(start_offset), _sanitize_for_numba(end_offset)))

@numba.njit
def qa_cloud_confidence_apply(data, cloud_rule, cloud_shadow_rule, snow_rule, cirrus_rule, landsat_8_mask):
    result_mask = np.zeros((data.shape[0], data.shape[1]), dtype=np.bool_)
    for _i in range(data.shape[0]):
        for _j in range(data.shape[1]):
            qa = np.int32(data[_i, _j])
            is_masked = False
            if cloud_rule == 1:
                cloud_arr = (qa >> 8) - (qa >> 10 << 2)
                if cloud_arr >= 2:
                    is_masked = True
            elif cloud_rule == 2:
                cloud_arr = (qa >> 8) - (qa >> 10 << 2)
                if cloud_arr >= 1:
                    is_masked = True
            elif cloud_rule == 3:
                cloud_arr = (qa >> 8) - (qa >> 10 << 2)
                if cloud_arr == 3:
                    is_masked = True
            if cloud_shadow_rule == 1:
                shadow_arr = (qa >> 10) - (qa >> 12 << 2)
                if shadow_arr >= 2:
                    is_masked = True
            elif cloud_shadow_rule == 2:
                shadow_arr = (qa >> 10) - (qa >> 12 << 2)
                if shadow_arr >= 1:
                    is_masked = True
            if snow_rule == 1:
                snow_arr = (qa >> 12) - (qa >> 14 << 2)
                if snow_arr >= 2:
                    is_masked = True
            elif snow_rule == 2:
                snow_arr = (qa >> 12) - (qa >> 14 << 2)
                if snow_arr >= 1:
                    is_masked = True
            if cirrus_rule >= 0 and landsat_8_mask[_i, _j]:
                cirrus_arr = (qa >> 14) - (qa >> 16 << 2)
                if cirrus_rule == 1 and cirrus_arr >= 2:
                    is_masked = True
                elif cirrus_rule == 2 and cirrus_arr >= 1:
                    is_masked = True
            result_mask[_i, _j] = is_masked
    return result_mask

@numba.njit
def _get_qa_mask_nodata_jit(data, nodata):
    res = np.zeros((data.shape[0], data.shape[1]), dtype=np.bool_)
    for _i in range(data.shape[0]):
        for _j in range(data.shape[1]):
            res[_i, _j] = data[_i, _j] == nodata
    return res

def get_qa_mask_nodata(data, nodata):
    return _deep_to_list(_get_qa_mask_nodata_jit(_sanitize_for_numba(data), _sanitize_for_numba(nodata)))

@numba.njit
def _get_qa_mask_clear_jit(data, clear_data_arr, not_clear_data_arr, nodata, cloud_rule, cloud_shadow_rule, snow_rule, cirrus_rule, landsat_8_mask_arr):
    final_mask = np.zeros((data.shape[0], data.shape[1]), dtype=np.bool_)
    conf_mask = qa_cloud_confidence_apply(data, cloud_rule, cloud_shadow_rule, snow_rule, cirrus_rule, landsat_8_mask_arr)
    for _i in range(data.shape[0]):
        for _j in range(data.shape[1]):
            qa = np.int32(data[_i, _j])
            if qa == nodata or conf_mask[_i, _j]:
                final_mask[_i, _j] = True
                continue
            is_not_clear = False
            for k in range(not_clear_data_arr.shape[0]):
                if qa & 1 << not_clear_data_arr[k]:
                    is_not_clear = True
            is_clear = False
            for k in range(clear_data_arr.shape[0]):
                if qa & 1 << clear_data_arr[k]:
                    is_clear = True
            if is_not_clear or not is_clear:
                final_mask[_i, _j] = True
    return final_mask

def get_qa_mask_clear(data, clear_data_arr, not_clear_data_arr, nodata, cloud_rule, cloud_shadow_rule, snow_rule, cirrus_rule, landsat_8_mask_arr):
    return _deep_to_list(_get_qa_mask_clear_jit(_sanitize_for_numba(data), _sanitize_for_numba(clear_data_arr), _sanitize_for_numba(not_clear_data_arr), _sanitize_for_numba(nodata), _sanitize_for_numba(cloud_rule), _sanitize_for_numba(cloud_shadow_rule), _sanitize_for_numba(snow_rule), _sanitize_for_numba(cirrus_rule), _sanitize_for_numba(landsat_8_mask_arr)))

@numba.njit
def _qa_statistics_compute_jit(raster, nodata, clear_data_arr, not_clear_data_arr):
    clear_count = np.int64(0)
    not_clear_count = np.int64(0)
    total_pixels = np.int64(raster.shape[0] * raster.shape[1])
    for _i in range(raster.shape[0]):
        for _j in range(raster.shape[1]):
            val = np.int32(raster[_i, _j])
            for k in range(clear_data_arr.shape[0]):
                if val == clear_data_arr[k]:
                    clear_count += 1
            for k in range(not_clear_data_arr.shape[0]):
                if val == not_clear_data_arr[k]:
                    not_clear_count += 1
    image_area = clear_count + not_clear_count
    not_clear_ratio = np.float64(100.0)
    if image_area != 0:
        not_clear_ratio = np.round(100.0 * not_clear_count / image_area, 2)
    efficacy = np.float64(0.0)
    if total_pixels > 0:
        efficacy = np.round(100.0 * clear_count / total_pixels, 2)
    return (efficacy, not_clear_ratio)

def qa_statistics_compute(raster, nodata, clear_data_arr, not_clear_data_arr):
    return _deep_to_list(_qa_statistics_compute_jit(_sanitize_for_numba(raster), _sanitize_for_numba(nodata), _sanitize_for_numba(clear_data_arr), _sanitize_for_numba(not_clear_data_arr)))

@numba.njit
def _rescale_raster_uint8_jit(raster, scale_factor):
    result_uint8 = np.zeros((raster.shape[0], raster.shape[1]), dtype=np.uint8)
    for _i in range(raster.shape[0]):
        for _j in range(raster.shape[1]):
            val = np.float32(raster[_i, _j])
            if val > 0.0:
                res = val / scale_factor * 255.0
                if res < 0.0:
                    res = 0.0
                if res > 255.0:
                    res = 255.0
                result_uint8[_i, _j] = np.uint8(res)
    return result_uint8

def rescale_raster_uint8(raster, scale_factor):
    return _deep_to_list(_rescale_raster_uint8_jit(_sanitize_for_numba(raster), _sanitize_for_numba(scale_factor)))

@numba.njit
def _compute_band_index_expression_jit(band_arrays, expression_coeffs, expression_ops, data_type_min, data_type_max, nodata):
    n_bands = band_arrays.shape[0]
    height = band_arrays.shape[1]
    width = band_arrays.shape[2]
    output = np.zeros((height, width), dtype=np.int16)
    for _i in range(height):
        for _j in range(width):
            is_invalid = False
            for k in range(n_bands):
                if band_arrays[k, _i, _j] == nodata:
                    is_invalid = True
            if is_invalid:
                output[_i, _j] = np.int16(nodata)
                continue
            res = 0.0
            if n_bands >= 2:
                a = np.float32(band_arrays[0, _i, _j])
                b = np.float32(band_arrays[1, _i, _j])
                if expression_ops[0] == 0:
                    res = a + b
                elif expression_ops[0] == 1:
                    res = a - b
                elif expression_ops[0] == 2:
                    res = a * b
                elif expression_ops[0] == 3:
                    denom = b if b != 0.0 else 1.0
                    res = a / denom
                elif expression_ops[0] == 4:
                    denom = a + b if a + b != 0.0 else 1.0
                    res = (a - b) / denom
            if res < data_type_min:
                res = data_type_min
            if res > data_type_max:
                res = data_type_max
            output[_i, _j] = np.int16(res)
    return output

def compute_band_index_expression(band_arrays, expression_coeffs, expression_ops, data_type_min, data_type_max, nodata):
    return _deep_to_list(_compute_band_index_expression_jit(_sanitize_for_numba(band_arrays), _sanitize_for_numba(expression_coeffs), _sanitize_for_numba(expression_ops), _sanitize_for_numba(data_type_min), _sanitize_for_numba(data_type_max), _sanitize_for_numba(nodata)))

@numba.njit
def _blend_stack_block_jit(stack_data, stack_mask, not_done_mask, nodata):
    num_scenes = stack_data.shape[0]
    height = stack_data.shape[1]
    width = stack_data.shape[2]
    median_result = np.full((height, width), fill_value=nodata, dtype=np.int16)
    valid_count = np.zeros((height, width), dtype=np.int32)
    for _i in range(height):
        for _j in range(width):
            for s in range(num_scenes):
                if not stack_mask[s, _i, _j]:
                    valid_count[_i, _j] += 1
    for row in range(height):
        for col in range(width):
            if not_done_mask[row, col]:
                continue
            valid_vals = np.zeros(num_scenes, dtype=np.int16)
            n_valid = np.int32(0)
            for s in range(num_scenes):
                if not stack_mask[s, row, col]:
                    valid_vals[n_valid] = stack_data[s, row, col]
                    n_valid += 1
            if n_valid > 0:
                sorted_vals = np.sort(valid_vals[:n_valid])
                mid = n_valid // 2
                if n_valid % 2 == 0:
                    median_result[row, col] = np.int16((np.int32(sorted_vals[mid - 1]) + np.int32(sorted_vals[mid])) // 2)
                else:
                    median_result[row, col] = sorted_vals[mid]
    return (median_result, valid_count)

def blend_stack_block(stack_data, stack_mask, not_done_mask, nodata):
    return _deep_to_list(_blend_stack_block_jit(_sanitize_for_numba(stack_data), _sanitize_for_numba(stack_mask), _sanitize_for_numba(not_done_mask), _sanitize_for_numba(nodata)))

@numba.njit
def _compute_clear_observation_jit(stack_mask, num_scenes):
    height = stack_mask.shape[1]
    width = stack_mask.shape[2]
    count = np.zeros((height, width), dtype=np.int32)
    for s in range(num_scenes):
        for _i in range(height):
            for _j in range(width):
                if not stack_mask[s, _i, _j]:
                    count[_i, _j] += 1
    return count

def compute_clear_observation(stack_mask, num_scenes):
    return _deep_to_list(_compute_clear_observation_jit(_sanitize_for_numba(stack_mask), _sanitize_for_numba(num_scenes)))

@numba.njit
def _compute_total_observation_jit(raster_stack, nodata):
    num_scenes = raster_stack.shape[0]
    height = raster_stack.shape[1]
    width = raster_stack.shape[2]
    total_ob = np.zeros((height, width), dtype=np.uint8)
    for s in range(num_scenes):
        for _i in range(height):
            for _j in range(width):
                if raster_stack[s, _i, _j] != nodata:
                    total_ob[_i, _j] += 1
    return total_ob

def compute_total_observation(raster_stack, nodata):
    return _deep_to_list(_compute_total_observation_jit(_sanitize_for_numba(raster_stack), _sanitize_for_numba(nodata)))

@numba.njit
def _merge_scenes_priority_jit(raster_stack, mask_stack, nodata, num_scenes):
    height = raster_stack.shape[1]
    width = raster_stack.shape[2]
    result = np.full((height, width), fill_value=nodata, dtype=np.int16)
    done_mask = np.zeros((height, width), dtype=np.bool_)
    for s in range(num_scenes):
        for _i in range(height):
            for _j in range(width):
                if not mask_stack[s, _i, _j] and (not done_mask[_i, _j]) and (raster_stack[s, _i, _j] != nodata):
                    result[_i, _j] = raster_stack[s, _i, _j]
                    done_mask[_i, _j] = True
    return result

def merge_scenes_priority(raster_stack, mask_stack, nodata, num_scenes):
    return _deep_to_list(_merge_scenes_priority_jit(_sanitize_for_numba(raster_stack), _sanitize_for_numba(mask_stack), _sanitize_for_numba(nodata), _sanitize_for_numba(num_scenes)))

@numba.njit
def _sort_scenes_by_efficacy_jit(efficacy_arr, resolution):
    n = efficacy_arr.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        scores[i] = 100.0 * efficacy_arr[i] / resolution
    indices = np.argsort(-scores)
    return indices

def sort_scenes_by_efficacy(efficacy_arr, resolution):
    return _deep_to_list(_sort_scenes_by_efficacy_jit(_sanitize_for_numba(efficacy_arr), _sanitize_for_numba(resolution)))

@numba.njit
def _apply_mask_to_raster_jit(raster, mask, nodata):
    result = raster.copy()
    for _i in range(result.shape[0]):
        for _j in range(result.shape[1]):
            if mask[_i, _j]:
                result[_i, _j] = np.int16(nodata)
    return result

def apply_mask_to_raster(raster, mask, nodata):
    return _deep_to_list(_apply_mask_to_raster_jit(_sanitize_for_numba(raster), _sanitize_for_numba(mask), _sanitize_for_numba(nodata)))

@numba.njit
def _count_valid_pixels_jit(raster, nodata):
    count = np.int64(0)
    for _i in range(raster.shape[0]):
        for _j in range(raster.shape[1]):
            if raster[_i, _j] != nodata:
                count += 1
    return count

def count_valid_pixels(raster, nodata):
    return _deep_to_list(_count_valid_pixels_jit(_sanitize_for_numba(raster), _sanitize_for_numba(nodata)))

@numba.njit
def _compute_ndvi_jit(nir, red, nodata):
    result = np.zeros((nir.shape[0], nir.shape[1]), dtype=np.int16)
    for _i in range(result.shape[0]):
        for _j in range(result.shape[1]):
            if nir[_i, _j] == nodata or red[_i, _j] == nodata:
                result[_i, _j] = np.int16(nodata)
            else:
                n_f = np.float32(nir[_i, _j])
                r_f = np.float32(red[_i, _j])
                denom = n_f + r_f
                if denom == 0.0:
                    denom = 1.0
                val = (n_f - r_f) / denom
                if val < -1.0:
                    val = -1.0
                if val > 1.0:
                    val = 1.0
                result[_i, _j] = np.int16(val * 10000.0)
    return result

def compute_ndvi(nir, red, nodata):
    return _deep_to_list(_compute_ndvi_jit(_sanitize_for_numba(nir), _sanitize_for_numba(red), _sanitize_for_numba(nodata)))

@numba.njit
def _compute_evi_jit(nir, red, blue, nodata):
    result = np.zeros((nir.shape[0], nir.shape[1]), dtype=np.int16)
    for _i in range(result.shape[0]):
        for _j in range(result.shape[1]):
            if nir[_i, _j] == nodata or red[_i, _j] == nodata or blue[_i, _j] == nodata:
                result[_i, _j] = np.int16(nodata)
            else:
                n_f = np.float32(nir[_i, _j])
                r_f = np.float32(red[_i, _j])
                b_f = np.float32(blue[_i, _j])
                denom = n_f + 6.0 * r_f - 7.5 * b_f + 1.0
                if denom == 0.0:
                    denom = 1.0
                val = 2.5 * ((n_f - r_f) / denom)
                if val < -1.0:
                    val = -1.0
                if val > 1.0:
                    val = 1.0
                result[_i, _j] = np.int16(val * 10000.0)
    return result

def compute_evi(nir, red, blue, nodata):
    return _deep_to_list(_compute_evi_jit(_sanitize_for_numba(nir), _sanitize_for_numba(red), _sanitize_for_numba(blue), _sanitize_for_numba(nodata)))

@numba.njit
def _blend_pixel_stack_jit(stack_data, stack_mask, nodata, num_scenes):
    height = stack_data.shape[1]
    width = stack_data.shape[2]
    stk_result = np.full((height, width), fill_value=nodata, dtype=np.int16)
    med_result = np.full((height, width), fill_value=nodata, dtype=np.int16)
    clear_ob = np.zeros((height, width), dtype=np.int32)
    done_mask = np.zeros((height, width), dtype=np.bool_)
    for s in range(num_scenes):
        for _i in range(height):
            for _j in range(width):
                if not stack_mask[s, _i, _j]:
                    clear_ob[_i, _j] += 1
                if not stack_mask[s, _i, _j] and (not done_mask[_i, _j]):
                    stk_result[_i, _j] = stack_data[s, _i, _j]
                    done_mask[_i, _j] = True
    for row in range(height):
        for col in range(width):
            vals = np.zeros(num_scenes, dtype=np.int16)
            n_valid = np.int32(0)
            for s in range(num_scenes):
                if not stack_mask[s, row, col]:
                    vals[n_valid] = stack_data[s, row, col]
                    n_valid += 1
            if n_valid > 0:
                sorted_v = np.sort(vals[:n_valid])
                mid = n_valid // 2
                if n_valid % 2 == 0:
                    med_result[row, col] = np.int16((np.int32(sorted_v[mid - 1]) + np.int32(sorted_v[mid])) // 2)
                else:
                    med_result[row, col] = sorted_v[mid]
    return (stk_result, med_result, clear_ob)

def blend_pixel_stack(stack_data, stack_mask, nodata, num_scenes):
    return _deep_to_list(_blend_pixel_stack_jit(_sanitize_for_numba(stack_data), _sanitize_for_numba(stack_mask), _sanitize_for_numba(nodata), _sanitize_for_numba(num_scenes)))

@numba.njit
def _provenance_update_jit(provenance_arr, day_of_year, valid_mask, row_off, col_off, row_end, col_end):
    result = provenance_arr.copy()
    for _i in range(row_off, row_end):
        for _j in range(col_off, col_end):
            if valid_mask[_i - row_off, _j - col_off]:
                result[_i, _j] = np.int16(day_of_year)
    return result

def provenance_update(provenance_arr, day_of_year, valid_mask, row_off, col_off, row_end, col_end):
    return _deep_to_list(_provenance_update_jit(_sanitize_for_numba(provenance_arr), _sanitize_for_numba(day_of_year), _sanitize_for_numba(valid_mask), _sanitize_for_numba(row_off), _sanitize_for_numba(col_off), _sanitize_for_numba(row_end), _sanitize_for_numba(col_end)))

@numba.njit
def _merge_warped_accumulate_jit(raster_merge, raster_new, nodata, is_quality_band):
    merge = raster_merge.copy()
    if is_quality_band:
        raster_mask = np.ones(merge.shape, dtype=np.uint16)
        for _i in range(merge.shape[0]):
            for _j in range(merge.shape[1]):
                val = raster_new[_i, _j]
                merge[_i, _j] = merge[_i, _j] + val * raster_mask[_i, _j]
                if val != nodata:
                    raster_mask[_i, _j] = np.uint16(0)
        return (merge.astype(np.int16), raster_mask)
    else:
        for _i in range(merge.shape[0]):
            for _j in range(merge.shape[1]):
                if raster_new[_i, _j] != nodata:
                    merge[_i, _j] = raster_new[_i, _j]
        return (merge, np.zeros(merge.shape, dtype=np.uint16))

def merge_warped_accumulate(raster_merge, raster_new, nodata, is_quality_band):
    return _deep_to_list(_merge_warped_accumulate_jit(_sanitize_for_numba(raster_merge), _sanitize_for_numba(raster_new), _sanitize_for_numba(nodata), _sanitize_for_numba(is_quality_band)))

@numba.njit
def _compute_pixel_intersection_jit(stack_nodata_pos, raster_data_pos):
    count = 0
    for i in range(stack_nodata_pos.shape[0]):
        for j in range(raster_data_pos.shape[0]):
            if stack_nodata_pos[i] == raster_data_pos[j]:
                count += 1
    res = np.zeros(count, dtype=np.int64)
    idx = 0
    for i in range(stack_nodata_pos.shape[0]):
        for j in range(raster_data_pos.shape[0]):
            if stack_nodata_pos[i] == raster_data_pos[j]:
                res[idx] = stack_nodata_pos[i]
                idx += 1
    return res

def compute_pixel_intersection(stack_nodata_pos, raster_data_pos):
    return _deep_to_list(_compute_pixel_intersection_jit(_sanitize_for_numba(stack_nodata_pos), _sanitize_for_numba(raster_data_pos)))

@numba.njit
def _raster_block_fill_jit(height, width, fill_value, dtype_int16):
    if dtype_int16:
        return np.full((height, width), fill_value=fill_value, dtype=np.int16)
    else:
        return np.full((height, width), fill_value=fill_value, dtype=np.uint16)

def raster_block_fill(height, width, fill_value, dtype_int16):
    return _deep_to_list(_raster_block_fill_jit(_sanitize_for_numba(height), _sanitize_for_numba(width), _sanitize_for_numba(fill_value), _sanitize_for_numba(dtype_int16)))