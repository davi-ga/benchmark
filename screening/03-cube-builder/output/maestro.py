import numba
import numpy as np
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _blend_stack_window_jit(stack_data, stack_mask, nodata, num_scenes, window_h, window_w):
    stk_result = np.full((window_h, window_w), fill_value=nodata, dtype=np.int16)
    done_mask = np.zeros((window_h, window_w), dtype=np.bool_)
    for s in range(num_scenes):
        for _i in range(stk_result.shape[0]):
            for _j in range(stk_result.shape[1]):
                if not stack_mask[s, _i, _j] and (not done_mask[_i, _j]):
                    stk_result[_i, _j] = stack_data[s, _i, _j]
                    done_mask[_i, _j] = True
    return stk_result

def blend_stack_window(stack_data, stack_mask, nodata, num_scenes, window_h, window_w):
    return _deep_to_list(_blend_stack_window_jit(_sanitize_for_numba(stack_data), _sanitize_for_numba(stack_mask), _sanitize_for_numba(nodata), _sanitize_for_numba(num_scenes), _sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _blend_median_window_jit(stack_data, stack_mask, nodata, num_scenes, window_h, window_w):
    med_result = np.full((window_h, window_w), fill_value=nodata, dtype=np.int16)
    for row in range(window_h):
        for col in range(window_w):
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
    return med_result

def blend_median_window(stack_data, stack_mask, nodata, num_scenes, window_h, window_w):
    return _deep_to_list(_blend_median_window_jit(_sanitize_for_numba(stack_data), _sanitize_for_numba(stack_mask), _sanitize_for_numba(nodata), _sanitize_for_numba(num_scenes), _sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _blend_clear_observation_jit(stack_mask, num_scenes, window_h, window_w):
    count = np.zeros((window_h, window_w), dtype=np.int32)
    for s in range(num_scenes):
        for _i in range(window_h):
            for _j in range(window_w):
                if not stack_mask[s, _i, _j]:
                    count[_i, _j] += 1
    return count

def blend_clear_observation(stack_mask, num_scenes, window_h, window_w):
    return _deep_to_list(_blend_clear_observation_jit(_sanitize_for_numba(stack_mask), _sanitize_for_numba(num_scenes), _sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _blend_total_observation_jit(raster_stack, nodata, num_scenes, window_h, window_w):
    total_ob = np.zeros((window_h, window_w), dtype=np.uint8)
    for s in range(num_scenes):
        for _i in range(window_h):
            for _j in range(window_w):
                if raster_stack[s, _i, _j] != nodata:
                    total_ob[_i, _j] += 1
    return total_ob

def blend_total_observation(raster_stack, nodata, num_scenes, window_h, window_w):
    return _deep_to_list(_blend_total_observation_jit(_sanitize_for_numba(raster_stack), _sanitize_for_numba(nodata), _sanitize_for_numba(num_scenes), _sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _blend_provenance_update_jit(provenance_block, day_of_year, valid_mask, window_h, window_w):
    result = provenance_block.copy()
    for _i in range(result.shape[0]):
        for _j in range(result.shape[1]):
            if valid_mask[_i, _j]:
                result[_i, _j] = np.int16(day_of_year)
    return result

def blend_provenance_update(provenance_block, day_of_year, valid_mask, window_h, window_w):
    return _deep_to_list(_blend_provenance_update_jit(_sanitize_for_numba(provenance_block), _sanitize_for_numba(day_of_year), _sanitize_for_numba(valid_mask), _sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _blend_nodata_intersection_jit(stack_raster_block, raster_block, nodata, window_h, window_w):
    count = 0
    for _i in range(window_h):
        for _j in range(window_w):
            if stack_raster_block[_i, _j] == nodata and raster_block[_i, _j] != nodata:
                count += 1
    res_i = np.zeros(count, dtype=np.int64)
    res_j = np.zeros(count, dtype=np.int64)
    idx = 0
    for _i in range(window_h):
        for _j in range(window_w):
            if stack_raster_block[_i, _j] == nodata and raster_block[_i, _j] != nodata:
                res_i[idx] = _i
                res_j[idx] = _j
                idx += 1
    return (res_i, res_j)

def blend_nodata_intersection(stack_raster_block, raster_block, nodata, window_h, window_w):
    return _deep_to_list(_blend_nodata_intersection_jit(_sanitize_for_numba(stack_raster_block), _sanitize_for_numba(raster_block), _sanitize_for_numba(nodata), _sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _merge_warp_accumulate_quality_jit(raster_merge, raster_new, raster_mask, nodata):
    merge = raster_merge.copy()
    mask_out = raster_mask.copy()
    for _i in range(mask_out.shape[0]):
        for _j in range(mask_out.shape[1]):
            val = raster_new[_i, _j]
            merge[_i, _j] = merge[_i, _j] + val * raster_mask[_i, _j]
            if val != nodata:
                mask_out[_i, _j] = 0
    return (merge, mask_out)

def merge_warp_accumulate_quality(raster_merge, raster_new, raster_mask, nodata):
    return _deep_to_list(_merge_warp_accumulate_quality_jit(_sanitize_for_numba(raster_merge), _sanitize_for_numba(raster_new), _sanitize_for_numba(raster_mask), _sanitize_for_numba(nodata)))

@numba.njit
def _merge_warp_accumulate_regular_jit(raster_merge, raster_new, nodata):
    merge = raster_merge.copy()
    for _i in range(merge.shape[0]):
        for _j in range(merge.shape[1]):
            if raster_new[_i, _j] != nodata:
                merge[_i, _j] = raster_new[_i, _j]
    return merge

def merge_warp_accumulate_regular(raster_merge, raster_new, nodata):
    return _deep_to_list(_merge_warp_accumulate_regular_jit(_sanitize_for_numba(raster_merge), _sanitize_for_numba(raster_new), _sanitize_for_numba(nodata)))

@numba.njit
def _merge_provenance_update_jit(raster_provenance, raster, nodata, dataset_index, window_h, window_w):
    result = raster_provenance.copy()
    for _i in range(result.shape[0]):
        for _j in range(result.shape[1]):
            if raster[_i, _j] != nodata:
                result[_i, _j] = np.uint8(dataset_index)
    return result

def merge_provenance_update(raster_provenance, raster, nodata, dataset_index, window_h, window_w):
    return _deep_to_list(_merge_provenance_update_jit(_sanitize_for_numba(raster_provenance), _sanitize_for_numba(raster), _sanitize_for_numba(nodata), _sanitize_for_numba(dataset_index), _sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _qa_statistics_blocks_jit(raster, nodata, clear_values, not_clear_values):
    clear_count = np.int64(0)
    not_clear_count = np.int64(0)
    total_pixels = np.int64(raster.shape[0] * raster.shape[1])
    for _i in range(raster.shape[0]):
        for _j in range(raster.shape[1]):
            val = raster[_i, _j]
            for k in range(clear_values.shape[0]):
                if val == clear_values[k]:
                    clear_count += 1
            for k in range(not_clear_values.shape[0]):
                if val == not_clear_values[k]:
                    not_clear_count += 1
    image_area = clear_count + not_clear_count
    not_clear_ratio = np.float64(100.0)
    if image_area != 0:
        not_clear_ratio = np.round(100.0 * not_clear_count / image_area, 2)
    efficacy = np.float64(0.0)
    if total_pixels > 0:
        efficacy = np.round(100.0 * clear_count / total_pixels, 2)
    return (efficacy, not_clear_ratio)

def qa_statistics_blocks(raster, nodata, clear_values, not_clear_values):
    return _deep_to_list(_qa_statistics_blocks_jit(_sanitize_for_numba(raster), _sanitize_for_numba(nodata), _sanitize_for_numba(clear_values), _sanitize_for_numba(not_clear_values)))

@numba.njit
def _sort_scenes_by_efficacy_jit(efficacy_values, resolution, num_scenes):
    scores = np.zeros(num_scenes, dtype=np.float64)
    for i in range(num_scenes):
        scores[i] = 100.0 * efficacy_values[i] / resolution
    indices = np.argsort(-scores)
    return indices

def sort_scenes_by_efficacy(efficacy_values, resolution, num_scenes):
    return _deep_to_list(_sort_scenes_by_efficacy_jit(_sanitize_for_numba(efficacy_values), _sanitize_for_numba(resolution), _sanitize_for_numba(num_scenes)))

@numba.njit
def _apply_qa_mask_bits_jit(qa_data, clear_values, not_clear_values, nodata):
    mask = np.zeros((qa_data.shape[0], qa_data.shape[1]), dtype=np.bool_)
    for _i in range(qa_data.shape[0]):
        for _j in range(qa_data.shape[1]):
            val = qa_data[_i, _j]
            if val == nodata:
                mask[_i, _j] = True
            for k in range(not_clear_values.shape[0]):
                if val & 1 << not_clear_values[k]:
                    mask[_i, _j] = True
            is_clear = False
            for k in range(clear_values.shape[0]):
                if val & 1 << clear_values[k]:
                    is_clear = True
            if not is_clear:
                mask[_i, _j] = True
    return mask

def apply_qa_mask_bits(qa_data, clear_values, not_clear_values, nodata):
    return _deep_to_list(_apply_qa_mask_bits_jit(_sanitize_for_numba(qa_data), _sanitize_for_numba(clear_values), _sanitize_for_numba(not_clear_values), _sanitize_for_numba(nodata)))

@numba.njit
def _apply_qa_mask_simple_jit(masked_data, not_clear_values, saturated_values, clear_values, nodata):
    mask = np.zeros((masked_data.shape[0], masked_data.shape[1]), dtype=np.bool_)
    for _i in range(masked_data.shape[0]):
        for _j in range(masked_data.shape[1]):
            val = masked_data[_i, _j]
            is_masked = False
            if val == nodata:
                is_masked = True
            for k in range(not_clear_values.shape[0]):
                if val == not_clear_values[k]:
                    is_masked = True
            for k in range(saturated_values.shape[0]):
                if val == saturated_values[k]:
                    is_masked = True
            for k in range(clear_values.shape[0]):
                if val == clear_values[k]:
                    is_masked = False
            mask[_i, _j] = is_masked
    return mask

def apply_qa_mask_simple(masked_data, not_clear_values, saturated_values, clear_values, nodata):
    return _deep_to_list(_apply_qa_mask_simple_jit(_sanitize_for_numba(masked_data), _sanitize_for_numba(not_clear_values), _sanitize_for_numba(saturated_values), _sanitize_for_numba(clear_values), _sanitize_for_numba(nodata)))

@numba.njit
def _blend_compute_notdone_jit(todomask, bmask):
    res = np.zeros(todomask.shape, dtype=np.bool_)
    for _i in range(todomask.shape[0]):
        for _j in range(todomask.shape[1]):
            res[_i, _j] = todomask[_i, _j] and bmask[_i, _j]
    return res

def blend_compute_notdone(todomask, bmask):
    return _deep_to_list(_blend_compute_notdone_jit(_sanitize_for_numba(todomask), _sanitize_for_numba(bmask)))

@numba.njit
def _blend_find_clear_pixels_jit(todomask, masked_mask):
    count = 0
    for _i in range(todomask.shape[0]):
        for _j in range(todomask.shape[1]):
            if todomask[_i, _j] and (not masked_mask[_i, _j]):
                count += 1
    res_i = np.zeros(count, dtype=np.int64)
    res_j = np.zeros(count, dtype=np.int64)
    idx = 0
    for _i in range(todomask.shape[0]):
        for _j in range(todomask.shape[1]):
            if todomask[_i, _j] and (not masked_mask[_i, _j]):
                res_i[idx] = _i
                res_j[idx] = _j
                idx += 1
    return (res_i, res_j)

def blend_find_clear_pixels(todomask, masked_mask):
    return _deep_to_list(_blend_find_clear_pixels_jit(_sanitize_for_numba(todomask), _sanitize_for_numba(masked_mask)))

@numba.njit
def _blend_update_stack_jit(stack_raster_block, raster_block, clear_pixels, nodata):
    result = stack_raster_block.copy()
    for _i in range(result.shape[0]):
        for _j in range(result.shape[1]):
            if clear_pixels[_i, _j]:
                result[_i, _j] = raster_block[_i, _j]
    return result

def blend_update_stack(stack_raster_block, raster_block, clear_pixels, nodata):
    return _deep_to_list(_blend_update_stack_jit(_sanitize_for_numba(stack_raster_block), _sanitize_for_numba(raster_block), _sanitize_for_numba(clear_pixels), _sanitize_for_numba(nodata)))

@numba.njit
def _compute_ndvi_blend_jit(nir_stack, red_stack, nodata, num_scenes, window_h, window_w):
    result = np.full((window_h, window_w), fill_value=nodata, dtype=np.int16)
    for row in range(window_h):
        for col in range(window_w):
            nir_vals = np.zeros(num_scenes, dtype=np.float32)
            red_vals = np.zeros(num_scenes, dtype=np.float32)
            n_valid = np.int32(0)
            for s in range(num_scenes):
                if nir_stack[s, row, col] != nodata and red_stack[s, row, col] != nodata:
                    nir_vals[n_valid] = np.float32(nir_stack[s, row, col])
                    red_vals[n_valid] = np.float32(red_stack[s, row, col])
                    n_valid += 1
            if n_valid > 0:
                nir_med = np.median(nir_vals[:n_valid])
                red_med = np.median(red_vals[:n_valid])
                denom = nir_med + red_med
                if denom != 0.0:
                    ndvi = (nir_med - red_med) / denom
                    ndvi_sc = ndvi * 10000.0
                    if ndvi_sc < -10000.0:
                        ndvi_sc = -10000.0
                    elif ndvi_sc > 10000.0:
                        ndvi_sc = 10000.0
                    result[row, col] = np.int16(ndvi_sc)
    return result

def compute_ndvi_blend(nir_stack, red_stack, nodata, num_scenes, window_h, window_w):
    return _deep_to_list(_compute_ndvi_blend_jit(_sanitize_for_numba(nir_stack), _sanitize_for_numba(red_stack), _sanitize_for_numba(nodata), _sanitize_for_numba(num_scenes), _sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _blend_datasource_block_jit(datasource_block, valid_mask, datasource_values, window_h, window_w):
    result = datasource_block.copy()
    for _i in range(result.shape[0]):
        for _j in range(result.shape[1]):
            if valid_mask[_i, _j]:
                result[_i, _j] = datasource_values[_i, _j]
    return result

def blend_datasource_block(datasource_block, valid_mask, datasource_values, window_h, window_w):
    return _deep_to_list(_blend_datasource_block_jit(_sanitize_for_numba(datasource_block), _sanitize_for_numba(valid_mask), _sanitize_for_numba(datasource_values), _sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _compute_pixel_resolution_jit(dist_x, dist_y, resx, resy):
    num_pixel_x = np.int32(round(dist_x / resx))
    num_pixel_y = np.int32(round(dist_y / resy))
    new_res_x = dist_x / num_pixel_x
    new_res_y = dist_y / num_pixel_y
    return (num_pixel_x, num_pixel_y, new_res_x, new_res_y)

def compute_pixel_resolution(dist_x, dist_y, resx, resy):
    return _deep_to_list(_compute_pixel_resolution_jit(_sanitize_for_numba(dist_x), _sanitize_for_numba(dist_y), _sanitize_for_numba(resx), _sanitize_for_numba(resy)))

@numba.njit
def _init_raster_quality_jit(numlin, numcol, source_nodata):
    raster = np.zeros((numlin, numcol), dtype=np.uint16)
    raster_merge = np.full((numlin, numcol), dtype=np.uint16, fill_value=source_nodata)
    raster_mask = np.ones((numlin, numcol), dtype=np.uint16)
    return (raster, raster_merge, raster_mask)

def init_raster_quality(numlin, numcol, source_nodata):
    return _deep_to_list(_init_raster_quality_jit(_sanitize_for_numba(numlin), _sanitize_for_numba(numcol), _sanitize_for_numba(source_nodata)))

@numba.njit
def _init_raster_regular_jit(numlin, numcol, nodata):
    raster = np.zeros((numlin, numcol), dtype=np.int16)
    raster_merge = np.full((numlin, numcol), fill_value=nodata, dtype=np.int16)
    return (raster, raster_merge)

def init_raster_regular(numlin, numcol, nodata):
    return _deep_to_list(_init_raster_regular_jit(_sanitize_for_numba(numlin), _sanitize_for_numba(numcol), _sanitize_for_numba(nodata)))

@numba.njit
def _blend_stack_ma_init_jit(num_scenes, window_h, window_w):
    stack_data = np.zeros((num_scenes, window_h, window_w), dtype=np.int16)
    stack_mask = np.ones((num_scenes, window_h, window_w), dtype=np.bool_)
    return (stack_data, stack_mask)

def blend_stack_ma_init(num_scenes, window_h, window_w):
    return _deep_to_list(_blend_stack_ma_init_jit(_sanitize_for_numba(num_scenes), _sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _blend_notdonemask_init_jit(window_h, window_w):
    return np.ones((window_h, window_w), dtype=np.bool_)

def blend_notdonemask_init(window_h, window_w):
    return _deep_to_list(_blend_notdonemask_init_jit(_sanitize_for_numba(window_h), _sanitize_for_numba(window_w)))

@numba.njit
def _provenance_determine_platform_jit(provenance_raster, nodata_value, num_datasets):
    counts = np.zeros(num_datasets, dtype=np.int64)
    valid_count = 0
    for _i in range(provenance_raster.shape[0]):
        for _j in range(provenance_raster.shape[1]):
            val = np.int32(provenance_raster[_i, _j])
            if val != nodata_value:
                valid_count += 1
                if val >= 0 and val < num_datasets:
                    counts[val] += 1
    if valid_count == 0:
        return np.int32(-1)
    max_idx = np.int32(0)
    max_count = counts[0]
    for i in range(1, num_datasets):
        if counts[i] > max_count:
            max_count = counts[i]
            max_idx = np.int32(i)
    return max_idx

def provenance_determine_platform(provenance_raster, nodata_value, num_datasets):
    return _deep_to_list(_provenance_determine_platform_jit(_sanitize_for_numba(provenance_raster), _sanitize_for_numba(nodata_value), _sanitize_for_numba(num_datasets)))

@numba.njit
def _qa_confidence_landsat8_from_provenance_jit(provenance_raster, index_landsat):
    res = np.zeros((provenance_raster.shape[0], provenance_raster.shape[1]), dtype=np.bool_)
    for _i in range(res.shape[0]):
        for _j in range(res.shape[1]):
            res[_i, _j] = provenance_raster[_i, _j] == index_landsat
    return res

def qa_confidence_landsat8_from_provenance(provenance_raster, index_landsat):
    return _deep_to_list(_qa_confidence_landsat8_from_provenance_jit(_sanitize_for_numba(provenance_raster), _sanitize_for_numba(index_landsat)))

@numba.njit
def _blend_apply_mask_block_jit(raster_block, mask_block):
    return mask_block

def blend_apply_mask_block(raster_block, mask_block):
    return _deep_to_list(_blend_apply_mask_block_jit(_sanitize_for_numba(raster_block), _sanitize_for_numba(mask_block)))

@numba.njit
def _blend_update_total_obs_jit(total_ob_block, raster_block, nodata):
    result = total_ob_block.copy()
    for _i in range(result.shape[0]):
        for _j in range(result.shape[1]):
            if raster_block[_i, _j] != nodata:
                result[_i, _j] += 1
    return result

def blend_update_total_obs(total_ob_block, raster_block, nodata):
    return _deep_to_list(_blend_update_total_obs_jit(_sanitize_for_numba(total_ob_block), _sanitize_for_numba(raster_block), _sanitize_for_numba(nodata)))