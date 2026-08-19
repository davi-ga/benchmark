import numpy as np


def blend_stack_window(stack_data, stack_mask, nodata, num_scenes, window_h, window_w):
    stk_result = np.full((window_h, window_w), fill_value=nodata, dtype=np.int16)
    done_mask = np.zeros((window_h, window_w), dtype=np.bool_)

    for s in range(num_scenes):
        valid = np.invert(stack_mask[s]) & np.invert(done_mask)
        stk_result[valid] = stack_data[s][valid]
        done_mask = done_mask | valid

    return stk_result


def blend_median_window(stack_data, stack_mask, nodata, num_scenes, window_h, window_w):
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
                    med_result[row, col] = np.int16(
                        (np.int32(sorted_v[mid - 1]) + np.int32(sorted_v[mid])) // 2)
                else:
                    med_result[row, col] = sorted_v[mid]

    return med_result


def blend_clear_observation(stack_mask, num_scenes, window_h, window_w):
    count = np.zeros((window_h, window_w), dtype=np.int32)
    for s in range(num_scenes):
        count += np.invert(stack_mask[s]).astype(np.int32)
    return count


def blend_total_observation(raster_stack, nodata, num_scenes, window_h, window_w):
    total_ob = np.zeros((window_h, window_w), dtype=np.uint8)
    for s in range(num_scenes):
        total_ob += (raster_stack[s] != nodata).astype(np.uint8)
    return total_ob


def blend_provenance_update(provenance_block, day_of_year, valid_mask, window_h, window_w):
    result = provenance_block.copy()
    result[valid_mask] = np.int16(day_of_year)
    return result


def blend_nodata_intersection(stack_raster_block, raster_block, nodata, window_h, window_w):
    stack_nodata = stack_raster_block == nodata
    raster_valid = raster_block != nodata

    stack_flat = np.ravel_multi_index(
        np.where(stack_nodata),
        (window_h, window_w)
    )
    raster_flat = np.ravel_multi_index(
        np.where(raster_valid),
        (window_h, window_w)
    )

    intersection = np.intersect1d(stack_flat, raster_flat)
    if intersection.size > 0:
        coords = np.unravel_index(intersection, (window_h, window_w))
        return coords
    return None


def merge_warp_accumulate_quality(raster_merge, raster_new, raster_mask, nodata):
    merge = raster_merge.astype(np.uint16)
    factor = raster_new.astype(np.uint16) * raster_mask
    merge = merge + factor
    mask_out = raster_mask.copy()
    mask_out[raster_new != nodata] = np.uint16(0)
    return merge, mask_out


def merge_warp_accumulate_regular(raster_merge, raster_new, nodata):
    merge = raster_merge.copy()
    valid = raster_new != nodata
    merge[valid] = raster_new[valid]
    return merge


def merge_provenance_update(raster_provenance, raster, nodata, dataset_index, window_h, window_w):
    result = raster_provenance.copy()
    raster_masked = np.ma.masked_where(raster == nodata, raster)
    where_valid = np.invert(raster_masked.mask)
    result[where_valid] = np.uint8(dataset_index)
    return result


def qa_statistics_blocks(raster, nodata, clear_values, not_clear_values):
    raster_arr = np.asarray(raster, dtype=np.int32)

    clear_count = np.int64(0)
    for i in range(clear_values.shape[0]):
        clear_count += np.int64(np.sum(raster_arr == clear_values[i]))

    not_clear_count = np.int64(0)
    for i in range(not_clear_values.shape[0]):
        not_clear_count += np.int64(np.sum(raster_arr == not_clear_values[i]))

    image_area = clear_count + not_clear_count
    total_pixels = np.int64(raster_arr.size)

    not_clear_ratio = np.float64(100.0)
    if image_area != 0:
        not_clear_ratio = np.round(100.0 * not_clear_count / image_area, 2)

    efficacy = np.float64(0.0)
    if total_pixels > 0:
        efficacy = np.round(100.0 * clear_count / total_pixels, 2)

    return efficacy, not_clear_ratio


def sort_scenes_by_efficacy(efficacy_values, resolution, num_scenes):
    scores = np.zeros(num_scenes, dtype=np.float64)
    for i in range(num_scenes):
        scores[i] = 100.0 * efficacy_values[i] / resolution
    indices = np.argsort(-scores)
    return indices


def apply_qa_mask_bits(qa_data, clear_values, not_clear_values, nodata):
    qa = np.asarray(qa_data, dtype=np.int32)
    mask = np.zeros(qa.shape, dtype=np.bool_)

    mask = mask | (qa == nodata)

    for i in range(not_clear_values.shape[0]):
        bit_val = not_clear_values[i]
        bits = qa & (1 << bit_val)
        mask = mask | (bits > 0)

    clear_mask = np.zeros(qa.shape, dtype=np.bool_)
    for i in range(clear_values.shape[0]):
        bit_val = clear_values[i]
        bits = qa & (1 << bit_val)
        clear_mask = clear_mask | (bits > 0)

    final_mask = mask | np.invert(clear_mask)
    return final_mask


def apply_qa_mask_simple(masked_data, not_clear_values, saturated_values, clear_values, nodata):
    data = np.asarray(masked_data, dtype=np.int32)
    mask = np.zeros(data.shape, dtype=np.bool_)

    for i in range(not_clear_values.shape[0]):
        mask = mask | (data == not_clear_values[i])

    mask = mask | (data == nodata)

    for i in range(saturated_values.shape[0]):
        mask = mask | (data == saturated_values[i])

    for i in range(clear_values.shape[0]):
        mask[data == clear_values[i]] = False

    return mask


def blend_compute_notdone(todomask, bmask):
    return todomask * bmask


def blend_find_clear_pixels(todomask, masked_mask):
    return np.where(np.logical_and(todomask, np.invert(masked_mask)))


def blend_update_stack(stack_raster_block, raster_block, clear_pixels, nodata):
    result = stack_raster_block.copy()
    result[clear_pixels] = raster_block[clear_pixels]
    return result


def compute_ndvi_blend(nir_stack, red_stack, nodata, num_scenes, window_h, window_w):
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
                    result[row, col] = np.int16(np.clip(ndvi * 10000.0, -10000.0, 10000.0))

    return result


def blend_datasource_block(datasource_block, valid_mask, datasource_values, window_h, window_w):
    result = datasource_block.copy()
    result[valid_mask] = datasource_values[valid_mask]
    return result


def compute_pixel_resolution(dist_x, dist_y, resx, resy):
    num_pixel_x = np.int32(round(dist_x / resx))
    num_pixel_y = np.int32(round(dist_y / resy))
    new_res_x = dist_x / num_pixel_x
    new_res_y = dist_y / num_pixel_y
    return num_pixel_x, num_pixel_y, new_res_x, new_res_y


def init_raster_quality(numlin, numcol, source_nodata):
    raster = np.zeros((numlin, numcol), dtype=np.uint16)
    raster_merge = np.full((numlin, numcol), dtype=np.uint16, fill_value=source_nodata)
    raster_mask = np.ones((numlin, numcol), dtype=np.uint16)
    return raster, raster_merge, raster_mask


def init_raster_regular(numlin, numcol, nodata):
    raster = np.zeros((numlin, numcol), dtype=np.int16)
    raster_merge = np.full((numlin, numcol), fill_value=nodata, dtype=np.int16)
    return raster, raster_merge


def blend_stack_ma_init(num_scenes, window_h, window_w):
    stack_data = np.zeros((num_scenes, window_h, window_w), dtype=np.int16)
    stack_mask = np.ones((num_scenes, window_h, window_w), dtype=np.bool_)
    return stack_data, stack_mask


def blend_notdonemask_init(window_h, window_w):
    return np.ones((window_h, window_w), dtype=np.bool_)


def provenance_determine_platform(provenance_raster, nodata_value, num_datasets):
    valid = provenance_raster[provenance_raster != nodata_value]
    if valid.size == 0:
        return np.int32(-1)

    counts = np.zeros(num_datasets, dtype=np.int64)
    for i in range(valid.size):
        val = np.int32(valid[i])
        if val >= 0 and val < num_datasets:
            counts[val] += 1

    max_idx = np.int32(0)
    max_count = counts[0]
    for i in range(1, num_datasets):
        if counts[i] > max_count:
            max_count = counts[i]
            max_idx = np.int32(i)

    return max_idx


def qa_confidence_landsat8_from_provenance(provenance_raster, index_landsat):
    return provenance_raster == index_landsat


def blend_apply_mask_block(raster_block, mask_block):
    return np.ma.masked_where(mask_block, raster_block)


def blend_update_total_obs(total_ob_block, raster_block, nodata):
    result = total_ob_block.copy()
    result += (raster_block != nodata).astype(np.uint8)
    return result
