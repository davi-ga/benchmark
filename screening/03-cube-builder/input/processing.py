import numpy as np


def extract_qa_bits(band_data, bit_location):
    qa_data = np.asarray(band_data, dtype=np.int32)
    return qa_data & (1 << bit_location)


def extract_qa_bits_range(qa_data, start_offset, end_offset):
    qa = np.asarray(qa_data, dtype=np.int32)
    return (qa >> start_offset) - ((qa >> end_offset) << 2)


def qa_cloud_confidence_apply(data, cloud_rule, cloud_shadow_rule, snow_rule,
                               cirrus_rule, landsat_8_mask):
    qa = np.asarray(data, dtype=np.int32)
    result_mask = np.zeros(qa.shape, dtype=np.bool_)

    if cloud_rule == 1:
        cloud_arr = (qa >> 8) - ((qa >> 10) << 2)
        result_mask = result_mask | (cloud_arr >= 2)
    elif cloud_rule == 2:
        cloud_arr = (qa >> 8) - ((qa >> 10) << 2)
        result_mask = result_mask | (cloud_arr >= 1)
    elif cloud_rule == 3:
        cloud_arr = (qa >> 8) - ((qa >> 10) << 2)
        result_mask = result_mask | (cloud_arr == 3)

    if cloud_shadow_rule == 1:
        shadow_arr = (qa >> 10) - ((qa >> 12) << 2)
        result_mask = result_mask | (shadow_arr >= 2)
    elif cloud_shadow_rule == 2:
        shadow_arr = (qa >> 10) - ((qa >> 12) << 2)
        result_mask = result_mask | (shadow_arr >= 1)

    if snow_rule == 1:
        snow_arr = (qa >> 12) - ((qa >> 14) << 2)
        result_mask = result_mask | (snow_arr >= 2)
    elif snow_rule == 2:
        snow_arr = (qa >> 12) - ((qa >> 14) << 2)
        result_mask = result_mask | (snow_arr >= 1)

    if cirrus_rule >= 0:
        cirrus_arr = (qa >> 14) - ((qa >> 16) << 2)
        if cirrus_rule == 1:
            cirrus_mask = cirrus_arr >= 2
        elif cirrus_rule == 2:
            cirrus_mask = cirrus_arr >= 1
        else:
            cirrus_mask = np.zeros(qa.shape, dtype=np.bool_)
        result_mask = result_mask | (cirrus_mask & landsat_8_mask)

    return result_mask


def get_qa_mask_nodata(data, nodata):
    qa = np.asarray(data, dtype=np.int32)
    nodata_mask = qa == nodata
    return nodata_mask


def get_qa_mask_clear(data, clear_data_arr, not_clear_data_arr, nodata,
                       cloud_rule, cloud_shadow_rule, snow_rule,
                       cirrus_rule, landsat_8_mask_arr):
    qa = np.asarray(data, dtype=np.int32)
    nodata_mask = qa == nodata

    conf_mask = qa_cloud_confidence_apply(qa, cloud_rule, cloud_shadow_rule,
                                           snow_rule, cirrus_rule,
                                           landsat_8_mask_arr)

    not_clear_mask = np.zeros(qa.shape, dtype=np.bool_)
    for i in range(not_clear_data_arr.shape[0]):
        bit_val = not_clear_data_arr[i]
        bits = extract_qa_bits(qa, bit_val)
        not_clear_mask = not_clear_mask | (bits > 0)

    not_clear_mask = not_clear_mask | conf_mask | nodata_mask

    clear_mask = np.zeros(qa.shape, dtype=np.bool_)
    for i in range(clear_data_arr.shape[0]):
        bit_val = clear_data_arr[i]
        bits = extract_qa_bits(qa, bit_val)
        clear_mask = clear_mask | (bits > 0)

    final_mask = np.invert(clear_mask) | not_clear_mask
    return final_mask


def qa_statistics_compute(raster, nodata, clear_data_arr, not_clear_data_arr):
    raster_arr = np.asarray(raster, dtype=np.int32)
    valid_mask = raster_arr != nodata

    clear_count = np.int64(0)
    for i in range(clear_data_arr.shape[0]):
        clear_count += np.sum(raster_arr == clear_data_arr[i])

    not_clear_count = np.int64(0)
    for i in range(not_clear_data_arr.shape[0]):
        not_clear_count += np.sum(raster_arr == not_clear_data_arr[i])

    image_area = clear_count + not_clear_count
    total_pixels = np.int64(raster_arr.size)

    not_clear_ratio = np.float64(100.0)
    if image_area != 0:
        not_clear_ratio = np.round(100.0 * not_clear_count / image_area, 2)

    efficacy = np.round(100.0 * clear_count / total_pixels, 2)

    return efficacy, not_clear_ratio


def rescale_raster_uint8(raster, scale_factor):
    raster_arr = np.asarray(raster, dtype=np.float32)
    nodata_mask = raster_arr <= 0.0
    result = raster_arr / scale_factor * 255.0
    result = np.clip(result, 0.0, 255.0)
    result_uint8 = result.astype(np.uint8)
    result_uint8 = result_uint8 * np.invert(nodata_mask).astype(np.uint8)
    return result_uint8


def compute_band_index_expression(band_arrays, expression_coeffs, expression_ops,
                                   data_type_min, data_type_max, nodata):
    n_bands = band_arrays.shape[0]
    height = band_arrays.shape[1]
    width = band_arrays.shape[2]

    result = np.zeros((height, width), dtype=np.float32)

    if n_bands >= 2:
        a = band_arrays[0].astype(np.float32)
        b = band_arrays[1].astype(np.float32)

        if expression_ops[0] == 0:
            result = a + b
        elif expression_ops[0] == 1:
            result = a - b
        elif expression_ops[0] == 2:
            result = a * b
        elif expression_ops[0] == 3:
            denom = b.copy()
            denom[denom == 0.0] = 1.0
            result = a / denom
        elif expression_ops[0] == 4:
            denom = a + b
            denom[denom == 0.0] = 1.0
            result = (a - b) / denom

    result = np.clip(result, data_type_min, data_type_max)

    invalid = np.zeros((height, width), dtype=np.bool_)
    for i in range(n_bands):
        invalid = invalid | (band_arrays[i] == nodata)

    output = result.astype(np.int16)
    output[invalid] = np.int16(nodata)

    return output


def blend_stack_block(stack_ma, not_done_mask, nodata):
    num_scenes = stack_ma.shape[0]
    height = stack_ma.shape[1]
    width = stack_ma.shape[2]

    median_result = np.full((height, width), fill_value=nodata, dtype=np.int16)

    valid_count = np.sum(np.invert(stack_ma.mask), axis=0).astype(np.int32)

    for row in range(height):
        for col in range(width):
            if not_done_mask[row, col]:
                continue
            valid_vals = np.zeros(num_scenes, dtype=np.int16)
            n_valid = np.int32(0)
            for s in range(num_scenes):
                if not stack_ma.mask[s, row, col]:
                    valid_vals[n_valid] = stack_ma.data[s, row, col]
                    n_valid += 1
            if n_valid > 0:
                sorted_vals = np.sort(valid_vals[:n_valid])
                mid = n_valid // 2
                if n_valid % 2 == 0:
                    median_result[row, col] = np.int16(
                        (np.int32(sorted_vals[mid - 1]) + np.int32(sorted_vals[mid])) // 2)
                else:
                    median_result[row, col] = sorted_vals[mid]

    return median_result, valid_count


def compute_clear_observation(stack_mask, num_scenes):
    height = stack_mask.shape[1]
    width = stack_mask.shape[2]
    count = np.zeros((height, width), dtype=np.int32)
    for s in range(num_scenes):
        count += np.invert(stack_mask[s]).astype(np.int32)
    return count


def compute_total_observation(raster_stack, nodata):
    num_scenes = raster_stack.shape[0]
    height = raster_stack.shape[1]
    width = raster_stack.shape[2]
    total_ob = np.zeros((height, width), dtype=np.uint8)
    for s in range(num_scenes):
        total_ob += (raster_stack[s] != nodata).astype(np.uint8)
    return total_ob


def merge_scenes_priority(raster_stack, mask_stack, nodata, num_scenes):
    height = raster_stack.shape[1]
    width = raster_stack.shape[2]
    result = np.full((height, width), fill_value=nodata, dtype=np.int16)
    done_mask = np.zeros((height, width), dtype=np.bool_)

    for s in range(num_scenes):
        scene = raster_stack[s]
        scene_mask = mask_stack[s]
        valid = np.invert(scene_mask) & np.invert(done_mask) & (scene != nodata)
        result[valid] = scene[valid]
        done_mask = done_mask | valid

    return result


def sort_scenes_by_efficacy(efficacy_arr, resolution):
    n = efficacy_arr.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        scores[i] = 100.0 * efficacy_arr[i] / resolution

    indices = np.argsort(-scores)
    return indices


def apply_mask_to_raster(raster, mask, nodata):
    raster_arr = np.asarray(raster, dtype=np.int16)
    mask_arr = np.asarray(mask, dtype=np.bool_)
    result = raster_arr.copy()
    result[mask_arr] = np.int16(nodata)
    return result


def count_valid_pixels(raster, nodata):
    raster_arr = np.asarray(raster)
    return np.int64(np.sum(raster_arr != nodata))


def compute_ndvi(nir, red, nodata):
    nir_f = nir.astype(np.float32)
    red_f = red.astype(np.float32)
    denom = nir_f + red_f
    denom[denom == 0.0] = 1.0
    ndvi = (nir_f - red_f) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0)
    invalid = (nir == nodata) | (red == nodata)
    result = (ndvi * 10000.0).astype(np.int16)
    result[invalid] = np.int16(nodata)
    return result


def compute_evi(nir, red, blue, nodata):
    nir_f = nir.astype(np.float32)
    red_f = red.astype(np.float32)
    blue_f = blue.astype(np.float32)
    denom = nir_f + 6.0 * red_f - 7.5 * blue_f + 1.0
    denom[denom == 0.0] = 1.0
    evi = 2.5 * ((nir_f - red_f) / denom)
    evi = np.clip(evi, -1.0, 1.0)
    invalid = (nir == nodata) | (red == nodata) | (blue == nodata)
    result = (evi * 10000.0).astype(np.int16)
    result[invalid] = np.int16(nodata)
    return result


def blend_pixel_stack(stack_data, stack_mask, nodata, num_scenes):
    height = stack_data.shape[1]
    width = stack_data.shape[2]

    stk_result = np.full((height, width), fill_value=nodata, dtype=np.int16)
    med_result = np.full((height, width), fill_value=nodata, dtype=np.int16)
    clear_ob = np.zeros((height, width), dtype=np.int32)
    done_mask = np.zeros((height, width), dtype=np.bool_)

    for s in range(num_scenes):
        valid = np.invert(stack_mask[s]) & np.invert(done_mask)
        stk_result[valid] = stack_data[s][valid]
        done_mask = done_mask | valid
        clear_ob += np.invert(stack_mask[s]).astype(np.int32)

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
                    med_result[row, col] = np.int16(
                        (np.int32(sorted_v[mid - 1]) + np.int32(sorted_v[mid])) // 2)
                else:
                    med_result[row, col] = sorted_v[mid]

    return stk_result, med_result, clear_ob


def provenance_update(provenance_arr, day_of_year, valid_mask, row_off, col_off,
                       row_end, col_end):
    result = provenance_arr.copy()
    block = result[row_off:row_end, col_off:col_end]
    block[valid_mask] = np.int16(day_of_year)
    result[row_off:row_end, col_off:col_end] = block
    return result


def merge_warped_accumulate(raster_merge, raster_new, nodata, is_quality_band):
    merge = raster_merge.copy()
    new_data = raster_new

    if is_quality_band:
        raster_mask = np.ones(merge.shape, dtype=np.uint16)
        factor = new_data.astype(np.uint16) * raster_mask
        merge = merge.astype(np.uint16) + factor
        raster_mask[new_data != nodata] = np.uint16(0)
        return merge.astype(np.int16), raster_mask
    else:
        valid = new_data != nodata
        merge[valid] = new_data[valid]
        return merge, np.zeros(merge.shape, dtype=np.uint16)


def compute_pixel_intersection(stack_nodata_pos, raster_data_pos):
    return np.intersect1d(stack_nodata_pos, raster_data_pos)


def raster_block_fill(height, width, fill_value, dtype_int16):
    if dtype_int16:
        return np.full((height, width), fill_value=fill_value, dtype=np.int16)
    else:
        return np.full((height, width), fill_value=fill_value, dtype=np.uint16)
