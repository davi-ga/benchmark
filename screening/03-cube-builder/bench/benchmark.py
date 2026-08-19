import sys
import os
import time
import numpy as np

sys.path.insert(0, os.environ.get("MODULE_DIR", "."))
import maestro
import processing

def bench(func, args, n_runs=50, warmup=5):
    for _ in range(warmup):
        func(*args)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return np.median(times) * 1000

def main():
    print(f"{'Function':<40} {'Original':>10} {'JIT':>10} {'Speedup':>10}")
    print("-" * 72)

    stack_data = np.random.randint(0, 10000, (10, 64, 64), dtype=np.int16)
    stack_mask = np.random.choice([True, False], (10, 64, 64), p=[0.1, 0.9])
    nodata = -9999

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "input"))
    import importlib
    orig_maestro = importlib.import_module("maestro")
    orig_processing = importlib.import_module("processing")
    sys.path.pop(0)

    results = []

    for name, func_orig, func_jit, args in [
        ("blend_stack_window", orig_maestro.blend_stack_window, maestro.blend_stack_window,
         (stack_data, stack_mask, nodata, 10, 64, 64)),
        ("blend_median_window", orig_maestro.blend_median_window, maestro.blend_median_window,
         (stack_data, stack_mask, nodata, 10, 64, 64)),
        ("blend_clear_observation", orig_maestro.blend_clear_observation, maestro.blend_clear_observation,
         (stack_mask, 10, 64, 64)),
        ("blend_total_observation", orig_maestro.blend_total_observation, maestro.blend_total_observation,
         (stack_data, nodata, 10, 64, 64)),
    ]:
        t_orig = bench(func_orig, args)
        t_jit = bench(func_jit, args)
        speedup = t_orig / t_jit if t_jit > 0 else float("inf")
        print(f"{name:<40} {t_orig:>8.2f}ms {t_jit:>8.2f}ms {speedup:>9.2f}x")
        results.append((name, t_orig, t_jit, speedup))

    nir = np.random.randint(1000, 10000, (128, 128), dtype=np.int16)
    red = np.random.randint(1000, 10000, (128, 128), dtype=np.int16)

    for name, func_orig, func_jit, args in [
        ("compute_ndvi", orig_processing.compute_ndvi, processing.compute_ndvi,
         (nir, red, nodata)),
        ("extract_qa_bits", orig_processing.extract_qa_bits, processing.extract_qa_bits,
         (np.random.randint(0, 65535, (128, 128), dtype=np.int32), 5)),
        ("qa_cloud_confidence_apply", orig_processing.qa_cloud_confidence_apply, processing.qa_cloud_confidence_apply,
         (np.random.randint(0, 65535, (128, 128), dtype=np.int32), 1, 1, 1, 1,
          np.random.choice([True, False], (128, 128)))),
    ]:
        t_orig = bench(func_orig, args)
        t_jit = bench(func_jit, args)
        speedup = t_orig / t_jit if t_jit > 0 else float("inf")
        print(f"{name:<40} {t_orig:>8.2f}ms {t_jit:>8.2f}ms {speedup:>9.2f}x")
        results.append((name, t_orig, t_jit, speedup))

    print("-" * 72)
    avg_speedup = np.mean([r[3] for r in results])
    print(f"{'Average':<40} {'':>10} {'':>10} {avg_speedup:>9.2f}x")

if __name__ == "__main__":
    main()
