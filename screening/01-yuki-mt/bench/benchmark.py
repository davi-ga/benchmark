import sys
import os
import time
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, "..", "input")
output_dir = os.path.join(base_dir, "..", "output")

sys.path.insert(0, input_dir)
import knn_feature as orig

sys.path.insert(0, output_dir)
import knn_feature as jit

np.random.seed(42)
N_NEIGHBORS = 50
N_CLASSES = 10

neighs_dist = np.sort(np.random.rand(N_NEIGHBORS).astype(np.float64))
neighs_y = np.random.randint(0, N_CLASSES, size=N_NEIGHBORS, dtype=np.int64)
k_list = np.array([5, 10, 20], dtype=np.int64)
k = 10

print("=" * 60)
print("YUKI-MT KNN FEATURE BENCHMARK")
print("=" * 60)

# Warmup JIT
_ = jit.compute_knn_features(neighs_dist, neighs_y, k_list, N_CLASSES)
_ = jit.compute_class_fractions(neighs_y, k, N_CLASSES)
_ = jit.compute_min_distances(neighs_dist, neighs_y, N_CLASSES)
_ = jit.compute_normalized_distances(neighs_dist, neighs_y, N_CLASSES)
_ = jit.compute_mean_distances(neighs_dist, neighs_y, k, N_CLASSES)

funcs = [
    ("compute_knn_features", (neighs_dist, neighs_y, k_list, N_CLASSES)),
    ("compute_class_fractions", (neighs_y, k, N_CLASSES)),
    ("compute_min_distances", (neighs_dist, neighs_y, N_CLASSES)),
    ("compute_normalized_distances", (neighs_dist, neighs_y, N_CLASSES)),
    ("compute_mean_distances", (neighs_dist, neighs_y, k, N_CLASSES)),
]

print(f"\n{'Function':<35} {'Original':>10} {'JIT':>10} {'Speedup':>10}")
print("-" * 60)

for name, args in funcs:
    orig_fn = getattr(orig, name)
    jit_fn = getattr(jit, name)

    times_orig = []
    for _ in range(5):
        t0 = time.perf_counter()
        orig_fn(*args)
        times_orig.append(time.perf_counter() - t0)

    times_jit = []
    for _ in range(5):
        t0 = time.perf_counter()
        jit_fn(*args)
        times_jit.append(time.perf_counter() - t0)

    avg_orig = np.mean(times_orig) * 1000
    avg_jit = np.mean(times_jit) * 1000
    speedup = avg_orig / avg_jit if avg_jit > 0 else float('inf')

    print(f"{name:<35} {avg_orig:>8.3f}ms {avg_jit:>8.3f}ms {speedup:>9.2f}x")

print("=" * 60)
