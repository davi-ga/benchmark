import sys
import os
import time
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, "..", "input")
output_dir = os.path.join(base_dir, "..", "output")

sys.path.insert(0, input_dir)
import infer_rt as orig_rt
import seir_model as orig_seir

sys.path.insert(0, output_dir)
import infer_rt as jit_rt
import seir_model as jit_seir

print("=" * 70)
print("COVID-MODEL BENCHMARK")
print("=" * 70)

print("\n--- Rt Inference Functions ---")

# Setup test data
n_days = 100
cases = np.random.poisson(50, size=n_days).astype(np.float64) + 1.0
r_list = np.linspace(0.5, 5.0, 50)
serial_period = 4.0
max_scaling_sigma = 0.1
default_process_sigma = 0.05
scale_sigma_from_count = 30.0

# Warmup JIT
try:
    _ = jit_rt.get_posteriors(cases, r_list, serial_period, max_scaling_sigma,
                               default_process_sigma, scale_sigma_from_count)
    print("✓ get_posteriors JIT compiled")
except Exception as e:
    print(f"✗ get_posteriors: {e}")

# Benchmark get_posteriors
try:
    times_orig = []
    for _ in range(3):
        t0 = time.perf_counter()
        orig_rt.get_posteriors(cases, r_list, serial_period, max_scaling_sigma,
                                default_process_sigma, scale_sigma_from_count)
        times_orig.append(time.perf_counter() - t0)

    times_jit = []
    for _ in range(3):
        t0 = time.perf_counter()
        jit_rt.get_posteriors(cases, r_list, serial_period, max_scaling_sigma,
                               default_process_sigma, scale_sigma_from_count)
        times_jit.append(time.perf_counter() - t0)

    avg_orig = np.mean(times_orig) * 1000
    avg_jit = np.mean(times_jit) * 1000
    speedup = avg_orig / avg_jit if avg_jit > 0 else float('inf')

    print(f"\nget_posteriors (100 days, 50 Rt values):")
    print(f"  Original: {avg_orig:.2f}ms")
    print(f"  JIT:      {avg_jit:.2f}ms")
    print(f"  Speedup:  {speedup:.2f}x")
except Exception as e:
    print(f"✗ Benchmark failed: {e}")

print("\n--- SEIR Model Functions ---")

# Setup SEIR parameters
params = orig_seir.build_params_array(
    N=1000000, R0=2.5, R0_hospital=1.5,
    sigma=1/5.2, delta=1/3.0, delta_hospital=1/5.0,
    gamma=1/7.0, kappa=1/10.0,
    hospitalization_rate_general=0.1, hospitalization_rate_icu=0.05,
    symptoms_to_hospital_days=5.0,
    hospitalization_length_of_stay_general=10.0,
    hospitalization_length_of_stay_icu=15.0,
    hospitalization_length_of_stay_icu_and_ventilator=20.0,
    fraction_icu_requiring_ventilator=0.3,
    mortality_rate_from_ICU=0.2, mortality_rate_from_hospital=0.1,
    mortality_rate_no_ICU_beds=0.3, mortality_rate_from_ICUVent=0.4,
    mortality_rate_no_general_beds=0.05,
    beds_general=1000, beds_ICU=100
)

y0 = np.array([999800, 200, 100, 100, 0, 50, 20, 10, 5, 0, 0, 0], dtype=np.float64)
t_list = np.arange(100, dtype=np.float64)
suppression_value = 1.0

# Warmup JIT
try:
    _ = jit_seir.seir_time_step(y0, 0.0, suppression_value, params)
    print("✓ seir_time_step JIT compiled")
except Exception as e:
    print(f"✗ seir_time_step: {e}")

# Benchmark seir_time_step
try:
    times_orig = []
    for _ in range(100):
        t0 = time.perf_counter()
        orig_seir.seir_time_step(y0, 0.0, suppression_value, params)
        times_orig.append(time.perf_counter() - t0)

    times_jit = []
    for _ in range(100):
        t0 = time.perf_counter()
        jit_seir.seir_time_step(y0, 0.0, suppression_value, params)
        times_jit.append(time.perf_counter() - t0)

    avg_orig = np.mean(times_orig) * 1000000
    avg_jit = np.mean(times_jit) * 1000000
    speedup = avg_orig / avg_jit if avg_jit > 0 else float('inf')

    print(f"\nseir_time_step (single step, 100 runs):")
    print(f"  Original: {avg_orig:.2f}μs")
    print(f"  JIT:      {avg_jit:.2f}μs")
    print(f"  Speedup:  {speedup:.2f}x")
except Exception as e:
    print(f"✗ Benchmark failed: {e}")

# Benchmark run_seir
try:
    times_orig = []
    for _ in range(3):
        t0 = time.perf_counter()
        orig_seir.run_seir(t_list, y0, suppression_value, params)
        times_orig.append(time.perf_counter() - t0)

    times_jit = []
    for _ in range(3):
        t0 = time.perf_counter()
        jit_seir.run_seir(t_list, y0, suppression_value, params)
        times_jit.append(time.perf_counter() - t0)

    avg_orig = np.mean(times_orig) * 1000
    avg_jit = np.mean(times_jit) * 1000
    speedup = avg_orig / avg_jit if avg_jit > 0 else float('inf')

    print(f"\nrun_seir (100 days):")
    print(f"  Original: {avg_orig:.2f}ms")
    print(f"  JIT:      {avg_jit:.2f}ms")
    print(f"  Speedup:  {speedup:.2f}x")
except Exception as e:
    print(f"✗ Benchmark failed: {e}")

print("\n" + "=" * 70)
