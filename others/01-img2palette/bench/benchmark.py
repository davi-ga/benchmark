"""
Benchmark para img2palette - comparando versões originais vs Numba JIT
"""
import sys
import os
import time
import numpy as np
from PIL import Image

# Adiciona diretório do módulo otimizado
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'output'))
import image as image_opt

# Adiciona diretório do módulo original
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'input'))
import image as image_orig

def benchmark_function(func_orig, func_opt, args, n_runs=100, warmup=10):
    """Benchmark uma função comparando original vs otimizada"""
    # Warmup (compilação JIT)
    for _ in range(warmup):
        func_opt(*args)
    
    # Mede tempo da versão original
    times_orig = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result_orig = func_orig(*args)
        times_orig.append(time.perf_counter() - start)
    
    # Mede tempo da versão otimizada
    times_opt = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result_opt = func_opt(*args)
        times_opt.append(time.perf_counter() - start)
    
    avg_orig = np.mean(times_orig) * 1000  # ms
    avg_opt = np.mean(times_opt) * 1000    # ms
    speedup = avg_orig / avg_opt if avg_opt > 0 else float('inf')
    
    return avg_orig, avg_opt, speedup

def main():
    print("=" * 80)
    print("BENCHMARK: img2palette - Conversão de Cores e Operações Numéricas")
    print("=" * 80)
    print()
    
    # Testa diferentes tamanhos de input
    sizes = [100, 500, 1000, 5000]
    
    for size in sizes:
        print(f"\n{'='*80}")
        print(f"Tamanho do input: {size} pixels")
        print(f"{'='*80}")
        
        # Gera dados de teste
        rgb_data = np.random.rand(size, 3).astype(np.float64)
        xyz_data = np.random.rand(size, 3).astype(np.float64) * 0.95
        lab_data = np.random.rand(size, 3).astype(np.float64) * 100
        rgb_uint8 = np.random.randint(0, 256, size=(size, 3), dtype=np.uint8)
        
        # Benchmark rgb2xyz
        avg_orig, avg_opt, speedup = benchmark_function(
            image_orig.rgb2xyz, image_opt.rgb2xyz, (rgb_data,)
        )
        print(f"rgb2xyz:       {avg_orig:7.3f} ms → {avg_opt:7.3f} ms | Speedup: {speedup:6.2f}x")
        
        # Benchmark xyz2lab
        avg_orig, avg_opt, speedup = benchmark_function(
            image_orig.xyz2lab, image_opt.xyz2lab, (xyz_data,)
        )
        print(f"xyz2lab:       {avg_orig:7.3f} ms → {avg_opt:7.3f} ms | Speedup: {speedup:6.2f}x")
        
        # Benchmark lab2xyz
        avg_orig, avg_opt, speedup = benchmark_function(
            image_orig.lab2xyz, image_opt.lab2xyz, (lab_data,)
        )
        print(f"lab2xyz:       {avg_orig:7.3f} ms → {avg_opt:7.3f} ms | Speedup: {speedup:6.2f}x")
        
        # Benchmark xyz2rgb
        avg_orig, avg_opt, speedup = benchmark_function(
            image_orig.xyz2rgb, image_opt.xyz2rgb, (xyz_data,)
        )
        print(f"xyz2rgb:       {avg_orig:7.3f} ms → {avg_opt:7.3f} ms | Speedup: {speedup:6.2f}x")
        
        # Benchmark rgb2lab
        avg_orig, avg_opt, speedup = benchmark_function(
            image_orig.rgb2lab, image_opt.rgb2lab, (rgb_data,)
        )
        print(f"rgb2lab:       {avg_orig:7.3f} ms → {avg_opt:7.3f} ms | Speedup: {speedup:6.2f}x")
        
        # Benchmark lab2rgb
        avg_orig, avg_opt, speedup = benchmark_function(
            image_orig.lab2rgb, image_opt.lab2rgb, (lab_data,)
        )
        print(f"lab2rgb:       {avg_orig:7.3f} ms → {avg_opt:7.3f} ms | Speedup: {speedup:6.2f}x")
        
        # Benchmark get_hex_string
        avg_orig, avg_opt, speedup = benchmark_function(
            image_orig.get_hex_string, image_opt.get_hex_string, (rgb_uint8[0],)
        )
        print(f"get_hex_string: {avg_orig:7.3f} ms → {avg_opt:7.3f} ms | Speedup: {speedup:6.2f}x")
        
        # Benchmark compute_medoid (com dados menores)
        small_data = np.random.rand(min(size, 100), 3).astype(np.float64)
        avg_orig, avg_opt, speedup = benchmark_function(
            image_orig.compute_medoid, image_opt.compute_medoid, (small_data,), n_runs=20
        )
        print(f"compute_medoid: {avg_orig:7.3f} ms → {avg_opt:7.3f} ms | Speedup: {speedup:6.2f}x")
        
        # Benchmark get_cluster
        centers = np.random.rand(5, 3).astype(np.float64)
        labels = np.random.randint(0, 5, size=size, dtype=np.int64)
        avg_orig, avg_opt, speedup = benchmark_function(
            image_orig.get_cluster, image_opt.get_cluster, (centers, labels)
        )
        print(f"get_cluster:   {avg_orig:7.3f} ms → {avg_opt:7.3f} ms | Speedup: {speedup:6.2f}x")
    
    print("\n" + "=" * 80)
    print("NOTAS:")
    print("- Funções de conversão de cor (rgb2xyz, xyz2lab, etc.) são otimizadas com Numba")
    print("- compute_medoid usa implementação Numba de pairwise_distances")
    print("- get_cluster usa loops explícitos em vez de np.vectorize")
    print("- Funções que usam sklearn/PIL/plotly NÃO são JIT (cluster_kmeans, cluster_agglo, etc.)")
    print("=" * 80)

if __name__ == "__main__":
    main()
