#!/usr/bin/env python3
"""
Teste de carga comparativo: yuki-mt original vs optimized (full pipeline)
"""

import requests
import json
import time
import numpy as np
from statistics import mean, median, stdev

ORIGINAL_URL = "https://yuki-mt-full-test-original-302846580571.us-central1.run.app/predict"
OPTIMIZED_URL = "https://yuki-mt-full-test-optimized-302846580571.us-central1.run.app/predict"

def test_service(url, name, num_requests=20):
    """Testa um serviço com múltiplas requisições"""
    print(f"\n{'='*60}")
    print(f"Testando: {name}")
    print(f"URL: {url}")
    print(f"Número de requisições: {num_requests}")
    print(f"{'='*60}")
    
    # Dados de teste
    payload = {
        "X": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1],
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2],
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3],
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4]
        ]
    }
    
    # Warmup (primeira requisição)
    print("\nWarmup...")
    response = requests.post(url, json=payload, timeout=60)
    if response.status_code != 200:
        print(f"❌ Warmup falhou: {response.status_code}")
        print(response.text)
        return None
    print("✅ Warmup OK")
    
    # Teste de carga
    print(f"\nExecutando {num_requests} requisições...")
    times = []
    server_times = []
    errors = 0
    
    for i in range(num_requests):
        try:
            start = time.perf_counter()
            response = requests.post(url, json=payload, timeout=60)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            if response.status_code == 200:
                data = response.json()
                server_time = data.get('time_ms', 0)
                times.append(elapsed)
                server_times.append(server_time)
                print(f"  Request {i+1:2d}: {elapsed:7.2f}ms (server: {server_time:7.2f}ms)")
            else:
                print(f"  Request {i+1:2d}: ❌ Error {response.status_code}")
                errors += 1
        except Exception as e:
            print(f"  Request {i+1:2d}: ❌ Exception: {e}")
            errors += 1
    
    if not times:
        print("❌ Nenhuma requisição bem-sucedida")
        return None
    
    # Estatísticas
    print(f"\n{'='*60}")
    print(f"Resultados: {name}")
    print(f"{'='*60}")
    print(f"Requisições bem-sucedidas: {len(times)}/{num_requests}")
    print(f"Erros: {errors}")
    print(f"\nLatência total (cliente):")
    print(f"  Média:  {mean(times):7.2f}ms")
    print(f"  Mediana: {median(times):7.2f}ms")
    print(f"  Desvio:  {stdev(times):7.2f}ms")
    print(f"  Min:     {min(times):7.2f}ms")
    print(f"  Max:     {max(times):7.2f}ms")
    print(f"\nTempo de processamento (servidor):")
    print(f"  Média:  {mean(server_times):7.2f}ms")
    print(f"  Mediana: {median(server_times):7.2f}ms")
    print(f"  Desvio:  {stdev(server_times):7.2f}ms")
    print(f"  Min:     {min(server_times):7.2f}ms")
    print(f"  Max:     {max(server_times):7.2f}ms")
    
    return {
        'name': name,
        'total_times': times,
        'server_times': server_times,
        'errors': errors
    }

def main():
    print("="*60)
    print("TESTE DE CARGA: yuki-mt original vs optimized")
    print("Pipeline completa (preprocessing + LLM + AST)")
    print("="*60)
    
    # Testar original
    result_original = test_service(ORIGINAL_URL, "ORIGINAL", num_requests=20)
    
    if result_original is None:
        print("\n❌ Teste do original falhou")
        return
    
    # Testar optimized
    result_optimized = test_service(OPTIMIZED_URL, "OPTIMIZED", num_requests=20)
    
    if result_optimized is None:
        print("\n❌ Teste do optimized falhou")
        return
    
    # Comparação
    print(f"\n{'='*60}")
    print(f"COMPARAÇÃO FINAL")
    print(f"{'='*60}")
    
    orig_total = mean(result_original['total_times'])
    opt_total = mean(result_optimized['total_times'])
    orig_server = mean(result_original['server_times'])
    opt_server = mean(result_optimized['server_times'])
    
    speedup_total = orig_total / opt_total
    speedup_server = orig_server / opt_server
    
    print(f"\nLatência total (cliente):")
    print(f"  Original:  {orig_total:7.2f}ms")
    print(f"  Optimized: {opt_total:7.2f}ms")
    print(f"  Speedup:   {speedup_total:7.2f}x")
    
    print(f"\nTempo de processamento (servidor):")
    print(f"  Original:  {orig_server:7.2f}ms")
    print(f"  Optimized: {opt_server:7.2f}ms")
    print(f"  Speedup:   {speedup_server:7.2f}x")
    
    print(f"\n{'='*60}")
    if speedup_server > 1.0:
        print(f"✅ Optimized é {speedup_server:.2f}x mais rápido que Original")
    else:
        print(f"❌ Optimized é {1/speedup_server:.2f}x mais lento que Original")
    print(f"{'='*60}")
    
    # Salvar resultados
    results = {
        'original': {
            'total_times': result_original['total_times'],
            'server_times': result_original['server_times'],
            'errors': result_original['errors']
        },
        'optimized': {
            'total_times': result_optimized['total_times'],
            'server_times': result_optimized['server_times'],
            'errors': result_optimized['errors']
        },
        'comparison': {
            'original_avg_total': orig_total,
            'optimized_avg_total': opt_total,
            'original_avg_server': orig_server,
            'optimized_avg_server': opt_server,
            'speedup_total': speedup_total,
            'speedup_server': speedup_server
        }
    }
    
    with open('benchmarks/gcr/yuki-mt-full-test/load_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Resultados salvos em: benchmarks/gcr/yuki-mt-full-test/load_test_results.json")

if __name__ == "__main__":
    main()
