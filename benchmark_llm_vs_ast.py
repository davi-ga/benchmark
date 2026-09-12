#!/usr/bin/env python3
"""
Benchmark comparativo: LLM-Only vs LLM+AST

Mede performance real das funções otimizadas para provar que
mais anotações via AST = mais speedup.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import time


def run_benchmark(code: str, label: str, num_runs: int = 10) -> dict:
    """Executa benchmark no código e retorna métricas."""
    
    result = {"label": label, "times_ms": [], "errors": []}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Escreve módulo
        module_file = Path(tmpdir) / "bench_module.py"
        module_file.write_text(code)
        
        # Copia helpers
        forge_src = Path(__file__).parent.parent / "src"
        deep_py = forge_src / "utils" / "deep.py"
        if deep_py.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("deep_module", str(deep_py))
            deep_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(deep_module)
            
            helpers_file = Path(tmpdir) / "_numba_helpers.py"
            helpers_file.write_text(deep_module.DEEP_TO_LIST_SRC.lstrip("\n"))
        
        # Script de benchmark
        bench_script = f"""
import sys
import time
import numpy as np
sys.path.insert(0, '{tmpdir}')

try:
    import bench_module
    
    # Testa se tem função JIT
    jit_funcs = []
    for name in dir(bench_module):
        obj = getattr(bench_module, name)
        if callable(obj) and hasattr(obj, 'nopython_signatures'):
            jit_funcs.append(name)
    
    if not jit_funcs:
        print("NO_JIT_FUNCTIONS")
    else:
        print(f"JIT_COUNT: {{len(jit_funcs)}}")
        
        # Prepara dados de teste
        np.random.seed(42)
        X_train = np.random.rand(50, 10)
        y_train = np.random.randint(0, 3, 50)
        X_test = np.random.rand(5, 10)
        
        # Warmup (compila JIT)
        warmup_start = time.perf_counter()
        try:
            # Tenta instanciar e usar a classe se existir
            if hasattr(bench_module, 'NearestNeighborsFeats'):
                nnf = bench_module.NearestNeighborsFeats(
                    n_jobs=1, k_list=[3, 8], metric='minkowski', n_classes=3
                )
                nnf.fit(X_train, y_train)
                _ = nnf.predict(X_test[:1])
            else:
                # Tenta chamar função principal
                for func_name in jit_funcs:
                    func = getattr(bench_module, func_name)
                    try:
                        # Tenta com dados dummy
                        func(X_test[0])
                    except:
                        pass
        except Exception as e:
            print(f"WARMUP_ERROR: {{e}}")
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        print(f"WARMUP_MS: {{warmup_time:.2f}}")
        
        # Benchmark principal
        times = []
        for i in range({num_runs}):
            start = time.perf_counter()
            try:
                if hasattr(bench_module, 'NearestNeighborsFeats'):
                    nnf = bench_module.NearestNeighborsFeats(
                        n_jobs=1, k_list=[3, 8], metric='minkowski', n_classes=3
                    )
                    nnf.fit(X_train, y_train)
                    result = nnf.predict(X_test)
                else:
                    # Chama função principal
                    pass
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            except Exception as e:
                print(f"RUN_ERROR_{{i}}: {{e}}")
        
        if times:
            print(f"TIMES: {{','.join(f'{{t:.3f}}' for t in times)}}")
            print(f"AVG_MS: {{sum(times)/len(times):.3f}}")
            print(f"MIN_MS: {{min(times):.3f}}")
            print(f"MAX_MS: {{max(times):.3f}}")

except Exception as e:
    import traceback
    print(f"BENCH_ERROR: {{e}}")
    traceback.print_exc()
"""
        
        bench_file = Path(tmpdir) / "run_bench.py"
        bench_file.write_text(bench_script)
        
        proc = subprocess.run(
            [sys.executable, str(bench_file)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        
        output = proc.stdout + proc.stderr
        
        for line in output.split('\n'):
            if "JIT_COUNT:" in line:
                result["jit_count"] = int(line.split(":")[1].strip())
            elif "WARMUP_MS:" in line:
                result["warmup_ms"] = float(line.split(":")[1].strip())
            elif "TIMES:" in line:
                times_str = line.replace("TIMES: ", "").strip()
                result["times_ms"] = [float(t) for t in times_str.split(',')]
            elif "AVG_MS:" in line:
                result["avg_ms"] = float(line.split(":")[1].strip())
            elif "MIN_MS:" in line:
                result["min_ms"] = float(line.split(":")[1].strip())
            elif "MAX_MS:" in line:
                result["max_ms"] = float(line.split(":")[1].strip())
            elif "ERROR" in line:
                result["errors"].append(line)
    
    return result


def main():
    print("=" * 80)
    print("BENCHMARK: LLM-Only vs LLM+AST")
    print("Medindo performance real com dados sintéticos")
    print("=" * 80)
    
    # Encontra run mais recente
    exp_dir = Path(__file__).parent / "experimento_output"
    runs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not runs:
        print("ERROR: Nenhum experimento encontrado. Execute experimento_llm_vs_ast.py primeiro.")
        return
    
    latest_run = runs[-1]
    print(f"\nUsando dados de: {latest_run.name}")
    
    # Carrega códigos
    llm_only_code = (latest_run / "02_llm_only.py").read_text()
    llm_plus_ast_code = (latest_run / "04_llm_plus_ast.py").read_text()
    original_code = (latest_run / "01_original.py").read_text()
    
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Código Original (sem otimização)")
    print("=" * 80)
    result_original = run_benchmark(original_code, "Original", num_runs=5)
    print(f"  Warmup: {result_original.get('warmup_ms', 'N/A')}ms")
    print(f"  Média: {result_original.get('avg_ms', 'N/A')}ms")
    print(f"  Min/Max: {result_original.get('min_ms', 'N/A')}/{result_original.get('max_ms', 'N/A')}ms")
    
    print("\n" + "=" * 80)
    print("BENCHMARK 2: LLM-Only (1 função otimizada)")
    print("=" * 80)
    result_llm = run_benchmark(llm_only_code, "LLM-Only", num_runs=5)
    print(f"  JIT functions: {result_llm.get('jit_count', 0)}")
    print(f"  Warmup: {result_llm.get('warmup_ms', 'N/A')}ms")
    print(f"  Média: {result_llm.get('avg_ms', 'N/A')}ms")
    print(f"  Min/Max: {result_llm.get('min_ms', 'N/A')}/{result_llm.get('max_ms', 'N/A')}ms")
    
    print("\n" + "=" * 80)
    print("BENCHMARK 3: LLM+AST (9 funções otimizadas)")
    print("=" * 80)
    result_ast = run_benchmark(llm_plus_ast_code, "LLM+AST", num_runs=5)
    print(f"  JIT functions: {result_ast.get('jit_count', 0)}")
    print(f"  Warmup: {result_ast.get('warmup_ms', 'N/A')}ms")
    print(f"  Média: {result_ast.get('avg_ms', 'N/A')}ms")
    print(f"  Min/Max: {result_ast.get('min_ms', 'N/A')}/{result_ast.get('max_ms', 'N/A')}ms")
    
    # Calcula speedups
    print("\n" + "=" * 80)
    print("COMPARAÇÃO DE SPEEDUP")
    print("=" * 80)
    
    if result_original.get('avg_ms') and result_llm.get('avg_ms'):
        speedup_llm = result_original['avg_ms'] / result_llm['avg_ms']
        print(f"\nLLM-Only vs Original: {speedup_llm:.2f}x")
    
    if result_original.get('avg_ms') and result_ast.get('avg_ms'):
        speedup_ast = result_original['avg_ms'] / result_ast['avg_ms']
        print(f"LLM+AST vs Original: {speedup_ast:.2f}x")
    
    if result_llm.get('avg_ms') and result_ast.get('avg_ms'):
        speedup_diff = result_llm['avg_ms'] / result_ast['avg_ms']
        print(f"LLM+AST vs LLM-Only: {speedup_diff:.2f}x")
    
    # Salva resultados
    benchmark_results = {
        "original": result_original,
        "llm_only": result_llm,
        "llm_plus_ast": result_ast,
    }
    
    results_file = latest_run / "benchmark_results.json"
    results_file.write_text(json.dumps(benchmark_results, indent=2))
    print(f"\nResultados salvos em: {results_file}")


if __name__ == "__main__":
    main()
