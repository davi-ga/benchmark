#!/usr/bin/env python3
"""
Benchmark de Reprodutibilidade da Pipeline Forge

Executa cada benchmark (yuki-mt, img2palette, cube-builder) 10 vezes,
registrando métricas detalhadas de cada chamada à API Gemini 3.1 Pro Preview.

Uso:
    python benchmarks/benchmark_reproducibility.py

Saída:
    benchmarks/reproducibility_results.json
"""

import sys
import os
import json
import re
import time
import shutil
import io
import statistics
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from build_pipeline import run as pipeline_run


# Configuração
BENCHMARKS = ["yuki-mt", "img2palette", "cube-builder"]
RUNS_PER_BENCHMARK = 10
MAX_ATTEMPTS = 3
BENCHMARKS_DIR = Path(__file__).parent / "gcr"
OUTPUT_DIR = Path(__file__).parent / "reproducibility"
RESULTS_FILE = Path(__file__).parent / "reproducibility_results.json"

# Preços do Gemini 3.1 Pro Preview (USD por 1M tokens)
PRICE_INPUT = 2.00
PRICE_OUTPUT = 12.00


def parse_time_ms(log_text: str, pattern: str) -> float:
    """Extrai tempo em ms de um padrão de log."""
    match = re.search(pattern, log_text)
    if match:
        return float(match.group(1))
    return 0.0


def parse_tokens(log_text: str, pattern: str) -> dict:
    """Extrai tokens de um padrão de log."""
    match = re.search(pattern, log_text)
    if match:
        total = int(match.group(1))
        return {"total": total}
    return {"total": 0}


def parse_logs(log_text: str) -> dict:
    """Parseia logs da pipeline para extrair métricas."""
    metrics = {
        "test_generation": {
            "time_ms": 0.0,
            "tokens": {"total": 0}
        },
        "modularization_attempts": [],
        "total_api_calls": 0,
        "total_time_ms": 0.0,
        "total_tokens": 0,
        "status": "unknown"
    }
    
    # Extrair geração de testes
    test_time = parse_time_ms(
        log_text,
        r'\[forge\] Test generation complete \(([\d.]+) ms'
    )
    test_tokens = parse_tokens(
        log_text,
        r'\[forge\] Test generation complete \([\d.]+ ms, (\d+) tokens\)'
    )
    
    metrics["test_generation"]["time_ms"] = test_time
    metrics["test_generation"]["tokens"] = test_tokens
    
    # Extrair tentativas de modularização
    mod_attempts = []
    attempt_pattern = re.compile(
        r'\[forge\] LLM response received \(([\d.]+) ms, (\d+) tokens\)',
        re.MULTILINE
    )
    
    for match in attempt_pattern.finditer(log_text):
        time_ms = float(match.group(1))
        tokens_total = int(match.group(2))
        
        attempt_num = len(mod_attempts) + 1
        mod_attempts.append({
            "attempt": attempt_num,
            "time_ms": time_ms,
            "tokens": {"total": tokens_total}
        })
    
    metrics["modularization_attempts"] = mod_attempts
    
    # Verificar status final
    if "All equivalence tests passed" in log_text:
        metrics["status"] = "success"
    elif "ERROR" in log_text or "failed" in log_text:
        metrics["status"] = "failed"
    
    # Calcular totais
    metrics["total_api_calls"] = 1 + len(mod_attempts)  # 1 test + N modularize
    metrics["total_time_ms"] = test_time + sum(a["time_ms"] for a in mod_attempts)
    metrics["total_tokens"] = test_tokens["total"] + sum(a["tokens"]["total"] for a in mod_attempts)
    
    return metrics


def calculate_cost(tokens_total: int, attempts: int) -> float:
    """Calcula custo estimado em USD."""
    # Estimativa: 30% input, 70% output (baseado em observações)
    input_tokens = tokens_total * 0.30
    output_tokens = tokens_total * 0.70
    
    cost = (input_tokens / 1_000_000 * PRICE_INPUT) + (output_tokens / 1_000_000 * PRICE_OUTPUT)
    return cost


def run_single_benchmark(benchmark: str, run_number: int) -> dict:
    """Executa um benchmark uma vez e retorna métricas."""
    source_dir = BENCHMARKS_DIR / benchmark / "original"
    output_dir = OUTPUT_DIR / benchmark / f"run_{run_number:02d}"
    
    # Limpar output anterior
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Benchmark: {benchmark} | Run: {run_number}/{RUNS_PER_BENCHMARK}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")
    
    # Capturar stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    start_time = time.time()
    status = "success"
    error_message = None
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            pipeline_run(
                source_dir=str(source_dir),
                output_dir=str(output_dir),
                max_attempts=MAX_ATTEMPTS
            )
    except Exception as e:
        status = "failed"
        error_message = str(e)
    
    end_time = time.time()
    wall_time_ms = (end_time - start_time) * 1000
    
    # Parsear logs
    log_text = stdout_capture.getvalue()
    metrics = parse_logs(log_text)
    
    # Atualizar status
    if status == "failed":
        metrics["status"] = "failed"
        metrics["error"] = error_message
    elif metrics["status"] == "unknown":
        metrics["status"] = "success"
    
    # Adicionar metadados
    metrics["run_number"] = run_number
    metrics["benchmark"] = benchmark
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["wall_time_ms"] = wall_time_ms
    metrics["estimated_cost_usd"] = calculate_cost(metrics["total_tokens"], len(metrics["modularization_attempts"]))
    
    # Log resumo
    print(f"\n📊 Resultados:")
    print(f"  Status: {metrics['status']}")
    print(f"  Chamadas API: {metrics['total_api_calls']}")
    print(f"  Tempo total: {metrics['total_time_ms']:.2f} ms")
    print(f"  Tokens totais: {metrics['total_tokens']}")
    print(f"  Custo estimado: ${metrics['estimated_cost_usd']:.4f}")
    print(f"  Tentativas de modularização: {len(metrics['modularization_attempts'])}")
    
    return metrics


def calculate_statistics(runs: list) -> dict:
    """Calcula estatísticas (média, desvio padrão, min, max) para um conjunto de runs."""
    if not runs:
        return {}
    
    # Extrair métricas numéricas
    total_api_calls = [r["total_api_calls"] for r in runs if r["status"] == "success"]
    total_time_ms = [r["total_time_ms"] for r in runs if r["status"] == "success"]
    total_tokens = [r["total_tokens"] for r in runs if r["status"] == "success"]
    estimated_cost = [r["estimated_cost_usd"] for r in runs if r["status"] == "success"]
    
    def calc_stats(values):
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values)
        }
    
    success_runs = len([r for r in runs if r["status"] == "success"])
    success_rate = success_runs / len(runs) if runs else 0
    
    return {
        "total_runs": len(runs),
        "successful_runs": success_runs,
        "success_rate": success_rate,
        "total_api_calls": calc_stats(total_api_calls),
        "total_time_ms": calc_stats(total_time_ms),
        "total_tokens": calc_stats(total_tokens),
        "estimated_cost_usd": calc_stats(estimated_cost)
    }


def main():
    """Executa todos os benchmarks e salva resultados."""
    print("="*70)
    print("BENCHMARK DE REPRODUTIBILIDADE - PIPELINE FORGE")
    print(f"Modelo: gemini-3.1-pro-preview")
    print(f"Benchmarks: {', '.join(BENCHMARKS)}")
    print(f"Runs por benchmark: {RUNS_PER_BENCHMARK}")
    print(f"Total de execuções: {len(BENCHMARKS) * RUNS_PER_BENCHMARK}")
    print("="*70)
    
    # Estrutura de resultados
    results = {
        "metadata": {
            "model": "gemini-3.1-pro-preview",
            "total_runs": len(BENCHMARKS) * RUNS_PER_BENCHMARK,
            "runs_per_benchmark": RUNS_PER_BENCHMARK,
            "max_attempts": MAX_ATTEMPTS,
            "start_time": datetime.now().isoformat(),
            "benchmarks": BENCHMARKS
        },
        "benchmarks": {}
    }
    
    # Executar benchmarks
    for benchmark in BENCHMARKS:
        print(f"\n{'#'*70}")
        print(f"# BENCHMARK: {benchmark}")
        print(f"{'#'*70}")
        
        runs = []
        
        for run_number in range(1, RUNS_PER_BENCHMARK + 1):
            try:
                run_metrics = run_single_benchmark(benchmark, run_number)
                runs.append(run_metrics)
                
                # Salvar progresso incremental
                results["benchmarks"][benchmark] = {
                    "runs": runs,
                    "statistics": calculate_statistics(runs)
                }
                
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=2)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrompido pelo usuário. Salvando progresso...")
                results["benchmarks"][benchmark] = {
                    "runs": runs,
                    "statistics": calculate_statistics(runs)
                }
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=2)
                return
            
            except Exception as e:
                print(f"\n❌ Erro na execução {run_number}: {e}")
                runs.append({
                    "run_number": run_number,
                    "status": "error",
                    "error": str(e)
                })
        
        # Estatísticas finais do benchmark
        stats = calculate_statistics(runs)
        results["benchmarks"][benchmark]["statistics"] = stats
        
        print(f"\n{'='*70}")
        print(f"📈 ESTATÍSTICAS - {benchmark}")
        print(f"{'='*70}")
        print(f"Taxa de sucesso: {stats['success_rate']*100:.1f}% ({stats['successful_runs']}/{stats['total_runs']})")
        print(f"Chamadas API: {stats['total_api_calls']['mean']:.1f} ± {stats['total_api_calls']['std']:.1f}")
        print(f"Tempo total: {stats['total_time_ms']['mean']:.1f} ± {stats['total_time_ms']['std']:.1f} ms")
        print(f"Tokens totais: {stats['total_tokens']['mean']:.0f} ± {stats['total_tokens']['std']:.0f}")
        print(f"Custo estimado: ${stats['estimated_cost_usd']['mean']:.4f} ± ${stats['estimated_cost_usd']['std']:.4f}")
    
    # Salvar resultados finais
    results["metadata"]["end_time"] = datetime.now().isoformat()
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ EXECUÇÃO CONCLUÍDA")
    print(f"{'='*70}")
    print(f"Resultados salvos em: {RESULTS_FILE}")
    print(f"Total de execuções: {results['metadata']['total_runs']}")
    
    # Resumo geral
    print(f"\n{'='*70}")
    print(f"📊 RESUMO GERAL")
    print(f"{'='*70}")
    
    total_cost = 0
    total_runs = 0
    total_success = 0
    
    for benchmark in BENCHMARKS:
        if benchmark in results["benchmarks"]:
            stats = results["benchmarks"][benchmark]["statistics"]
            total_cost += stats["estimated_cost_usd"]["mean"] * stats["total_runs"]
            total_runs += stats["total_runs"]
            total_success += stats["successful_runs"]
    
    print(f"Execuções completadas: {total_runs}")
    print(f"Taxa de sucesso geral: {total_success/total_runs*100:.1f}% ({total_success}/{total_runs})")
    print(f"Custo total estimado: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
