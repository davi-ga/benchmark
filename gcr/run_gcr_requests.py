#!/usr/bin/env python3
"""
Reprodutibilidade de runtime no GCR — N requests por repo, com tempo e custo.

Para cada app e cada variante (original, optimized) nos serviços Cloud Run:
  1. Aquecimento: o primeiro request é registrado como "cold" (absorve partida a
     frio e compilação JIT); os demais aquecem a instância.
  2. N requests medidos ("warm"), registrando CADA execução:
       wall_ms    — latência de parede observada pelo cliente
       server_ms  — time_ms do compute interno do handler
       billed_ms  — tempo de processamento medido no servidor (quando o app expõe
                    X-Billed-Ms / billed_ms); base preferencial do custo
       cost_usd   — custo estimado (CPU + memória + taxa por request)

Custo (Cloud Run, Tier 1 us-central1):
  CPU $0,000024/vCPU-s | memória $0,0000025/GiB-s | requests $0,40/1M
  Duração cobrada arredondada para cima em 100 ms (definição oficial).

--validate-metrics consulta o Cloud Monitoring (billable_instance_time e
request_count) e compara com o que foi medido localmente.

Uso:
    python benchmarks/gcr/run_gcr_requests.py --project tcc-dev-496317 --requests 30
    python benchmarks/gcr/run_gcr_requests.py --project tcc-dev-496317 --apps fbosler --validate-metrics
"""

import argparse
import json
import math
import random
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BASE_URL = "https://{svc}-b56c2yqz6q-uc.a.run.app{path}"

# Preços de referência (Tier 1, us-central1).
CPU_PRICE_PER_VCPU_SEC = 0.000024
MEM_PRICE_PER_GIB_SEC = 0.0000025
REQUEST_PRICE = 0.40 / 1e6
BILLING_INCREMENT_MS = 100

APPS = {
    "yuki-mt": {
        "svc": "yuki-mt",
        "path": "/predict",
        "payload": {"X": [[0.5] * 10 for _ in range(100)]},
    },
    "img2palette": {
        "svc": "img2palette",
        "path": "/palette-local",
        "payload": {"n_colors": 5, "size": 100},
    },
    "cube-builder": {
        "svc": "cube-builder",
        "path": "/qa",
        "payload": {"band_data": [i % 256 for i in range(1000)], "bit_location": 3},
    },
    "fbosler": {
        "svc": "fbosler",
        "path": "/score",
        "payload": {"n": 200_000},
    },
}

RESULTS_FILE = Path(__file__).parent / "gcr_reproducibility_results.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gcloud_json(args: list[str]) -> dict | None:
    proc = subprocess.run(args, capture_output=True, text=True)
    try:
        return json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        return None


def resolve_url(svc: str, path: str, project: str | None, region: str, base_url: str) -> str:
    if project:
        data = _gcloud_json(
            ["gcloud", "run", "services", "describe", svc, "--project", project,
             "--region", region, "--format", "json"]
        )
        url = (data or {}).get("status", {}).get("url")
        if url:
            return url + path
        print(f"    WARNING: could not resolve {svc} via gcloud; falling back to --base-url")
    return base_url.format(svc=svc, path=path)


def parse_memory_gib(mem: str) -> float:
    mem = (mem or "").strip()
    if mem.endswith("Gi"):
        return float(mem[:-2])
    if mem.endswith("Mi"):
        return float(mem[:-2]) / 1024.0
    if mem.endswith("G"):
        return float(mem[:-1]) * 1e9 / (1024.0 ** 3)
    if mem.endswith("M"):
        return float(mem[:-1]) * 1e6 / (1024.0 ** 3)
    try:
        return float(mem)
    except ValueError:
        return 0.5


def resolve_resources(
    svc: str, project: str | None, region: str, default_cpu: float, default_mem_gb: float
) -> tuple[float, float]:
    if project:
        data = _gcloud_json(
            ["gcloud", "run", "services", "describe", svc, "--project", project,
             "--region", region, "--format", "json"]
        )
        try:
            limits = data["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
            return float(str(limits.get("cpu", default_cpu))), parse_memory_gib(
                str(limits.get("memory", f"{default_mem_gb}Gi"))
            )
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    return default_cpu, default_mem_gb


def cost_basis(run: dict) -> tuple[float | None, str | None]:
    """Duration used for billing: billed_ms > server_ms."""
    if run.get("billed_ms") is not None:
        return run["billed_ms"], "billed_ms"
    if run.get("server_ms") is not None:
        return run["server_ms"], "server_ms"
    return None, None


def request_cost(duration_ms: float | None, cpu: float, mem_gb: float) -> float | None:
    if duration_ms is None:
        return None
    billed_s = math.ceil(duration_ms / BILLING_INCREMENT_MS) * BILLING_INCREMENT_MS / 1000.0
    return billed_s * (cpu * CPU_PRICE_PER_VCPU_SEC + mem_gb * MEM_PRICE_PER_GIB_SEC) + REQUEST_PRICE


def bootstrap_ci(values: list[float], iterations: int = 5000, alpha: float = 0.05) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    means = []
    n = len(values)
    for _ in range(iterations):
        means.append(sum(random.choices(values, k=n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * iterations)]
    hi = means[int((1 - alpha / 2) * iterations) - 1]
    return [lo, hi]


def make_request(url: str, payload: dict, timeout: int = 180):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            header_billed = response.headers.get("X-Billed-Ms")
            body = json.loads(response.read().decode("utf-8"))
        wall_ms = (time.perf_counter() - start) * 1000
        billed = body.get("billed_ms")
        if billed is None and header_billed is not None:
            try:
                billed = float(header_billed)
            except ValueError:
                billed = None
        return {
            "ok": True,
            "wall_ms": wall_ms,
            "server_ms": body.get("time_ms"),
            "billed_ms": billed,
        }
    except Exception as exc:  # noqa: BLE001
        wall_ms = (time.perf_counter() - start) * 1000
        return {"ok": False, "wall_ms": wall_ms, "server_ms": None, "billed_ms": None, "error": str(exc)}


def _finalize_run(run: dict, cpu: float, mem_gb: float) -> dict:
    duration, basis = cost_basis(run)
    run["cost_basis"] = basis
    run["cost_usd"] = request_cost(duration, cpu, mem_gb)
    return run


def stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "iqr": 0, "ci95": [0, 0]}
    s = sorted(values)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    return {
        "mean": statistics.mean(s),
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "median": statistics.median(s),
        "iqr": q3 - q1,
        "ci95": bootstrap_ci(s),
    }


def measure(url: str, payload: dict, n: int, warmups: int, cpu: float, mem_gb: float) -> dict:
    cold_run = None
    for i in range(max(warmups, 0)):
        r = make_request(url, payload)
        if not r["ok"]:
            return {"error": f"warm-up failed: {r.get('error')}", "runs": []}
        if i == 0:
            cold_run = _finalize_run(r, cpu, mem_gb)

    runs = []
    for i in range(n):
        r = _finalize_run(make_request(url, payload), cpu, mem_gb)
        if r["ok"]:
            print(
                f"    req {i + 1}/{n}: wall={r['wall_ms']:.1f}ms server={r['server_ms']:.1f}ms "
                f"billed={r['billed_ms'] if r['billed_ms'] is not None else '-'} "
                f"cost=${r['cost_usd']:.8f} [{r['cost_basis']}]"
            )
        else:
            print(f"    req {i + 1}/{n}: FAIL {r.get('error')}")
        runs.append(r)

    ok = [r for r in runs if r["ok"]]
    wall = [r["wall_ms"] for r in ok]
    server = [r["server_ms"] for r in ok if r["server_ms"] is not None]
    billed = [r["billed_ms"] for r in ok if r["billed_ms"] is not None]
    costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]

    cost_stats = stats(costs)
    return {
        "cold": cold_run,
        "n_requested": n,
        "n_ok": len(ok),
        "resources": {"cpu": cpu, "mem_gb": mem_gb},
        "wall_ms": stats(wall),
        "server_ms": stats(server),
        "billed_ms": stats(billed),
        "cost_usd": cost_stats,
        "cost_per_million": cost_stats["mean"] * 1e6,
        "runs": runs,
    }


def query_monitoring(project: str, metric: str, svc: str, start_iso: str, end_iso: str) -> dict:
    """Sum a Cloud Monitoring metric for one service over [start, end]."""
    try:
        token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"token: {exc}"[:200]}
    flt = f'metric.type="{metric}" AND resource.labels.service_name="{svc}"'
    qs = urllib.parse.urlencode({
        "filter": flt,
        "interval.startTime": start_iso,
        "interval.endTime": end_iso,
        "aggregation.alignmentPeriod": "60s",
        "aggregation.perSeriesAligner": "ALIGN_SUM",
        "pageSize": "200",
    })
    url = f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries?{qs}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}
    total = 0.0
    points = 0
    for ts in data.get("timeSeries", []):
        for p in ts.get("points", []):
            value = p.get("value", {})
            total += float(value.get("doubleValue", value.get("int64Value", 0)))
            points += 1
    return {"total": total, "points": points}


def validate_service(svc: str, project: str, start_iso: str, end_iso: str, measured: dict) -> dict:
    billable = query_monitoring(
        project, "run.googleapis.com/container/billable_instance_time", svc, start_iso, end_iso
    )
    requests = query_monitoring(project, "run.googleapis.com/request_count", svc, start_iso, end_iso)
    expected_s = 0.0
    for run in measured.get("runs", []):
        duration, _ = cost_basis(run)
        if duration is not None:
            expected_s += duration / 1000.0
    result = {
        "billable_instance_time_s": billable,
        "request_count": requests,
        "expected_billed_s_from_measured": expected_s,
        "n_ok": measured.get("n_ok", 0),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--apps", nargs="*", default=list(APPS))
    parser.add_argument("--variants", nargs="*", default=["original", "optimized"])
    parser.add_argument("--project", default=None, help="GCP project; resolve URLs/resources via gcloud.")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="URL template used when --project is not given: 'https://{svc}-HASH.a.run.app{path}'.",
    )
    parser.add_argument("--cpu", type=float, default=1.0, help="Fallback vCPU when --project is not given.")
    parser.add_argument(
        "--memory-gb", type=float, default=0.5, help="Fallback GiB when --project is not given."
    )
    parser.add_argument(
        "--validate-metrics",
        action="store_true",
        help="Consult billable_instance_time/request_count and compare with measured values.",
    )
    parser.add_argument("--output", default=str(RESULTS_FILE))
    args = parser.parse_args()

    collect_start = now_utc()
    results = {
        "metadata": {
            "start_time": collect_start,
            "requests_per_service": args.requests,
            "warmup_per_service": args.warmups,
            "variants": args.variants,
            "project": args.project,
            "region": args.region,
            "base_url": args.base_url,
            "prices": {
                "cpu_per_vcpu_sec": CPU_PRICE_PER_VCPU_SEC,
                "mem_per_gib_sec": MEM_PRICE_PER_GIB_SEC,
                "per_request": REQUEST_PRICE,
                "billing_increment_ms": BILLING_INCREMENT_MS,
            },
            "note": "cold=primeiro request; warm=N medidos; custo usa billed_ms (fallback server_ms)",
        },
        "apps": {},
    }

    for app in args.apps:
        cfg = APPS[app]
        print(f"\n{'=' * 70}\n# {app}\n{'=' * 70}")
        app_result = {}
        for variant in args.variants:
            svc = f"{cfg['svc']}-{variant}"
            url = resolve_url(svc, cfg["path"], args.project, args.region, args.base_url)
            cpu, mem_gb = resolve_resources(
                svc, args.project, args.region, args.cpu, args.memory_gb
            )
            print(f"\n  [{variant}] {url}  (cpu={cpu}, mem={mem_gb}GiB)")
            app_result[variant] = measure(
                url, cfg["payload"], args.requests, args.warmups, cpu, mem_gb
            )
            app_result[variant]["url"] = url
            if args.validate_metrics and args.project:
                print("    validando contra o Cloud Monitoring...")
                app_result[variant]["validation"] = validate_service(
                    svc, args.project, collect_start, now_utc(), app_result[variant]
                )

        base = args.variants[0]
        for variant in args.variants[1:]:
            orig = app_result[base].get("server_ms", {}).get("mean", 0)
            opt = app_result[variant].get("server_ms", {}).get("mean", 0)
            app_result[f"speedup_server_{base}_vs_{variant}"] = (orig / opt) if opt else 0.0
            orig_w = app_result[base].get("wall_ms", {}).get("mean", 0)
            opt_w = app_result[variant].get("wall_ms", {}).get("mean", 0)
            app_result[f"speedup_wall_{base}_vs_{variant}"] = (orig_w / opt_w) if opt_w else 0.0
            orig_c = app_result[base].get("cost_usd", {}).get("mean", 0)
            opt_c = app_result[variant].get("cost_usd", {}).get("mean", 0)
            app_result[f"speedup_cost_{base}_vs_{variant}"] = (orig_c / opt_c) if opt_c else 0.0
            print(
                f"\n  -> {base} vs {variant}: server={orig:.1f}->{opt:.1f}ms "
                f"({app_result[f'speedup_server_{base}_vs_{variant}']:.2f}x) | "
                f"custo=${orig_c:.8f}->${opt_c:.8f}/req "
                f"({app_result[f'speedup_cost_{base}_vs_{variant}']:.2f}x)"
            )
        results["apps"][app] = app_result

    results["metadata"]["end_time"] = now_utc()
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 92}")
    print("RESUMO — tempo e custo médios por request (IC 95% bootstrap)")
    print(f"{'=' * 92}")
    print(f"{'app':<14}{'variant':<11}{'server_ms':>11}{'billed_ms':>11}{'wall_ms':>10}{'cost/req':>14}{'cost/1M':>10}{'IC95 server':>20}")
    for app, r in results["apps"].items():
        for v in args.variants:
            s = r[v].get("server_ms", {})
            b = r[v].get("billed_ms", {})
            w = r[v].get("wall_ms", {})
            c = r[v].get("cost_usd", {})
            ci = s.get("ci95", [0, 0])
            print(
                f"{app:<14}{v:<11}{s.get('mean', 0):>11.1f}{b.get('mean', 0):>11.1f}"
                f"{w.get('mean', 0):>10.1f}{('$%.8f' % c.get('mean', 0)):>14}"
                f"{('$%.2f' % (c.get('mean', 0) * 1e6)):>10}"
                f"{('[%.1f, %.1f]' % (ci[0], ci[1])):>20}"
            )

    if args.validate_metrics:
        print(f"\n{'=' * 92}")
        print("VALIDAÇÃO — Cloud Monitoring")
        print(f"{'=' * 92}")
        for app, r in results["apps"].items():
            for v in args.variants:
                val = r[v].get("validation")
                if not val:
                    continue
                bill = val["billable_instance_time_s"]
                reqs = val["request_count"]
                print(
                    f"{app}/{v}: billable={bill.get('total', 0):.2f}s ({bill.get('points', 0)} pts) | "
                    f"requests={reqs.get('total', 0):.0f} ({reqs.get('points', 0)} pts) | "
                    f"esperado(medido)={val['expected_billed_s_from_measured']:.2f}s / n={val['n_ok']}"
                )
    print(f"\nResultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
