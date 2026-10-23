#!/usr/bin/env python3
"""
Reprodutibilidade de runtime no GCR — N requests por repo.

Para cada app e cada variante (original, optimized) nos serviços Cloud Run:
  1. 1 request de warm-up (descartado — absorve cold start / compilação JIT)
  2. N requests medidos, registrando CADA execução (wall_ms e time_ms do servidor)

As URLs podem ser resolvidas dinamicamente via `gcloud run services describe`
(--project/--region) ou por um template (--base-url).

Uso:
    python benchmarks/gcr/run_gcr_requests.py --project tcc-dev-496317 --region us-central1
    python benchmarks/gcr/run_gcr_requests.py --project tcc-dev-496317 --apps fbosler --requests 10
    python benchmarks/gcr/run_gcr_requests.py --base-url "https://{svc}-HASH-uc.a.run.app{path}"
"""

import argparse
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

DEFAULT_BASE_URL = "https://{svc}-b56c2yqz6q-uc.a.run.app{path}"

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


def resolve_url(svc: str, path: str, project: str | None, region: str, base_url: str) -> str:
    """Resolve the service URL via gcloud when --project is given, else use the template."""
    if project:
        proc = subprocess.run(
            [
                "gcloud", "run", "services", "describe", svc,
                "--project", project,
                "--region", region,
                "--format", "value(status.url)",
            ],
            capture_output=True,
            text=True,
        )
        base = proc.stdout.strip()
        if base:
            return base + path
        print(f"    WARNING: could not resolve {svc} via gcloud; falling back to --base-url")
    return base_url.format(svc=svc, path=path)


def make_request(url: str, payload: dict, timeout: int = 180):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        wall_ms = (time.perf_counter() - start) * 1000
        return {"ok": True, "wall_ms": wall_ms, "server_ms": body.get("time_ms")}
    except Exception as exc:  # noqa: BLE001
        wall_ms = (time.perf_counter() - start) * 1000
        return {"ok": False, "wall_ms": wall_ms, "server_ms": None, "error": str(exc)}


def measure(url: str, payload: dict, n: int, warmups: int = 1) -> dict:
    warm = {"wall_ms": None}
    for _ in range(max(warmups, 0)):
        warm = make_request(url, payload)
        if not warm["ok"]:
            return {"error": f"warm-up failed: {warm.get('error')}", "runs": []}

    runs = []
    for i in range(n):
        r = make_request(url, payload)
        print(
            f"    req {i + 1}/{n}: "
            + (f"wall={r['wall_ms']:.1f}ms server={r['server_ms']:.1f}ms" if r["ok"] else f"FAIL {r.get('error')}")
        )
        runs.append(r)

    ok = [r for r in runs if r["ok"]]
    wall = [r["wall_ms"] for r in ok]
    server = [r["server_ms"] for r in ok if r["server_ms"] is not None]

    def stats(values):
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    return {
        "warmup_wall_ms": warm["wall_ms"],
        "n_requested": n,
        "n_ok": len(ok),
        "wall_ms": stats(wall),
        "server_ms": stats(server),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--apps", nargs="*", default=list(APPS))
    parser.add_argument("--variants", nargs="*", default=["original", "optimized"])
    parser.add_argument("--project", default=None, help="GCP project; resolve URLs via gcloud.")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="URL template used when --project is not given: 'https://{svc}-HASH.a.run.app{path}'.",
    )
    parser.add_argument("--output", default=str(RESULTS_FILE))
    args = parser.parse_args()

    results = {
        "metadata": {
            "start_time": datetime.now().isoformat(),
            "requests_per_service": args.requests,
            "warmup_per_service": args.warmups,
            "variants": args.variants,
            "project": args.project,
            "region": args.region,
            "base_url": args.base_url,
            "note": "wall=latência de parede; server=time_ms reportado pelo serviço",
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
            print(f"\n  [{variant}] {url}")
            app_result[variant] = measure(url, cfg["payload"], args.requests, args.warmups)
            app_result[variant]["url"] = url

        base = args.variants[0]
        for variant in args.variants[1:]:
            orig = app_result[base].get("server_ms", {}).get("mean", 0)
            opt = app_result[variant].get("server_ms", {}).get("mean", 0)
            app_result[f"speedup_server_{base}_vs_{variant}"] = (orig / opt) if opt else 0.0
            orig_w = app_result[base].get("wall_ms", {}).get("mean", 0)
            opt_w = app_result[variant].get("wall_ms", {}).get("mean", 0)
            app_result[f"speedup_wall_{base}_vs_{variant}"] = (orig_w / opt_w) if opt_w else 0.0
            print(
                f"\n  -> {base} server={orig:.1f}ms  {variant} server={opt:.1f}ms  "
                f"speedup_server={app_result[f'speedup_server_{base}_vs_{variant}']:.2f}x  "
                f"speedup_wall={app_result[f'speedup_wall_{base}_vs_{variant}']:.2f}x"
            )
        results["apps"][app] = app_result

    results["metadata"]["end_time"] = datetime.now().isoformat()
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("RESUMO — latência média (server_ms)")
    print(f"{'=' * 70}")
    header = f"{'app':<14}" + "".join(f"{v:>22}" for v in args.variants)
    print(header)
    for app, r in results["apps"].items():
        row = f"{app:<14}"
        for v in args.variants:
            row += f"{r[v].get('server_ms', {}).get('mean', 0):>20.1f}ms"
        print(row)
    print(f"\nResultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
