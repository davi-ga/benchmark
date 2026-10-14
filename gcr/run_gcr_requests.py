#!/usr/bin/env python3
"""
Reprodutibilidade de runtime no GCR — 10 requests por repo.

Para cada app (yuki-mt, img2palette, cube-builder) e cada variante
(original, optimized) nos serviços Cloud Run já deployados:
  1. 1 request de warm-up (descartado — absorve cold start / compilação JIT)
  2. 10 requests medidos
Registra latência de parede (wall-clock) e o `time_ms` reportado pelo servidor,
calcula estatísticas (média, desvio-padrão, min, max) e o speedup.

Uso:
    python benchmarks/gcr/run_gcr_requests.py
    python benchmarks/gcr/run_gcr_requests.py --requests 10
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

API = "https://{svc}-b56c2yqz6q-uc.a.run.app{path}"

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
}

RESULTS_FILE = Path(__file__).parent / "gcr_reproducibility_results.json"


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
    # Warm-up (descartado) — pode ser desligado (warmups=0) para medir tambem
    # o primeiro request (compilacao JIT no request)
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
    parser.add_argument("--output", default=str(RESULTS_FILE))
    args = parser.parse_args()

    results = {
        "metadata": {
            "start_time": datetime.now().isoformat(),
            "requests_per_service": args.requests,
            "warmup_per_service": args.warmups,
            "variants": args.variants,
            "note": "wall=latência de parede; server=time_ms reportado pelo serviço",
        },
        "apps": {},
    }

    for app in args.apps:
        cfg = APPS[app]
        print(f"\n{'=' * 70}\n# {app}\n{'=' * 70}")
        app_result = {}
        for variant in args.variants:
            url = API.format(svc=f"{cfg['svc']}-{variant}", path=cfg["path"])
            print(f"\n  [{variant}] {url}")
            app_result[variant] = measure(url, cfg["payload"], args.requests, args.warmups)
            app_result[variant]["url"] = url

        base = args.variants[0]
        for variant in args.variants[1:]:
            orig = app_result[base].get("server_ms", {}).get("mean", 0)
            opt = app_result[variant].get("server_ms", {}).get("mean", 0)
            app_result[f"speedup_server_{base}_vs_{variant}"] = (orig / opt) if opt else 0.0
            print(
                f"\n  -> {base} server={orig:.1f}ms  {variant} server={opt:.1f}ms  "
                f"speedup={app_result[f'speedup_server_{base}_vs_{variant}']:.2f}x"
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
