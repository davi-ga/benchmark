#!/usr/bin/env python3
import json
import urllib.request
import urllib.error
import time
import numpy as np

SERVICES = {
    "original": "https://yuki-mt-original-b56c2yqz6q-uc.a.run.app",
    "optimized-pro": "https://yuki-mt-optimized-b56c2yqz6q-uc.a.run.app",
    "optimized-flash": "https://yuki-mt-flash-b56c2yqz6q-uc.a.run.app",
}

SAMPLES = [50, 100, 200, 500]
FEATURES = 10


def make_payload(n_samples):
    X = np.random.rand(n_samples, FEATURES).tolist()
    return json.dumps({"X": X}).encode("utf-8")


def call_service(url, payload, timeout=180):
    req = urllib.request.Request(
        f"{url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.perf_counter() - start
        return {"status": "ok", "elapsed_ms": elapsed * 1000, "server_ms": body.get("time_ms", 0)}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"status": "error", "elapsed_ms": elapsed * 1000, "error": str(e)}


def main():
    results = []

    for n in SAMPLES:
        print(f"\n=== {n} samples x {FEATURES} features ===")
        payload = make_payload(n)
        row = {"samples": n}

        for name, url in SERVICES.items():
            resp = call_service(url, payload)
            row[name] = resp["elapsed_ms"] if resp["status"] == "ok" else None
            if resp["status"] == "ok":
                print(f"  {name}: {resp['elapsed_ms']:.2f}ms (server: {resp['server_ms']:.2f}ms)")
            else:
                print(f"  {name}: ERROR - {resp['error']}")

        results.append(row)

    print("\n=== RESUMO ===")
    print(f"{'Samples':<10} {'Original':<15} {'Pro':<15} {'Flash':<15} {'Speedup Pro':<15} {'Speedup Flash':<15}")
    for r in results:
        orig = r.get("original")
        pro = r.get("optimized-pro")
        flash = r.get("optimized-flash")
        sp = f"{orig/pro:.2f}x" if orig and pro else "-"
        sf = f"{orig/flash:.2f}x" if orig and flash else "-"
        print(
            f"{r['samples']:<10} "
            f"{orig if orig else '-':<15.2f} "
            f"{pro if pro else '-':<15.2f} "
            f"{flash if flash else '-':<15.2f} "
            f"{sp:<15} "
            f"{sf:<15}"
        )

    with open("compare_models.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em compare_models.json")


if __name__ == "__main__":
    main()
