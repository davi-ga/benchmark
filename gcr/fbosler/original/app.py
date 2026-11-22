import os
import sys
import time

import numpy as np
from flask import Flask, g, jsonify, request

# A pasta de funções é o artefato produzido/consumido pelo forge.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "functions"))
import scoring  # noqa: E402

app = Flask(__name__)
VERSION = os.getenv("VERSION", "unknown")

_VALS = [float(v) for v in np.linspace(-3.0, 3.0, 1000)]

# Warm-up: absorve a compilação JIT (quando a variante for a do forge).
_warm = 0.0
for _v in _VALS:
    _warm += scoring.scaled_sigmoid(_v)
print("[warmup] ok", _warm)


@app.before_request
def _mark_request_start():
    g.t0 = time.perf_counter()


@app.after_request
def _add_billed_header(resp):
    # Tempo de processamento no servidor (roteamento + handler + serialização),
    # mais próximo do que o Cloud Run cobra do que o compute interno do handler.
    if hasattr(g, "t0"):
        resp.headers["X-Billed-Ms"] = f"{(time.perf_counter() - g.t0) * 1000:.3f}"
    return resp


@app.route("/score", methods=["POST"])
def score():
    data = request.get_json(silent=True) or {}
    n = int(data.get("n", 200_000))
    start = time.perf_counter()
    total = 0.0
    for i in range(n):
        total += scoring.scaled_sigmoid(_VALS[i % 1000])
    elapsed = time.perf_counter() - start
    billed_ms = (time.perf_counter() - g.t0) * 1000.0
    return jsonify(
        {
            "time_ms": elapsed * 1000.0,
            "billed_ms": billed_ms,
            "n": n,
            "result": float(total),
            "version": VERSION,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": VERSION})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
