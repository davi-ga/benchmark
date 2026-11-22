import os
import sys
import time

import numpy as np
from flask import Flask, g, jsonify, request

# A pasta de funções é o artefato produzido/consumido pelo forge.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "functions"))
import processing  # noqa: E402

app = Flask(__name__)
VERSION = os.getenv("VERSION", "unknown")

# Warm-up: absorve a compilação JIT (quando a variante for a do forge).
_ = processing.extract_qa_bits(np.zeros(16, dtype=np.uint16), 1)
print("[warmup] ok")


@app.before_request
def _mark_request_start():
    g.t0 = time.perf_counter()


@app.after_request
def _add_billed_header(resp):
    if hasattr(g, "t0"):
        resp.headers["X-Billed-Ms"] = f"{(time.perf_counter() - g.t0) * 1000:.3f}"
    return resp


@app.route("/qa", methods=["POST"])
def qa():
    data = request.get_json(silent=True) or {}
    band_data = np.array(data["band_data"], dtype=np.uint16)
    bit_location = int(data["bit_location"])

    start = time.perf_counter()
    result = processing.extract_qa_bits(band_data, bit_location)
    elapsed = time.perf_counter() - start
    billed_ms = (time.perf_counter() - g.t0) * 1000.0

    return jsonify(
        {
            "result": result.tolist() if hasattr(result, "tolist") else list(result),
            "time_ms": elapsed * 1000.0,
            "billed_ms": billed_ms,
            "version": VERSION,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": VERSION})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
