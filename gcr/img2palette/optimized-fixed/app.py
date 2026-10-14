from flask import Flask, request, jsonify
import time
import sys
from io import BytesIO

import numpy as np
from PIL import Image
import requests

sys.path.insert(0, "optimized")
import image as image_mod  # noqa: E402

from sentinel import Middleware  # noqa: E402

app = Flask(__name__)

VERSION = "optimized-fixed"

mw = Middleware(modules=[image_mod])

# SEM warm-up deliberadamente: mede a compilacao JIT acontecendo no request.


@app.route("/palette", methods=["POST"])
def palette():
    data = request.json
    image_url = data.get("image_url")
    k = data.get("n_colors", 5)

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

    out = mw.call("get_palette", img, k)
    hex_colors = out["result"][1]

    return jsonify(
        {
            "palette": hex_colors.tolist()
            if hasattr(hex_colors, "tolist")
            else list(hex_colors),
            "time_ms": out["time_ms"],
            "cpu_avg": out["cpu_avg"],
            "mem_mb_avg": out["mem_mb_avg"],
            "fallback_used": out["fallback_used"],
            "version": VERSION,
        }
    )


@app.route("/palette-local", methods=["POST"])
def palette_local():
    """Test with a generated image instead of downloading."""
    data = request.json
    k = data.get("n_colors", 5)
    size = data.get("size", 100)

    img_array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")

    out = mw.call("get_palette", img, k)
    hex_colors = out["result"][1]

    return jsonify(
        {
            "palette": hex_colors.tolist()
            if hasattr(hex_colors, "tolist")
            else list(hex_colors),
            "time_ms": out["time_ms"],
            "cpu_avg": out["cpu_avg"],
            "mem_mb_avg": out["mem_mb_avg"],
            "fallback_used": out["fallback_used"],
            "version": VERSION,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": VERSION, "warmup": "none (desabilitado neste teste)"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
