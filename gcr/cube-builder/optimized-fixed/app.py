from flask import Flask, jsonify, request
import numpy as np

from forge_middleware import FlaskMiddleware

app = Flask(__name__)
fm = FlaskMiddleware("forge-middleware.toml").init_app(app)

VERSION = "optimized-fixed"


@app.route("/qa", methods=["POST"])
def qa():
    data = request.json
    band_data = np.array(data["band_data"], dtype=np.uint16)
    bit_location = int(data["bit_location"])

    out = fm.call("extract_qa_bits", band_data, bit_location)
    result = out["result"]

    return jsonify(
        {
            "result": result.tolist() if hasattr(result, "tolist") else list(result),
            "time_ms": out["time_ms"],
            "cpu_avg": out["cpu_avg"],
            "mem_mb_avg": out["mem_mb_avg"],
            "fallback_used": out["fallback_used"],
            "version": VERSION,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
