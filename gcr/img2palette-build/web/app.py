import os
import sys
import time
from io import BytesIO

import numpy as np
import requests
from flask import Flask, g, jsonify, request
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# A pasta de funções é o artefato produzido/consumido pelo forge.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "functions"))
import image  # noqa: E402

app = Flask(__name__)
VERSION = os.getenv("VERSION", "unknown")
default_k = 4


def get_lab_data(im):
    img_size = 150, 150
    im.thumbnail(img_size)
    pixel_rgb = np.asarray(im)
    pixel_lab = image.rgb2lab(pixel_rgb.reshape(-1, pixel_rgb.shape[-1]) / 255)
    return pixel_lab.reshape(-1, pixel_lab.shape[-1])


def make_img(colors, counts):
    img_size = 512, 512
    n_clusters = len(colors)
    lengths = (
        ((counts / np.sum(counts)) + (1.0 / n_clusters)) / 2.0 * img_size[0]
    ).astype(np.uint16)
    lengths[0] = lengths[0] + (img_size[0] - np.sum(lengths))
    pixel_group = np.concatenate(
        [np.tile(colors[i], (lengths[i], img_size[1], 1)) for i in range(n_clusters)],
        axis=0,
    )
    pixel_rgb = np.transpose(pixel_group, (1, 0, 2))
    return Image.fromarray(pixel_rgb, mode="RGB")


def compute_medoid(data):
    dist_mat = pairwise_distances(data)
    return data[np.argmin(dist_mat.sum(axis=0))]


def cluster_agglo(data, n_clusters):
    ac = AgglomerativeClustering(n_clusters)
    labels = ac.fit_predict(data)
    centers = np.empty([n_clusters, 3])
    for i in range(n_clusters):
        centers[i] = compute_medoid(data[labels == i])
    return centers, labels


def get_palette(im, k):
    if k is None:
        k = default_k
    elif k < 2 or k > 10:
        k = default_k
    data = get_lab_data(im)
    centers, labels = cluster_agglo(data, k)
    sorted_centers, _, counts = image.get_cluster(centers, labels)
    centers_rgb = (255 * image.lab2rgb(sorted_centers)).astype(np.uint8)
    return (
        make_img(centers_rgb, counts),
        np.apply_along_axis(image.get_hex_string, 1, centers_rgb),
    )


# Warm-up: absorve a compilação JIT (quando a variante for a do forge).
_warm_rgb = np.array([[0.5, 0.4, 0.3], [0.2, 0.8, 0.1]], dtype=np.float64)
_ = image.rgb2lab(_warm_rgb)
_ = image.lab2rgb(image.rgb2lab(_warm_rgb))
print("[warmup] ok")


@app.before_request
def _mark_request_start():
    g.t0 = time.perf_counter()


@app.after_request
def _add_billed_header(resp):
    if hasattr(g, "t0"):
        resp.headers["X-Billed-Ms"] = f"{(time.perf_counter() - g.t0) * 1000:.3f}"
    return resp


@app.route("/palette", methods=["POST"])
def palette():
    data = request.get_json(silent=True) or {}
    image_url = data.get("image_url")
    k = data.get("n_colors", 5)

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

    start = time.perf_counter()
    _, hex_colors = get_palette(img, k)
    elapsed = time.perf_counter() - start
    billed_ms = (time.perf_counter() - g.t0) * 1000.0

    return jsonify(
        {
            "palette": hex_colors.tolist() if hasattr(hex_colors, "tolist") else list(hex_colors),
            "time_ms": elapsed * 1000.0,
            "billed_ms": billed_ms,
            "version": VERSION,
        }
    )


@app.route("/palette-local", methods=["POST"])
def palette_local():
    data = request.get_json(silent=True) or {}
    k = data.get("n_colors", 5)
    size = data.get("size", 100)

    img_array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")

    start = time.perf_counter()
    _, hex_colors = get_palette(img, k)
    elapsed = time.perf_counter() - start
    billed_ms = (time.perf_counter() - g.t0) * 1000.0

    return jsonify(
        {
            "palette": hex_colors.tolist() if hasattr(hex_colors, "tolist") else list(hex_colors),
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
