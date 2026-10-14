#!/usr/bin/env python3
"""
Fase 2 — gera as versões `optimized-fixed` com a pipeline atual e valida localmente.

Para cada app:
  1. Roda a pipeline forge 1× sobre `original/` (saída em `optimized-fixed/build/`).
  2. Copia os módulos otimizados para `optimized-fixed/optimized/`.
  3. Valida localmente com o Flask test client (health + endpoint principal).

NÃO faz deploy. O deploy é feito depois com `gcloud builds submit` a partir de
`optimized-fixed/`, criando os serviços `{app}-optimized-fixed` (sem sobrescrever
os serviços existentes).

Uso:
    python benchmarks/gcr/build_optimized_fixed.py
    python benchmarks/gcr/build_optimized_fixed.py --apps yuki-mt
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from build_pipeline import run  # noqa: E402

APPS = {
    "yuki-mt": {
        "modules": ["src/knn_feature.py"],
        "endpoint": "/predict",
        "payload": {"X": [[0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8, 0.9, 0.15] for _ in range(10)]},
    },
    "img2palette": {
        "modules": ["src/image.py"],
        "endpoint": "/palette-local",
        "payload": {"n_colors": 5, "size": 50},
    },
    "cube-builder": {
        "modules": ["src/maestro.py", "src/processing.py"],
        "endpoint": "/qa",
        "payload": {"band_data": [i % 256 for i in range(100)], "bit_location": 3},
    },
}

GCR = ROOT / "benchmarks" / "gcr"


def build_app(app: str) -> bool:
    cfg = APPS[app]
    src = GCR / app / "original"
    build = GCR / app / "optimized-fixed" / "build"
    dest = GCR / app / "optimized-fixed" / "optimized"

    print(f"\n{'=' * 70}\n# {app}: rodando pipeline\n{'=' * 70}")
    try:
        run(source_dir=str(src), output_dir=str(build), max_attempts=3)
    except Exception as exc:  # noqa: BLE001
        print(f"[{app}] PIPELINE FAIL: {exc}")
        return False

    dest.mkdir(parents=True, exist_ok=True)
    for rel in cfg["modules"]:
        shutil.copy(build / rel, dest / Path(rel).name)
    shutil.copy(build / "_numba_helpers.py", dest / "_numba_helpers.py")

    # Habilita cache do Numba nos módulos gerados (usado pelo warmup de build)
    for path in dest.glob("*.py"):
        if path.name == "_numba_helpers.py":
            continue
        text = path.read_text()
        text = text.replace("@numba.njit(cache=True)", "@numba.njit")  # idempotente
        text = text.replace("@njit(cache=True)", "@njit")
        text = text.replace("@numba.njit", "@numba.njit(cache=True)")
        text = text.replace("@njit", "@njit(cache=True)")
        path.write_text(text)

    print(f"[{app}] módulos copiados: {sorted(p.name for p in dest.glob('*.py'))}")
    return True


def validate_app(app: str) -> bool:
    cfg = APPS[app]
    app_dir = GCR / app / "optimized-fixed"
    script = (
        "import sys, json\n"
        "sys.path.insert(0, '.')\n"
        "import importlib\n"
        "mod = importlib.import_module('app')\n"
        "mod.app.testing = True\n"
        "c = mod.app.test_client()\n"
        "h = c.get('/health')\n"
        "assert h.status_code == 200, h.status_code\n"
        f"r = c.post({json.dumps(cfg['endpoint'])}, json={json.dumps(cfg['payload'])})\n"
        "assert r.status_code == 200, r.get_data(as_text=True)[:400]\n"
        "b = r.get_json()\n"
        f"print({json.dumps(app)}, 'OK', b.get('version'), round(b.get('time_ms', 0), 2), 'ms')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(app_dir),
    )
    if result.returncode != 0:
        print(f"[{app}] VALIDAÇÃO FALHOU:\n{result.stderr[-1500:]}")
        return False
    print(f"[{app}] {result.stdout.strip()}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apps", nargs="*", default=list(APPS))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    status = {}
    for app in args.apps:
        ok = True
        if not args.skip_build:
            ok = build_app(app)
        if ok:
            ok = validate_app(app)
        status[app] = ok

    print(f"\n{'=' * 70}\nRESUMO\n{'=' * 70}")
    for app, ok in status.items():
        print(f"  {app}: {'OK' if ok else 'FALHOU'}")
    print("\nPróximo passo: deploy com gcloud builds submit (ver README).")
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
