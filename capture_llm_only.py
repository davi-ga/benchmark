#!/usr/bin/env python3
"""Captura e valida a saída da variante LLM-Only (prompt de anotação).

Replica exatamente o braço ``llm_only_docs()`` de ``benchmarks/validacao_local.py``:
  - entrada: código BRUTO do arquivo principal (sem pré-processamento AST);
  - prompt: ``PROMPT_NUMBA_STRONG`` (anotação, em português);
  - chamada: ``generate_content`` com ``temperature=0.2``, saída em texto livre.

Além de persistir a resposta crua, o script difa o resultado contra o original
e resume o que o modelo alterou (``print`` removidos, ``np.vectorize``/``lambda``
reescritos, ``@numba.njit`` adicionados etc.). Com ``--measure``, reutiliza a
sonda do app para medir o código capturado e conferir se ele reproduz o ganho
reportado (ex.: 1,25x no img2palette) e se permanece correto frente ao original.

Uso:
    PYTHONPATH=src .venv/bin/python benchmarks/capture_llm_only.py [app] [--measure]
    PYTHONPATH=src .venv/bin/python benchmarks/capture_llm_only.py img2palette --measure

Requer ``GEMINI_API_KEY`` (e opcionalmente ``GEMINI_MODEL``, default
``gemini-3.1-pro-preview``).
"""

import argparse
import difflib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from experimento_hibrido import (  # noqa: E402
    MAIN_FILE,
    PROBES,
    PROMPT_NUMBA_STRONG,
    count_njit,
    import_from_dir,
)

GCR = ROOT / "benchmarks" / "gcr"
NUM_WARM = 3
NUM_TIMED = 10


def strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def build_diff(original: str, generated: str) -> tuple[list[str], list[str], list[str]]:
    diff = list(
        difflib.unified_diff(
            original.splitlines(),
            generated.splitlines(),
            fromfile="original",
            tofile="llm_only",
            lineterm="",
        )
    )
    added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
    return diff, added, removed


# Padrões cuja presença/ausência explica o ganho sem @numba.njit.
HEURISTICS = [
    ("print(", "print() removido"),
    ("np.vectorize", "np.vectorize alterado"),
    ("list(sort_idx).index", "list(sort_idx).index reescrito"),
    ("lambda", "lambda alterada"),
    ("np.apply_along_axis", "np.apply_along_axis alterado"),
    ("np.where", "np.where alterado"),
    ("@numba.njit", "@numba.njit adicionado"),
]


def run_heuristics(original: str, generated: str) -> list[tuple[str, bool, bool]]:
    ol = original.lower()
    gl = generated.lower()
    rows = []
    for needle, label in HEURISTICS:
        rows.append((label, needle in ol, needle in gl))
    return rows


def capture(app: str) -> Path:
    main = MAIN_FILE[app]
    original_path = GCR / app / "original" / main
    if not original_path.is_file():
        raise SystemExit(f"[capture] original não encontrado: {original_path}")
    original_code = original_path.read_text()

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
    if not api_key:
        raise SystemExit("[capture] GEMINI_API_KEY não definido")

    client = genai.Client(api_key=api_key)
    contents = f"{PROMPT_NUMBA_STRONG}\n\n```\n{original_code}\n```"

    print(f"[capture] app={app} arquivo={main} modelo={model_name} temperature=0.2")
    start = time.perf_counter()
    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    elapsed = time.perf_counter() - start

    raw = resp.text or ""
    generated = strip_fences(raw)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "benchmarks" / "experimento_output" / f"llm_only_{app}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "prompt.txt").write_text(PROMPT_NUMBA_STRONG, encoding="utf-8")
    (out_dir / "input_original.py").write_text(original_code, encoding="utf-8")
    (out_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
    (out_dir / "output_llm_only.py").write_text(generated, encoding="utf-8")

    diff, added, removed = build_diff(original_code, generated)
    (out_dir / "diff.patch").write_text("\n".join(diff), encoding="utf-8")

    stats = {
        "app": app,
        "arquivo": main,
        "modelo": model_name,
        "temperature": 0.2,
        "elapsed_ms": round(elapsed * 1000, 2),
        "tokens": {
            "entrada": resp.usage_metadata.prompt_token_count,
            "saida": resp.usage_metadata.candidates_token_count,
            "total": resp.usage_metadata.total_token_count,
        },
        "njit_na_saida": generated.count("@numba.njit"),
        "linhas_removidas": len(removed),
        "linhas_adicionadas": len(added),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[capture] tempo={stats['elapsed_ms']}ms tokens={stats['tokens']['total']}")
    print(f"[capture] @numba.njit na saída: {stats['njit_na_saida']}")
    print(f"[capture] diff: -{len(removed)} / +{len(added)} linhas")
    print(f"[capture] artefatos em {out_dir}")

    print("\n[análise] o que o modelo mudou:")
    changed = False
    for label, before, after in run_heuristics(original_code, generated):
        if before != after:
            changed = True
            print(f"  - {label}: {'sim' if before else 'não'} → {'sim' if after else 'não'}")
    if not changed:
        print("  (nenhum dos padrões rastreados mudou; ver diff.patch)")

    return out_dir


def measure(app: str, generated: str) -> None:
    main = MAIN_FILE[app]
    probe_file, make_ref, make_callable, _fallback = PROBES[app](None, "llm_only")

    tmp = Path(tempfile.mkdtemp(prefix=f"vl_{app}_llmonly_"))
    try:
        (tmp / Path(main).name).write_text(generated)

        ref_vals = make_ref()
        mod = import_from_dir(tmp, Path(main).stem)
        njit_count, _compiled = count_njit(mod)
        fn = make_callable(mod)

        out = fn()
        try:
            correto = bool(
                np.allclose(
                    np.asarray(out, dtype=np.float64),
                    np.asarray(ref_vals, dtype=np.float64),
                    equal_nan=True,
                )
            )
        except Exception:
            correto = bool(
                (np.asarray(out, dtype=object) == np.asarray(ref_vals, dtype=object)).all()
            )

        for _ in range(NUM_WARM):
            fn()
        ts = []
        for _ in range(NUM_TIMED):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(ts)
        sd = statistics.stdev(ts) if len(ts) > 1 else 0.0

        print("\n[medição] saída LLM-Only capturada:")
        print(f"  - @numba.njit: {njit_count}")
        print(f"  - correto vs original: {correto}")
        print(f"  - mediana: {med:.3f} ms | desvio: {sd:.3f} ms")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Captura e valida a variante LLM-Only (prompt de anotação)."
    )
    parser.add_argument(
        "app",
        nargs="?",
        default="img2palette",
        choices=sorted(MAIN_FILE),
        help="aplicação (default: img2palette)",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="mede a saída capturada com a sonda do app",
    )
    args = parser.parse_args()

    capture(args.app)
    if args.measure:
        out_dir = sorted(
            (ROOT / "benchmarks" / "experimento_output").glob(f"llm_only_{args.app}_*")
        )[-1]
        generated = (out_dir / "output_llm_only.py").read_text(encoding="utf-8")
        measure(args.app, generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
