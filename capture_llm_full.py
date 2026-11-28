#!/usr/bin/env python3
"""Captura e testa a variante "LLM-Full" (modularização + anotação no mesmo prompt).

Braço experimental que combina as duas responsabilidades normalmente separadas no
forge:
  - o prompt de modularização (reescrever em forma compatível com ``nopython``);
  - o prompt de anotação (inserir ``@numba.njit``).

Entrada: código BRUTO do arquivo principal (nenhum AST). O objetivo é medir se o
LLM, sozinho, consegue reescrever E anotar, e quais riscos surgem sem a camada
sintática (``Inserter`` + ``NumbaDecoratorFilter`` + teste de equivalência).

Uso:
    PYTHONPATH=src .venv/bin/python benchmarks/capture_llm_full.py [app] [--measure]
    PYTHONPATH=src .venv/bin/python benchmarks/capture_llm_full.py img2palette --measure
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
    count_njit,
    import_from_dir,
)

GCR = ROOT / "benchmarks" / "gcr"
NUM_WARM = 3
NUM_TIMED = 10

COMBINED_PROMPT = """You are a code optimization tool. You receive a single Python file.

Do BOTH tasks in a single pass:

1. REFACTOR each function that has a CPU-bound numerical core into a Numba-compatible form:
   - Replace np.where, np.select and np.piecewise with explicit loops.
   - Replace boolean masking on arrays (arr[mask] = value, arr[~mask] /= x) with explicit loops.
   - Replace np.vectorize, list.append followed by np.array, and np.array(list) with pre-allocated numpy arrays and explicit loops.
   - Preserve array dimensionality and shapes exactly.
   - Do NOT use dict/set, try/except, print, logging, lambda, or string operations inside the numerical cores.
   - Preserve the EXACT names and signatures of all functions. If you extract the numerical core into a helper, keep the original function as a plain non-compiled wrapper that calls the helper.

2. ANNOTATE every function you made JIT-compatible with @numba.njit:
   - Add "import numba" at the top if missing.
   - Do NOT annotate functions that use external libraries (sklearn, PIL, requests, plotly, etc.) or that remain I/O/string handling.

Return ONLY the complete Python code. No markdown fences, no explanations.
"""


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
            tofile="llm_full",
            lineterm="",
        )
    )
    added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
    return diff, added, removed


def capture(app: str) -> tuple[Path, str]:
    main = MAIN_FILE[app]
    original_path = GCR / app / "original" / main
    original_code = original_path.read_text()

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
    if not api_key:
        raise SystemExit("[capture] GEMINI_API_KEY não definido")

    client = genai.Client(api_key=api_key)
    contents = f"{COMBINED_PROMPT}\n\n```\n{original_code}\n```"

    print(f"[capture] app={app} arquivo={main} modelo={model_name} (LLM-Full)")
    start = time.perf_counter()
    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=65536,
            thinking_config=types.ThinkingConfig(thinking_budget=500),
        ),
    )
    elapsed = time.perf_counter() - start

    raw = resp.text or ""
    generated = strip_fences(raw)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "benchmarks" / "experimento_output" / f"llm_full_{app}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "prompt.txt").write_text(COMBINED_PROMPT, encoding="utf-8")
    (out_dir / "input_original.py").write_text(original_code, encoding="utf-8")
    (out_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
    (out_dir / "output_llm_full.py").write_text(generated, encoding="utf-8")

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
    return out_dir, generated


def measure(app: str, generated: str) -> None:
    main = MAIN_FILE[app]
    probe_file, make_ref, make_callable, _fallback = PROBES[app](None, "llm_full")

    tmp = Path(tempfile.mkdtemp(prefix=f"vl_{app}_llmfull_"))
    try:
        (tmp / Path(main).name).write_text(generated)

        ref_vals = make_ref()

        try:
            mod = import_from_dir(tmp, Path(main).stem)
        except Exception as exc:
            print(f"\n[medição] FALHA AO IMPORTAR: {type(exc).__name__}: {str(exc)[:200]}")
            return

        fn = make_callable(mod)
        try:
            out = fn()
        except Exception as exc:
            print(f"\n[medição] FALHA AO RODAR: {type(exc).__name__}: {str(exc)[:200]}")
            return

        njit_count, compiled = count_njit(mod)
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

        print("\n[medição] saída LLM-Full:")
        print(f"  - @numba.njit (decorados): {njit_count}")
        print(f"  - compilados (signatures): {len(compiled)}")
        print(f"  - correto vs original: {correto}")
        print(f"  - mediana: {med:.3f} ms | desvio: {sd:.3f} ms")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Captura e testa a variante LLM-Full (modularização + anotação)."
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
            (ROOT / "benchmarks" / "experimento_output").glob(f"llm_full_{args.app}_*")
        )[-1]
        generated = (out_dir / "output_llm_full.py").read_text(encoding="utf-8")
        measure(args.app, generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
