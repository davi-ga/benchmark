#!/usr/bin/env python3
"""Validação local (pós-outubro): Original vs LLM-Only vs AST-Only vs LLM+AST.

Para cada aplicação, constrói os quatro braços e mede, por braço:
  - aquecimento descartado (NUM_WARM) e NUM_TIMED execuções medidas;
  - mediana e **desvio padrão** das execuções medidas;
  - número de kernels @numba.njit compilados, primeira chamada, correção
    (np.allclose contra o código original, que também serve de gabarito).

O braço LLM+AST executa o forge completo localmente (build_pipeline.run);
o braço LLM-Only usa o prompt forte de anotação sobre o código bruto; o
braço AST-Only roda pré + pós + anotação, sem chamar o modelo.

Uso:
    python benchmarks/validacao_local.py                 # todas as apps
    python benchmarks/validacao_local.py yuki-mt fbosler
"""

import importlib.util
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

import build_pipeline  # noqa: E402
from services.annotator import AnnotatorService  # noqa: E402
from services.patcher import PatcherService  # noqa: E402
from services.preprocessor import PreprocessorService  # noqa: E402
from experimento_hibrido import PROBES as _EH_PROBES  # noqa: E402
from experimento_hibrido import count_njit, import_from_dir  # noqa: E402

GCR = ROOT / "benchmarks" / "gcr"

NUM_WARM = 3
NUM_TIMED = 10

PROMPT_NUMBA_STRONG = """Você é um especialista em Numba. Sua tarefa é, para CADA
função do arquivo, decidir se ela pode ser compilada em modo nopython e, quando
puder, adicionar o decorador @numba.njit.

Regras de decisão — NÃO anote a função se qualquer item abaixo for verdadeiro:
- usa classes, `self`, ou é um método de classe;
- usa bibliotecas externas (sklearn, PIL, requests, os, multiprocessing, hashlib, datetime, plotly, etc.);
- usa dict/set (literais `{}` ou chamadas `dict()`/`set()`);
- usa strings, f-strings, `.format`, `print`, `logging`, `try/except`, `raise`, `open`;
- usa indexação booleana em array 2D (`arr[mask]` com `mask` 2D);
- usa `np.where`, `np.select`, `np.vectorize` ou `lambda`;
- constrói array com `np.array(lista)` dentro do corpo.

Anote SOMENTE funções com operações numéricas puras: loops, aritmética de
arrays, indexação inteira, funções suportadas de `np`/`numba`.

Instruções obrigatórias:
1. Preserve EXATAMENTE os nomes e assinaturas das funções.
2. Preserve TODAS as funções: não remova, não renomeie, não substitua o corpo por `pass`.
3. Adicione `import numba` no topo (se não existir). Não duplique imports.
4. Se nenhuma função for elegível, devolva o arquivo praticamente inalterado.
5. Retorne APENAS o código Python completo — sem explicações e sem cercas de código.
"""

MAIN_FILE = {
    "yuki-mt": "src/knn_feature.py",
    "img2palette": "src/image.py",
    "cube-builder": "src/processing.py",
    "fbosler": "functions/scoring.py",
}

ORIGINAL_DIR = {
    "yuki-mt": GCR / "yuki-mt" / "original",
    "img2palette": GCR / "img2palette" / "original",
    "cube-builder": GCR / "cube-builder" / "original",
    "fbosler": GCR / "fbosler" / "original",
}


# ------------------------------------------------------------------ probes ---


def _probe_fbosler(src_dir, arm):
    """Sonda do fbosler: lote de 200k chamadas de scaled_sigmoid (carga do web/app.py)."""
    vals = np.linspace(-3.0, 3.0, 1000)
    n = 200_000
    src = GCR / "fbosler" / "original" / "functions" / "scoring.py"

    def _total(module):
        fn = module.scaled_sigmoid
        total = 0.0
        for i in range(n):
            total += fn(vals[i % 1000])
        return total

    def make_reference():
        spec = importlib.util.spec_from_file_location("scoring_ref", src)
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        return _total(ref)

    def make_callable(module):
        return lambda: _total(module)

    return "functions/scoring.py", make_reference, make_callable, None


PROBES = dict(_EH_PROBES)
PROBES["fbosler"] = _probe_fbosler


# ------------------------------------------------------------- build arms ---


def original_docs(app):
    src = ORIGINAL_DIR[app]
    return [
        {"path": str(p.relative_to(src)), "body": p.read_text()}
        for p in sorted(src.rglob("*.py"))
        if "__pycache__" not in str(p)
    ]


def ast_only_docs(app):
    """Braço AST-Only: pré + pós + anotação, sem chamar o LLM."""
    docs = original_docs(app)
    pp = PreprocessorService().transform_documents(docs)
    pp = build_pipeline._postprocess_documents(pp)
    ann = build_pipeline._annotate_documents(AnnotatorService(), pp)
    ann = build_pipeline._strip_unused_imports(ann)
    orig = {d["path"]: d["body"] for d in docs}
    return build_pipeline._restore_external_imports(ann, orig)


def llm_plus_ast_docs(app):
    """Braço LLM+AST: forge completo rodando localmente (build_pipeline.run)."""
    out_dir = tempfile.mkdtemp(prefix=f"forge_{app}_")
    build_pipeline.run(str(ORIGINAL_DIR[app]), out_dir)
    docs = []
    for p in sorted(Path(out_dir).rglob("*.py")):
        rel = p.relative_to(out_dir)
        if rel.name == "__init__.py" or rel.name.startswith("test_"):
            continue
        if "_numba_helpers" in rel.name:
            continue
        docs.append({"path": str(rel), "body": p.read_text()})
    shutil.rmtree(out_dir, ignore_errors=True)
    return docs


def llm_only_docs(app):
    """Braço LLM-Only: prompt forte de anotação sobre o código bruto."""
    from google import genai
    from google.genai import types

    main = MAIN_FILE[app]
    code = (ORIGINAL_DIR[app] / main).read_text()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview"),
        contents=f"{PROMPT_NUMBA_STRONG}\n\n```\n{code}\n```",
        config=types.GenerateContentConfig(temperature=0.2),
    )
    out = (resp.text or "").strip()
    if out.startswith("```"):
        out = out.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return [{"path": main, "body": out}]


# --------------------------------------------------------------- measurement ---


def write_arm(tmp, docs):
    PatcherService().emit_helpers_module(str(tmp))
    for d in docs:
        name = Path(d["path"]).name
        if name == "__init__.py":
            continue
        (tmp / name).write_text(d["body"])


def measure(fn):
    for _ in range(NUM_WARM):
        fn()
    ts = []
    for _ in range(NUM_TIMED):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    med = statistics.median(ts)
    sd = statistics.stdev(ts) if len(ts) > 1 else 0.0
    return med, sd, ts


def eval_docs(app, label, docs, ref_vals):
    probe_file, _mr, make_callable, _fb = PROBES[app](None, label)
    res = {
        "arm": label, "importou": False, "njit_count": 0, "compilou": [],
        "rodou": False, "correto": False, "primeira_chamada_ms": None,
        "mediana_ms": None, "desvio_ms": None, "amostras_ms": [], "erro": "",
    }
    tmp = Path(tempfile.mkdtemp(prefix=f"vl_{app}_{label}_"))
    try:
        write_arm(tmp, docs)
        mod = import_from_dir(tmp, Path(probe_file).stem)
        res["importou"] = True
        res["njit_count"], res["compilou"] = count_njit(mod)
        fn = make_callable(mod)
        t0 = time.perf_counter()
        out = fn()
        res["primeira_chamada_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        res["rodou"] = True
        try:
            res["correto"] = bool(np.allclose(
                np.asarray(out, dtype=np.float64),
                np.asarray(ref_vals, dtype=np.float64), equal_nan=True))
        except Exception:
            res["correto"] = bool((np.asarray(out, dtype=object) ==
                                   np.asarray(ref_vals, dtype=object)).all())
        med, sd, ts = measure(fn)
        res["mediana_ms"] = round(med, 3)
        res["desvio_ms"] = round(sd, 3)
        res["amostras_ms"] = [round(t, 3) for t in ts]
    except Exception as exc:  # noqa: BLE001
        res["erro"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def reference(app):
    probe_file, make_ref, make_callable, fallback = PROBES[app](None, "ref")
    ref_vals = make_ref()
    tmp = Path(tempfile.mkdtemp(prefix=f"vl_{app}_ref_"))
    write_arm(tmp, original_docs(app))
    try:
        mod = import_from_dir(tmp, Path(probe_file).stem)
        ref_fn = make_callable(mod)
    except Exception:
        if fallback is None:
            raise
        ref_fn = fallback()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    med, sd, ts = measure(ref_fn)
    return ref_vals, med, sd, ts


def run_app(app):
    print(f"\n{'=' * 78}\n# {app}\n{'=' * 78}", flush=True)
    ref_vals, ref_med, ref_sd, ref_ts = reference(app)
    print(f"[original] mediana={ref_med:.3f} ms  desvio={ref_sd:.3f} ms", flush=True)

    rows = []

    def record(label, docs):
        if docs is None:
            rows.append({"arm": label, "importou": False, "erro": "arm não gerado"})
            return
        r = eval_docs(app, label, docs, ref_vals)
        r["speedup"] = round(ref_med / r["mediana_ms"], 2) if r.get("mediana_ms") else 0
        rows.append(r)
        print(
            f"[{label}] importa={r['importou']} njit={r['njit_count']} "
            f"rodou={r['rodou']} correto={r['correto']} "
            f"mediana={r['mediana_ms']} ms desvio={r['desvio_ms']} ms "
            f"speedup={r['speedup']}x err={r['erro'][:120]}",
            flush=True,
        )

    for label, builder in (
        ("ast_only", ast_only_docs),
        ("llm_plus_ast", llm_plus_ast_docs),
        ("llm_only", llm_only_docs),
    ):
        try:
            record(label, builder(app))
        except Exception as exc:  # noqa: BLE001
            rows.append({"arm": label, "importou": False,
                         "erro": f"{type(exc).__name__}: {str(exc)[:300]}"})
            print(f"[{label}] FALHA: {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    return {
        "app": app,
        "original": {"mediana_ms": round(ref_med, 3), "desvio_ms": round(ref_sd, 3),
                     "amostras_ms": [round(t, 3) for t in ref_ts]},
        "arms": rows,
    }


def main():
    apps = sys.argv[1:] or list(PROBES)
    results = []
    for app in apps:
        try:
            results.append(run_app(app))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append({"app": app, "error": str(exc)[:300]})

    out = GCR / f"validacao_local_{time.strftime('%Y%m%d_%H%M')}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nresultados em {out}")


if __name__ == "__main__":
    main()
