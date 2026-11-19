"""Ablação fbosler: Original vs LLM-Only vs AST-Only vs LLM+AST (forge).

Mesmo padrão de ``benchmarks/experimento_hibrido.py``, aplicado ao app
``fbosler`` (``functions/scoring.py``). Gera o braço AST-only e o forge
completo localmente, o LLM-only via chamada ao modelo, e mede a mediana
aquecida de lotes de ``n=200_000`` chamadas de ``scaled_sigmoid`` (a mesma
carga do ``web/app.py``).

Uso:
    python benchmarks/gcr/fbosler/ablate_fbosler.py
"""

import importlib
import importlib.util
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

from services.annotator import AnnotatorService  # noqa: E402
from services.patcher import PatcherService  # noqa: E402
from services.preprocessor import PreprocessorService  # noqa: E402
import build_pipeline  # noqa: E402

FUNCS = ROOT / "benchmarks" / "gcr" / "fbosler" / "functions"
SRC = FUNCS / "scoring.py"

_VALS = np.linspace(-3.0, 3.0, 1000)
N = 200_000

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


def batch(module, n=N):
    fn = module.scaled_sigmoid
    vals = _VALS
    total = 0.0
    t0 = time.perf_counter()
    for i in range(n):
        total += fn(vals[i % 1000])
    return (time.perf_counter() - t0) * 1000.0, total


def count_njit(module):
    n = 0
    compiled = []
    for name in dir(module):
        obj = getattr(module, name)
        if hasattr(obj, "nopython_signatures"):
            n += 1
            if obj.nopython_signatures:
                compiled.append(name)
    return n, compiled


def import_from_dir(directory, prefer):
    sys.path.insert(0, str(directory))
    mod_name = None
    try:
        for f in sorted(Path(directory).glob("*.py")):
            if prefer in f.stem:
                mod_name = f.stem
                break
        if mod_name is None:
            raise ImportError(f"no {prefer!r}")
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        return importlib.import_module(mod_name)
    finally:
        sys.path.pop(0)


def write_arm(tmp, docs):
    PatcherService().emit_helpers_module(str(tmp))
    for d in docs:
        name = Path(d["path"]).name
        if name == "__init__.py":
            continue
        (tmp / name).write_text(d["body"])


def load_reference():
    spec = importlib.util.spec_from_file_location("scoring_ref", SRC)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    return ref


def ast_only_docs():
    docs = [{"path": "scoring.py", "body": SRC.read_text()}]
    pp = PreprocessorService().transform_documents(docs)
    pp = build_pipeline._postprocess_documents(pp)
    ann = build_pipeline._annotate_documents(AnnotatorService(), pp)
    ann = build_pipeline._strip_unused_imports(ann)
    orig = {d["path"]: d["body"] for d in docs}
    return build_pipeline._restore_external_imports(ann, orig)


def llm_only_docs():
    from google import genai
    from google.genai import types

    code = SRC.read_text()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview"),
        contents=f"{PROMPT_NUMBA_STRONG}\n\n```\n{code}\n```",
        config=types.GenerateContentConfig(temperature=0.2),
    )
    out = resp.text.strip()
    if out.startswith("```"):
        out = out.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return [{"path": "scoring.py", "body": out}]


def full_forge_docs():
    out_dir = tempfile.mkdtemp(prefix="fbosler_forge_")
    build_pipeline.run(str(FUNCS), out_dir)
    docs = []
    for p in sorted(Path(out_dir).glob("*.py")):
        if p.name == "__init__.py" or p.name.startswith("test_"):
            continue
        docs.append({"path": p.name, "body": p.read_text()})
    return docs


def eval_arm(label, docs, ref_total, ref_med):
    res = {
        "arm": label, "importou": False, "njit_decor": 0, "njit_compilou": [],
        "rodou": False, "correto": False, "mediana_ms": None, "speedup": None,
        "first_call_ms": None, "erro": "",
    }
    tmp = Path(tempfile.mkdtemp(prefix=f"fb_{label}_"))
    try:
        write_arm(tmp, docs)
        mod = import_from_dir(tmp, "scoring")
        res["importou"] = True
        res["njit_decor"], _ = count_njit(mod)
        ms, total = batch(mod)
        res["first_call_ms"] = round(ms, 2)
        res["rodou"] = True
        res["njit_decor"], res["njit_compilou"] = count_njit(mod)
        res["correto"] = bool(np.isclose(total, ref_total, rtol=1e-9, atol=1e-6))
        med = statistics.median(batch(mod)[0] for _ in range(5))
        res["mediana_ms"] = round(med, 2)
        res["speedup"] = round(ref_med / med, 2) if med else 0
    except Exception as exc:  # noqa: BLE001
        res["erro"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def main():
    ref = load_reference()
    ref_total = batch(ref)[1]
    ref_med = statistics.median(batch(ref)[0] for _ in range(5))
    print(f"[reference] mediana={ref_med:.2f} ms  total={ref_total:.6f}", flush=True)

    rows = []
    for label, docs in (
        ("ast_only", ast_only_docs()),
        ("llm_plus_ast", full_forge_docs()),
    ):
        r = eval_arm(label, docs, ref_total, ref_med)
        rows.append(r)
        print(f"[{label}] importa={r['importou']} njit_decor={r['njit_decor']} "
              f"compilou={r['njit_compilou']} rodou={r['rodou']} correto={r['correto']} "
              f"t={r['mediana_ms']}ms first={r['first_call_ms']}ms speedup={r['speedup']}x "
              f"err={r['erro'][:140]}", flush=True)

    r = eval_arm("llm_only", llm_only_docs(), ref_total, ref_med)
    rows.append(r)
    print(f"[llm_only] importa={r['importou']} njit_decor={r['njit_decor']} "
          f"compilou={r['njit_compilou']} rodou={r['rodou']} correto={r['correto']} "
          f"t={r['mediana_ms']}ms first={r['first_call_ms']}ms speedup={r['speedup']}x "
          f"err={r['erro'][:140]}", flush=True)

    out = {
        "app": "fbosler",
        "ref_mediana_ms": round(ref_med, 2),
        "ref_total": ref_total,
        "n_calls": N,
        "arms": rows,
    }
    dest = FUNCS.parent / f"ablation_results_{time.strftime('%Y%m%d_%H%M')}.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nresultados em {dest}")


if __name__ == "__main__":
    main()
