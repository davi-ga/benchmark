#!/usr/bin/env python3
"""
Experimento controlado: Original vs LLM-Only vs AST-Only vs LLM+AST (forge)

Prova experimental da necessidade do hibrido LLM+AST.

Brasos (o prompt difere por PAPEL, nao por capricho — ver docs/COMPROVACAO_HIBRIDO.md §1.1):
  - reference        : codigo original (baseline de tempo/correcao)
  - llm_only_annotate: LLM como ANOTADOR — prompt que pede @numba.njit (ANNOTATE_PROMPT/
                       PROMPT_NUMBA_STRONG); fonte crua ou pre-processada (modo default:
                       EXPERIMENTO_LLM_ONLY_MODE=annotate). E a variante de performance.
  - llm_only_modular : LLM como MODULARIZADOR — MODULARIZE_PROMPT oficial sobre codigo
                       cru. Como esse prompt DELEGA a anotacao ao AST (Inserter), a saida
                       tem ZERO @numba.njit por construcao; NAO e variante de performance,
                       serve so para evidenciar a delegacao.
  - ast_only         : pre + post + anotacao AST, sem LLM (modulo gerado e guardado
                       em optimized_astonly_backup/)
  - llm_plus_ast     : pipeline completa (LLM modulariza com o prompt oficial + AST
                       anota); reusa o modulo ja gerado em optimized/

Metricas por braco: importa?, njit_count, compila/roda na chamada-sonda,
correcao numerica vs reference, tempo mediano aquecido e speedup.

Uso:
    python benchmarks/experimento_hibrido.py            # todos
    python benchmarks/experimento_hibrido.py cube-builder
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GCR = ROOT / "benchmarks" / "gcr"

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

REPS_LLM_ONLY = int(os.getenv("EXPERIMENTO_REPS", "2"))
N_WARM = 3
N_TIMED = 15

# ---------------------------------------------------------------- helpers ---


def import_from_dir(directory: Path, prefer: str):
    """Importa o .py cujo basename contem `prefer` e devolve o modulo."""
    sys.path.insert(0, str(directory))
    mod_name = None
    try:
        for f in sorted(directory.glob("*.py")):
            if prefer in f.stem:
                mod_name = f.stem
                break
        if mod_name is None:
            raise ImportError(f"no file matching {prefer!r}")
        import importlib

        if mod_name in sys.modules:
            del sys.modules[mod_name]
        return importlib.import_module(mod_name)
    finally:
        sys.path.pop(0)


def write_arm(tmp: Path, docs: list[dict]) -> None:
    helpers = GCR / "yuki-mt" / "optimized" / "optimized" / "_numba_helpers.py"
    shutil.copy(helpers, tmp / "_numba_helpers.py")
    for d in docs:
        name = Path(d["path"]).name
        if name in ("__init__.py",):
            continue
        (tmp / name).write_text(d["body"])


def count_njit(module) -> tuple[int, list]:
    n = 0
    compiled = []
    for name in dir(module):
        obj = getattr(module, name)
        if hasattr(obj, "nopython_signatures"):
            n += 1
            if obj.nopython_signatures:
                compiled.append(name)
    if hasattr(module, "NearestNeighborsFeats"):
        for _, m in vars(module.NearestNeighborsFeats).items():
            if hasattr(m, "nopython_signatures"):
                n += 1
    return n, compiled


def timed_median(fn, n_warm=N_WARM, n=N_TIMED):
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), min(ts), max(ts)


# ------------------------------------------------------------- probes/apps ---


def probe_yuki(dir_path: Path, arm: str):
    import importlib.util

    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(42)
    X = rng.random((1000, 10))
    y = rng.integers(0, 2, 1000)
    k_list = np.array([1, 3, 5, 7, 9], dtype=np.int64)
    eps = 1e-6
    Xt = rng.random((10, 10))

    def make_reference():
        spec = importlib.util.spec_from_file_location(
            "knn_ref", GCR / "yuki-mt" / "original" / "src" / "knn_feature.py")
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        m = ref.NearestNeighborsFeats(n_jobs=1, k_list=[1, 3, 5, 7, 9], metric="euclidean")
        m.fit(X, y)
        return np.asarray(m.predict(Xt))

    def make_callable(module):
        entry = getattr(module, "nearestneighborsfeats__get_features_for_one", None)
        if entry is not None:
            nn = NearestNeighbors(n_neighbors=9, metric="euclidean", n_jobs=1).fit(X)
            def run():
                dist, idx = nn.kneighbors(Xt)
                return np.asarray([
                    entry(Xt[i:i + 1], (dist[i:i + 1], idx[i:i + 1]), eps, k_list, 2, y)
                    for i in range(len(Xt))
                ])
            return run
        cls = getattr(module, "NearestNeighborsFeats", None)
        if cls is not None:
            m = cls(n_jobs=1, k_list=[1, 3, 5, 7, 9], metric="euclidean")
            m.fit(X, y)
            return lambda: np.asarray(m.predict(Xt))
        raise AttributeError("no yuki entry point (class or *_get_features_for_one)")

    return "src/knn_feature.py", make_reference, make_callable, None


def probe_img2(dir_path: Path, arm: str):
    import importlib.util

    from PIL import Image

    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    k = 5

    def img():
        return Image.fromarray(arr, mode="RGB")

    def make_reference():
        spec = importlib.util.spec_from_file_location(
            "img_ref", GCR / "img2palette" / "original" / "src" / "image.py")
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        return np.asarray(ref.get_palette(img(), k)[1], dtype=object)

    def make_callable(module):
        entry = getattr(module, "get_palette", None)
        if entry is None:
            raise AttributeError("no get_palette")
        return lambda: np.asarray(entry(img(), k)[1], dtype=object)

    return "src/image.py", make_reference, make_callable, None


def probe_cube(dir_path: Path, arm: str):
    rng = np.random.default_rng(3)
    band = rng.integers(0, 65536, 1000, dtype=np.uint16)
    bit = 3

    def make_reference():
        return band & (1 << bit)  # kernel original, conhecido e trivial

    def make_callable(module):
        entry = getattr(module, "extract_qa_bits", None)
        if entry is None:
            raise AttributeError("no extract_qa_bits")
        return lambda: np.asarray(entry(band.copy(), bit), dtype=np.uint16)

    def make_fallback():
        # baseline sem importar o modulo bruto (depende de shapely/rasterio):
        # o corpo original de extract_qa_bits e exatamente esta expressao
        return lambda: band & (1 << bit)

    return "src/processing.py", make_reference, make_callable, make_fallback


PROBES = {
    "yuki-mt": probe_yuki,
    "img2palette": probe_img2,
    "cube-builder": probe_cube,
}


# ------------------------------------------------------------- arm runners ---


def eval_arm(app: str, arm: str, docs: list[dict], reference_val) -> dict:
    probe_file, _mr, make_callable, _fb = PROBES[app](None, arm)
    res = {"arm": arm, "importou": False, "njit_count": 0, "rodou": False,
           "correto": False, "mediana_ms": None, "speedup": None, "erro": ""}
    tmp = Path(tempfile.mkdtemp(prefix=f"exp_{app}_{arm}_"))
    try:
        write_arm(tmp, docs)
        mod = import_from_dir(tmp, Path(probe_file).stem)
        res["importou"] = True
        res["njit_count"], _ = count_njit(mod)
        fn = make_callable(mod)
        t0 = time.perf_counter()
        out = fn()
        res["first_call_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        res["rodou"] = True
        try:
            res["correto"] = bool(np.allclose(
                np.asarray(out, dtype=np.float64),
                np.asarray(reference_val, dtype=np.float64),
                equal_nan=True))
        except Exception:
            res["correto"] = bool((np.asarray(out, dtype=object) ==
                                   np.asarray(reference_val, dtype=object)).all())
        med, _mn, _mx = timed_median(fn)
        res["mediana_ms"] = round(med, 3)
    except Exception as exc:  # noqa: BLE001
        res["erro"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def llm_modularize_only_arm(app: str, rep: int) -> tuple[list[dict], str]:
    """LLM como MODULARIZADOR (MODULARIZE_PROMPT oficial), sem AST.

    NAO e uma variante de performance: o MODULARIZE_PROMPT delega a anotacao
    @numba.njit ao AST (Inserter), de modo que a saida tem ZERO decorators por
    construcao. Existe apenas para evidenciar essa delegacao (brasos 'b' do doc
    em docs/COMPROVACAO_HIBRIDO.md §1.1), nao para medir speedup.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from services.model import ModelService

    import json as _json
    src_dir = GCR / app / "original"
    docs = []
    for p in sorted(src_dir.rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        docs.append({"path": str(p.relative_to(src_dir)),
                     "body": p.read_text()})
    model = ModelService()
    try:
        out = model.modularize(_json.dumps(docs, ensure_ascii=False, indent=2))
        raw = out["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return _json.loads(raw), ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)[:200]


ANNOTATE_PROMPT = """Você é um expert em otimização Python com Numba.

Analise o código abaixo e:
1. Identifique funções que podem ser compiladas com Numba (funções puras, sem classes, sem sklearn)
2. Adicione @numba.njit APENAS nessas funções
3. Adicione `import numba` no topo
4. Mantenha o resto do código inalterado

RETORNE APENAS O CÓDIGO COMPLETO MODIFICADO.
"""

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

# arquivo que contem o kernel principal por app (mesmo escopo do experimento 11/set)
MAIN_FILE = {
    "yuki-mt": "src/knn_feature.py",
    "img2palette": "src/image.py",
    "cube-builder": "src/processing.py",
}


def llm_only_annotate_arm(app: str, source: str = "raw") -> tuple[list[dict], str]:
    """LLM-Only como ANOTADOR: prompt pede @numba.njit.

    source="raw"          -> codigo bruto do original
    source="preprocessed" -> saida do PreprocessorService (limpa, sem classe/imports externos)
    """
    import os as _os
    from google import genai
    from google.genai import types as genai_types

    main = MAIN_FILE[app]
    if source == "preprocessed":
        from services.preprocessor import PreprocessorService

        src_dir = GCR / app / "original"
        docs_in = [
            {"path": str(p.relative_to(src_dir)), "body": p.read_text()}
            for p in sorted(src_dir.rglob("*.py")) if "__pycache__" not in str(p)
        ]
        docs_pp = PreprocessorService().transform_documents(docs_in)
        match = next((d for d in docs_pp if Path(d["path"]).as_posix() == main), None)
        if match is None:
            return None, f"preprocessed main file {main!r} not found"
        code = match["body"]
    else:
        code = (GCR / app / "original" / main).read_text()

    prompt = (PROMPT_NUMBA_STRONG
              if _os.getenv("EXPERIMENTO_ANNOTATE_PROMPT", "strong") == "strong"
              else ANNOTATE_PROMPT)
    try:
        client = genai.Client(api_key=_os.getenv("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=f"{prompt}\n\n```\n{code}\n```",
            config=genai_types.GenerateContentConfig(temperature=0.2),
        )
        out = resp.text.strip()
        if out.startswith("```"):
            out = out.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return [{"path": main, "body": out}], ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)[:200]



def load_saved_arm(app: str, sub: str) -> list[dict]:
    d = GCR / app / "optimized-fixed" / sub
    docs = []
    for p in sorted(d.glob("*.py")):
        if p.name == "_numba_helpers.py":
            continue
        docs.append({"path": p.name, "body": p.read_text()})
    return docs


# ------------------------------------------------------------------- main ---


def run_app(app: str):
    print(f"\n{'=' * 70}\n# {app}\n{'=' * 70}", flush=True)
    # stub opcional p/ import do original (plotly nao instalado; ramo conditional)
    for _n in ("plotly", "plotly.graph_objects"):
        sys.modules.setdefault(_n, types.ModuleType(_n))
    probe_file, make_ref, _mc, fallback = PROBES[app](None, "ref")
    # reference = codigo original roda e vira gabarito + baseline de tempo
    tmp = Path(tempfile.mkdtemp(prefix=f"exp_{app}_ref_"))
    write_arm(tmp, [
        {"path": Path(p).name, "body": p.read_text()}
        for p in (GCR / app / "original").rglob("*.py") if "__pycache__" not in str(p)
    ])
    _, mr, make_callable = PROBES[app](None, "ref")[:3]
    reference_val = mr()
    try:
        mod = import_from_dir(tmp, Path(probe_file).stem)
        ref_fn = make_callable(mod)
    except Exception:
        if fallback is None:
            raise
        ref_fn = fallback()
    ref_med, _, _ = timed_median(ref_fn, n_warm=1, n=5)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[reference] mediana={ref_med:.3f} ms", flush=True)

    rows = []

    def row_eval(label, docs):
        r = eval_arm(app, label, docs, reference_val)
        r["mediana_ms"] = r["mediana_ms"] or float("nan")
        r["speedup"] = (round(ref_med / r["mediana_ms"], 2)
                        if r["mediana_ms"] and r["mediana_ms"] == r["mediana_ms"] else 0)
        print(f"[{label}] importa={r['importou']} njit={r['njit_count']} "
              f"rodou={r['rodou']} correto={r['correto']} "
              f"t={r['mediana_ms']}ms speedup={r['speedup']}x err={r['erro'][:120]}",
              flush=True)
        rows.append(r)

    row_eval("ast_only", load_saved_arm(app, "optimized_astonly_backup"))
    row_eval("llm_plus_ast", load_saved_arm(app, "optimized"))
    llm_mode = os.getenv("EXPERIMENTO_LLM_ONLY_MODE", "annotate")
    if llm_mode == "annotate":
        sources = os.getenv("EXPERIMENTO_ANNOTATE_SOURCES", "raw,preprocessed").split(",")
        runs = [(f"llm_only_{s.strip()}", s.strip()) for s in sources if s.strip()]
    else:
        runs = [("llm_only_modularize", None)]
    print(f"[llm_only] modo: {llm_mode} | runs: {[r[0] for r in runs]}", flush=True)
    for base_label, src in runs:
        for i in range(REPS_LLM_ONLY):
            if src is None:
                docs, err = llm_modularize_only_arm(app, i)
            else:
                docs, err = llm_only_annotate_arm(app, source=src)
            label = f"{base_label}_{i + 1}"
            if docs is None:
                rows.append({"arm": label, "importou": False, "njit_count": 0,
                             "rodou": False, "correto": False, "mediana_ms": None,
                             "speedup": 0, "erro": f"LLM call: {err}"})
                print(f"[{label}] FALHA CHAMADA LLM: {err}", flush=True)
            else:
                row_eval(label, docs)
    return {"app": app, "ref_mediana_ms": round(ref_med, 3), "arms": rows}


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
    out = GCR / f"experimento_hibrido_results_{time.strftime('%Y%m%d_%H%M')}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nresultados em {out}")


if __name__ == "__main__":
    main()
