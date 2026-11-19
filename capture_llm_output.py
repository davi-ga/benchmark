#!/usr/bin/env python3
"""Captura a saída crua do modelo na etapa de modularização da pipeline forge.

Executa apenas o estágio do LLM (descoberta -> pré-processamento -> modularização)
sobre um app de referência, sem pós-processamento nem anotação, e grava a resposta
verbatim do modelo para uso como artefato de documentação.

Uso:
    PYTHONPATH=src .venv/bin/python benchmarks/capture_llm_output.py [app] [source_subdir]

Exemplo:
    PYTHONPATH=src .venv/bin/python benchmarks/capture_llm_output.py yuki-mt original/src
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT / "src"))

from services.model import ModelService  # noqa: E402
from services.patcher import PatcherService  # noqa: E402
from services.preprocessor import PreprocessorService  # noqa: E402


def strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def main() -> int:
    app = sys.argv[1] if len(sys.argv) > 1 else "yuki-mt"
    source_subdir = sys.argv[2] if len(sys.argv) > 2 else "original/src"
    source_dir = ROOT / "benchmarks" / "gcr" / app / source_subdir

    if not source_dir.is_dir():
        print(f"[capture] source dir not found: {source_dir}", file=sys.stderr)
        return 1

    print(f"[capture] app={app} source={source_dir}")

    patcher = PatcherService()
    paths = patcher.discover(str(source_dir), extensions=[".py"])
    print(f"[capture] discovered {len(paths)} file(s): {paths}")
    if not paths:
        return 1

    documents = json.loads(patcher.to_json(str(source_dir), paths))

    preprocessor = PreprocessorService()
    preprocessed = preprocessor.transform_documents(documents)
    payload = json.dumps(preprocessed, ensure_ascii=False, indent=2)
    print(f"[capture] preprocessed payload: {len(payload)} chars")

    model = ModelService()
    print(f"[capture] model={model.model_name} — calling modularize()...")
    result = model.modularize(payload)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "benchmarks" / "experimento_output" / f"raw_llm_{app}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "prompt_input.json").write_text(payload, encoding="utf-8")
    (out_dir / "raw_response.txt").write_text(result["text"], encoding="utf-8")

    parsed_bodies = 0
    try:
        parsed = json.loads(strip_fences(result["text"]))
        if isinstance(parsed, list):
            for doc in parsed:
                rel = doc.get("path", f"unnamed_{parsed_bodies}.py")
                dest = out_dir / "modularized" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(doc.get("body", ""), encoding="utf-8")
                parsed_bodies += 1
        else:
            print("[capture] response is not a JSON array", file=sys.stderr)
    except json.JSONDecodeError as exc:
        print(f"[capture] could not parse response as JSON: {exc}", file=sys.stderr)

    stats = {
        "app": app,
        "source_subdir": source_subdir,
        "model": model.model_name,
        "elapsed": result["time"],
        "tokens": result["tokens"],
        "n_decorators": result["text"].count("@numba.njit"),
        "files_written": parsed_bodies,
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[capture] elapsed={result['time']} tokens={result['tokens']}")
    print(f"[capture] @numba.njit in raw output: {stats['n_decorators']}")
    print(f"[capture] wrote {parsed_bodies} modularized file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
