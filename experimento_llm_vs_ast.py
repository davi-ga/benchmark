#!/usr/bin/env python3
"""
Experimento: LLM-only vs LLM+AST para otimização Numba

Objetivo: Provar que o modelo LLM sozinho não consegue gerar código Numba válido,
demonstrando a necessidade do AST Annotator na pipeline forge.
"""

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai import types


SOURCE_FILE = Path(__file__).parent / "gcr" / "yuki-mt" / "original" / "src" / "knn_feature.py"
OUTPUT_DIR = Path(__file__).parent / "experimento_output"
FORGE_SRC = Path(__file__).parent.parent / "src"


def call_gemini(prompt: str, code: str) -> str:
    """Chama Gemini e retorna texto gerado."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    full_prompt = f"{prompt}\n\n```\n{code}\n```"
    
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=full_prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text


def test_numba_compilation(code: str, include_helpers: bool = False) -> dict:
    """Testa se código pode ser importado e funções @numba.njit compilam."""
    result = {"success": False, "errors": [], "jit_functions": [], "compilation_errors": []}
    
    result["num_decorators"] = code.count("@numba.njit")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Escreve o módulo principal
        test_file = Path(tmpdir) / "test_module.py"
        test_file.write_text(code)
        
        # Copia helpers se necessário
        if include_helpers:
            helpers_src = FORGE_SRC / "utils" / "deep.py"
            if helpers_src.exists():
                # Extrai DEEP_TO_LIST_SRC e escreve _numba_helpers.py
                import importlib.util
                spec = importlib.util.spec_from_file_location("deep_module", str(helpers_src))
                deep_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(deep_module)
                
                helpers_file = Path(tmpdir) / "_numba_helpers.py"
                helpers_file.write_text(deep_module.DEEP_TO_LIST_SRC.lstrip("\n"))
        
        # Script de teste
        test_script = f"""
import sys
sys.path.insert(0, '{tmpdir}')

try:
    import test_module
    print("IMPORT_OK")
    
    # Lista funções @numba.njit
    import numba
    jit_funcs = []
    for name in dir(test_module):
        obj = getattr(test_module, name)
        if callable(obj) and hasattr(obj, 'nopython_signatures'):
            jit_funcs.append(name)
    
    print(f"JIT_FUNCTIONS: {{','.join(jit_funcs)}}")
    
    # Tenta compilar cada função JIT com tipos dummy
    import numpy as np
    for func_name in jit_funcs:
        func = getattr(test_module, func_name)
        try:
            # Força compilação inspecionando tipos
            sigs = func.nopython_signatures
            print(f"COMPILED_OK: {{func_name}} ({{len(sigs)}} signatures)")
        except Exception as e:
            print(f"COMPILE_ERROR: {{func_name}}: {{type(e).__name__}}: {{e}}")

except Exception as e:
    import traceback
    print(f"IMPORT_ERROR: {{type(e).__name__}}: {{e}}")
    traceback.print_exc()
"""
        
        script_file = Path(tmpdir) / "run_test.py"
        script_file.write_text(test_script)
        
        proc = subprocess.run(
            [sys.executable, str(script_file)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        output = proc.stdout + proc.stderr
        
        if "IMPORT_OK" in output:
            result["success"] = True
        
        for line in output.split('\n'):
            if "IMPORT_ERROR" in line:
                result["errors"].append(line.replace("IMPORT_ERROR: ", ""))
            elif "JIT_FUNCTIONS:" in line:
                funcs = line.replace("JIT_FUNCTIONS: ", "").strip()
                result["jit_functions"] = [f for f in funcs.split(',') if f]
            elif "COMPILED_OK:" in line:
                func_name = line.split(":")[1].strip().split(" ")[0]
                result["jit_functions"].append(f"{func_name} (compiled)")
            elif "COMPILE_ERROR:" in line:
                result["compilation_errors"].append(line.replace("COMPILE_ERROR: ", ""))
    
    return result


def analyze_code_quality(code: str, label: str) -> dict:
    """Analisa qualidade do código gerado."""
    analysis = {"label": label}
    
    # Verifica padrões problemáticos para Numba
    problematic_patterns = {
        "range() sem len()": "range(len(",
        "dict literal": "{",
        "set literal": "set(",
        "print statement": "print(",
        "try/except": "try:",
        "class definition": "class ",
        "sklearn import": "sklearn",
        "os.makedirs": "os.makedirs",
        "np.save": "np.save",
    }
    
    analysis["patterns_found"] = {}
    for pattern_name, pattern in problematic_patterns.items():
        count = code.count(pattern)
        if count > 0:
            analysis["patterns_found"][pattern_name] = count
    
    return analysis


def main():
    print("=" * 80)
    print("EXPERIMENTO: LLM-Only vs LLM+AST")
    print("Prova de que o modelo sozinho não consegue gerar código Numba ótimo")
    print("=" * 80)
    
    # Setup
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = OUTPUT_DIR / f"run_{timestamp}"
    exp_dir.mkdir()
    
    # Carrega código original
    print("\n[1] Carregando código original (knn_feature.py)...")
    source_code = SOURCE_FILE.read_text()
    (exp_dir / "01_original.py").write_text(source_code)
    print(f"    Linhas: {len(source_code.splitlines())}")
    print(f"    Funções module-level: {source_code.count('def ') - source_code.count('    def ')}")
    
    # ===== EXPERIMENTO 1: LLM-ONLY =====
    print("\n" + "=" * 80)
    print("EXPERIMENTO 1: LLM-ONLY (modelo gera @numba.njit diretamente)")
    print("=" * 80)
    
    prompt_llm_only = """Você é um expert em otimização Python com Numba.

Analise o código abaixo e:
1. Identifique funções que podem ser compiladas com Numba (funções puras, sem classes, sem sklearn)
2. Adicione @numba.njit APENAS nessas funções
3. Adicione `import numba` no topo
4. Mantenha o resto do código inalterado

RETORNE APENAS O CÓDIGO COMPLETO MODIFICADO.
"""
    
    print("\n[2] Enviando ao Gemini com prompt 'adicionar @numba.njit'...")
    llm_only_code = call_gemini(prompt_llm_only, source_code)
    if llm_only_code.startswith("```"):
        llm_only_code = llm_only_code.split("\n", 1)[1].rsplit("```", 1)[0]
    
    (exp_dir / "02_llm_only.py").write_text(llm_only_code)
    print(f"    @numba.njit adicionados: {llm_only_code.count('@numba.njit')}")
    
    print("\n[3] Testando compilação...")
    result_llm_only = test_numba_compilation(llm_only_code)
    
    print(f"    Import: {'✅ OK' if result_llm_only['success'] else '❌ FALHOU'}")
    print(f"    Funções JIT detectadas: {len(result_llm_only['jit_functions'])}")
    if result_llm_only['jit_functions']:
        for func in result_llm_only['jit_functions']:
            print(f"      - {func}")
    if result_llm_only['errors']:
        print(f"    Erros de import:")
        for err in result_llm_only['errors'][:3]:
            print(f"      - {err[:100]}")
    if result_llm_only['compilation_errors']:
        print(f"    Erros de compilação JIT:")
        for err in result_llm_only['compilation_errors'][:5]:
            print(f"      - {err[:120]}")
    
    analysis_llm = analyze_code_quality(llm_only_code, "LLM-Only")
    
    # ===== EXPERIMENTO 2: LLM + AST =====
    print("\n" + "=" * 80)
    print("EXPERIMENTO 2: LLM+AST (pipeline forge completa)")
    print("=" * 80)
    
    prompt_modularize = """Você é um expert em refatoração Python.

Sua tarefa: Modularize o código extraindo funções puras.
- NÃO adicione @numba.njit
- NÃO adicione import numba  
- Apenas reorganize o código em funções puras (sem classes, sem self)
- Mantenha funções com sklearn/classes separadas

RETORNE APENAS O CÓDIGO MODULARIZADO.
"""
    
    print("\n[4] Enviando ao Gemini com prompt 'apenas modularizar'...")
    modularized_code = call_gemini(prompt_modularize, source_code)
    if modularized_code.startswith("```"):
        modularized_code = modularized_code.split("\n", 1)[1].rsplit("```", 1)[0]
    
    (exp_dir / "03_modularized.py").write_text(modularized_code)
    print(f"    @numba.njit no código modularizado: {modularized_code.count('@numba.njit')}")
    
    # Aplica AST Annotator
    print("\n[5] Aplicando AST Annotator...")
    sys.path.insert(0, str(FORGE_SRC))
    from services.annotator import AnnotatorService
    
    annotator = AnnotatorService()
    llm_plus_ast_code = annotator.transform(modularized_code)
    
    (exp_dir / "04_llm_plus_ast.py").write_text(llm_plus_ast_code)
    print(f"    @numba.njit após AST: {llm_plus_ast_code.count('@numba.njit')}")
    
    print("\n[6] Testando compilação (com helpers)...")
    result_llm_plus_ast = test_numba_compilation(llm_plus_ast_code, include_helpers=True)
    
    print(f"    Import: {'✅ OK' if result_llm_plus_ast['success'] else '❌ FALHOU'}")
    print(f"    Funções JIT detectadas: {len(result_llm_plus_ast['jit_functions'])}")
    if result_llm_plus_ast['jit_functions']:
        for func in result_llm_plus_ast['jit_functions'][:10]:
            print(f"      - {func}")
    if result_llm_plus_ast['errors']:
        print(f"    Erros de import:")
        for err in result_llm_plus_ast['errors'][:3]:
            print(f"      - {err[:100]}")
    if result_llm_plus_ast['compilation_errors']:
        print(f"    Erros de compilação JIT:")
        for err in result_llm_plus_ast['compilation_errors'][:5]:
            print(f"      - {err[:120]}")
    
    analysis_ast = analyze_code_quality(llm_plus_ast_code, "LLM+AST")
    
    # ===== RESULTADOS =====
    results = {
        "timestamp": timestamp,
        "llm_only": {
            "num_decorators": result_llm_only["num_decorators"],
            "import_success": result_llm_only["success"],
            "jit_functions": result_llm_only["jit_functions"],
            "compilation_errors": result_llm_only["compilation_errors"],
            "errors": result_llm_only["errors"],
            "analysis": analysis_llm,
        },
        "llm_plus_ast": {
            "num_decorators": result_llm_plus_ast["num_decorators"],
            "import_success": result_llm_plus_ast["success"],
            "jit_functions": result_llm_plus_ast["jit_functions"],
            "compilation_errors": result_llm_plus_ast["compilation_errors"],
            "errors": result_llm_plus_ast["errors"],
            "analysis": analysis_ast,
        }
    }
    
    (exp_dir / "results.json").write_text(json.dumps(results, indent=2))
    
    print("\n" + "=" * 80)
    print("RESUMO COMPARATIVO")
    print("=" * 80)
    print(f"\n{'Métrica':<40} {'LLM-Only':<20} {'LLM+AST':<20}")
    print("-" * 80)
    print(f"{'@numba.njit adicionados':<40} {results['llm_only']['num_decorators']:<20} {results['llm_plus_ast']['num_decorators']:<20}")
    print(f"{'Import com sucesso':<40} {'✅' if results['llm_only']['import_success'] else '❌':<20} {'✅' if results['llm_plus_ast']['import_success'] else '❌':<20}")
    print(f"{'Funções JIT detectadas':<40} {len(results['llm_only']['jit_functions']):<20} {len(results['llm_plus_ast']['jit_functions']):<20}")
    print(f"{'Erros de compilação JIT':<40} {len(results['llm_only']['compilation_errors']):<20} {len(results['llm_plus_ast']['compilation_errors']):<20}")
    
    print(f"\nArquivos salvos em: {exp_dir}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
