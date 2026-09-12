# Documentacao Tecnica: forge

**Versao:** 2.0  
**Data:** 2026-08-21  
**Autor:** forge team

---

## Sumario

1. [Visao Geral](#1-visao-geral)
2. [Arquitetura](#2-arquitetura)
3. [Pipeline Build-Time](#3-pipeline-build-time)
4. [Middleware Runtime](#4-middleware-runtime)
5. [Servicos](#5-servicos)
6. [Utilitarios](#6-utilitarios)
7. [Como Usar](#7-como-usar)
8. [Exemplos Praticos](#8-exemplos-praticos)
9. [Limitacoes e Consideracoes](#9-limitacoes-e-consideracoes)

---

## 1. Visao Geral

O **forge** e uma ferramenta de otimizacao automatica de codigo Python que combina:

- **LLM (Gemini)** para refatoracao inteligente de codigo
- **Transformers AST** para pre e pos-processamento automatico
- **Numba @njit** para compilacao JIT (Just-In-Time) de funcoes numericas
- **Testes de equivalencia** para garantir correcao semantica

### Objetivo

Transformar codigo Python puro em codigo otimizado com Numba, automaticamente, sem intervencao manual.

### Casos de Uso

- **Aplicacoes serverless** (Cloud Run, AWS Lambda) com hotspots numericos
- **Processamento de dados** com loops pesados e operacoes element-wise
- **Modelos cientificos** com computacao intensiva (ODEs, inferencia Bayesiana)
- **Processamento de imagens** com conversao de cores e operacoes matriciais

### Resultados Tipicos

- **Speedup:** 1.0x a 9.67x (dependendo do padrao de codigo)
- **Taxa de sucesso:** 84% dos testes passando em 6 aplicacoes reais
- **Custo:** ~$0.14 por aplicacao (via Gemini API)
- **Automacao:** 100% automatico para codigo bem-comportado

---

## 2. Arquitetura

O forge opera em duas fases distintas:

### 2.1 Build-Time (CI/CD)

Executado durante o build da aplicacao (ex: Cloud Build, GitHub Actions):

```
+-------------------------------------------------------------+
|                    FORGE BUILD PIPELINE                       |
+-------------------------------------------------------------+
|                                                              |
|  1. DISCOVER                                                 |
|     +-> Encontra arquivos Python                            |
|                                                              |
|  2. PREPROCESS (AST Transformers)                           |
|     |-> ClassExtractor                                      |
|     |-> IOStripper                                          |
|     |-> BooleanMaskRewriter                                 |
|     |-> VectorizeRewriter                                   |
|     |-> BuiltinRewriter                                     |
|     +-> UniqueRewriter                                      |
|                                                              |
|  3. LLM (Gemini)                                            |
|     +-> Modulariza e refatora codigo                        |
|                                                              |
|  4. COMPATIBILITY CHECK                                     |
|     +-> Valida padroes Numba-compativeis                    |
|                                                              |
|  5. POST-PROCESS (AST Transformers)                         |
|     +-> Aplica transformers novamente                       |
|                                                              |
|  6. ANNOTATE (Inserter)                                     |
|     |-> Adiciona @numba.njit                                |
|     |-> Detecta cadeias transitivas                         |
|     +-> NumbaDecoratorFilter                                |
|                                                              |
|  7. TEST                                                    |
|     +-> Testes de equivalencia                              |
|                                                              |
|  8. RETRY (se necessario, ate 3x)                           |
|                                                              |
+-------------------------------------------------------------+
```

**Output:** Arquivos Python otimizados prontos para deploy.

### 2.2 Runtime (Cloud Run Container)

Middleware leve que gerencia funcoes JIT em producao:

```python
from sentinel import Middleware

mw = Middleware(modules=[calculator])
mw.warmup()  # Compila JIT no startup

result = mw.call("add", 1, 2)  # Executa com telemetria
```

**Responsabilidades:**
- **Warm-up:** Compila JIT antes do primeiro request
- **Telemetria:** Mede CPU, RAM, tempo de execucao
- **Fallback:** Re-executa com codigo original se JIT falhar

---

## 3. Pipeline Build-Time

A pipeline e orquestrada por `src/build_pipeline.py` e executa 8 etapas sequenciais.

### 3.1 Etapa 1: Discover (Descoberta)

**Servico:** `PatcherService.discover()`  
**Arquivo:** `src/services/patcher.py:9-31`

**O que faz:**
- Percorre recursivamente o diretorio fonte
- Encontra todos os arquivos `.py`
- Ignora diretorios como `.git`, `__pycache__`, `.venv`, `node_modules`
- Retorna caminhos relativos ordenados

**Exemplo:**
```python
patcher = PatcherService()
paths = patcher.discover("my_project/", extensions=[".py"])
# Retorna: ["calculator.py", "utils/math.py", "__init__.py"]
```

**Implementacao:**
```python
def discover(self, root: str, extensions=None, ignore_dirs=None):
    ignore_dirs = set(ignore_dirs or [".git", "__pycache__", ".venv", "node_modules"])
    relative_paths = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        
        for filename in filenames:
            if extensions and not any(filename.endswith(ext) for ext in extensions):
                continue
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root)
            relative_paths.append(rel_path)
    
    return sorted(relative_paths)
```

### 3.2 Etapa 2: Preprocess (Pre-processamento AST)

**Servico:** `PreprocessorService.transform_documents()`  
**Arquivo:** `src/services/preprocessor.py:48-74`

**O que faz:**
Aplica 6 transformers AST **antes** de enviar codigo ao LLM:

1. **ClassExtractor** - Extrai metodos de classe para funcoes module-level
2. **IOStripper** - Remove I/O incompativel (print, logging, try/except)
3. **BooleanMaskRewriter** - Converte `array[mask] = value` para loops
4. **VectorizeRewriter** - Converte `np.vectorize` para list comprehension
5. **BuiltinRewriter** - Converte `hex()[2:]` para `_int_to_hex()`
6. **UniqueRewriter** - Converte `np.unique(return_counts=True)` para funcao auxiliar

**Por que pre-processar?**
- Codigo real-world usa classes, I/O, boolean masking 2D
- Numba nao suporta esses padroes
- Pre-processamento automatico evita falhas de compilacao

**Exemplo:**
```python
# Antes (incompativel com Numba):
class Calculator:
    def add(self, a, b):
        print(f"Adding {a} + {b}")  # I/O
        result[a == 0] = b[a == 0]  # Boolean masking
        return result

# Depois (compativel):
def calculator_add(self, a, b):
    # print removido
    for _i in range(result.shape[0]):
        if a[_i] == 0:
            result[_i] = b[_i]
    return result
```

### 3.3 Etapa 3: LLM (Gemini API)

**Servico:** `ModelService.modularize()`  
**Arquivo:** `src/services/model.py:50-66`

**O que faz:**
- Envia codigo pre-processado ao Gemini 3.1 Pro
- Recebe codigo modularizado e refatorado
- Retorna JSON com `[{path, body}]`

**Prompt:**
O `MODULARIZE_PROMPT` (definido em `.env`) instrui o LLM a:
- Modularizar funcoes para compilacao JIT
- Preservar semantica original
- Usar padroes Numba-compativeis (pre-alocar arrays, evitar `np.array(list)`)
- Preservar dimensionalidade de arrays (1D, 2D, 3D)

**Exemplo:**
```python
model = ModelService()
result = model.modularize(payload)
# Retorna: {
#   "time": "22.1 ms",
#   "text": "[{\"path\": \"calc.py\", \"body\": \"...\"}]",
#   "tokens": {"prompt": 5000, "output": 3000, "total": 8000}
# }
```

**Retry automatico:**
```python
def _handle_model(self, prompt: str):
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            return self.client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=500),
                ),
            )
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = 2**attempt
            time.sleep(wait)
```

### 3.4 Etapa 4: Compatibility Check (Validacao)

**Servico:** `NumbaCompatibilityChecker.check()`  
**Arquivo:** `src/services/compatibility.py:54-71`

**O que faz:**
Valida codigo gerado pelo LLM **antes** da anotacao AST:

- Detecta padroes incompativeis: `np.where`, `np.ravel_multi_index`, dicts, sets
- Detecta funcoes scipy nao suportadas: `gammaln`, `gamma`, `betainc`
- Detecta boolean masking residual
- Detecta bibliotecas externas: sklearn, PIL, plotly, requests

**Por que validar?**
- LLM pode gerar codigo Python idiomatico que falha na compilacao
- Deteccao precoce evita gastar tokens em retries desnecessarios
- Feedback especifico melhora qualidade do codigo

**Exemplo:**
```python
checker = NumbaCompatibilityChecker()
is_compatible, issues = checker.check(code)

if not is_compatible:
    for severity, line, message in issues:
        print(f"[{severity.upper()}] Line {line}: {message}")
    # [ERROR] Line 42: np.where() is not supported in Numba nopython mode
```

**Padroes detectados:**
```python
_UNSUPPORTED_NUMPY_FUNCS = {
    "ravel_multi_index", "unravel_index", "clip",
    "where", "select", "piecewise", "vectorize",
}

_UNSUPPORTED_BUILTINS = {
    "dict", "set", "frozenset",
    "print", "input", "open", "exec", "eval",
    "hex", "bin", "oct", "chr", "ord",
}

_UNSUPPORTED_SCIPY_FUNCS = {
    "gammaln", "gamma", "betainc", "betaincinv",
    "erf", "erfc", "erfinv", "erfcinv",
}

_UNSUPPORTED_EXTERNAL_LIBS = {
    "sklearn", "KMeans", "AgglomerativeClustering",
    "pairwise_distances", "PIL", "Image",
    "plotly", "requests", "cv2",
}
```

### 3.5 Etapa 5: Post-Process (Pos-processamento)

**Funcao:** `_postprocess_documents()`  
**Arquivo:** `src/build_pipeline.py:99-184`

**O que faz:**
Aplica transformers AST **depois** do LLM:

1. **VectorizeRewriter** - Converte `np.vectorize` que o LLM pode ter gerado
2. **BuiltinRewriter** - Converte `hex()`, `bin()`, `oct()`
3. **UniqueRewriter** - Converte `np.unique(return_counts=True)`

**Adiciona funcoes auxiliares:**
- `_int_to_hex()`, `_int_to_bin()`, `_int_to_oct()` (se necessario)
- `_np_unique_with_counts()` (se necessario)
- `import numba` (se nao presente)

**Exemplo:**
```python
# LLM gerou (incompativel):
result = np.vectorize(hex)(array)

# Post-process converte para:
result = [_int_to_hex(x) for x in array]
# E adiciona funcao auxiliar _int_to_hex() no topo do arquivo
```

### 3.6 Etapa 6: Annotate (Anotacao AST)

**Servico:** `AnnotatorService.transform()`  
**Arquivo:** `src/services/annotator.py:11-21`

**O que faz:**
1. **Inserter** - Adiciona `@numba.njit` a funcoes elegiveis
2. **NumbaDecoratorFilter** - Remove `@numba.njit` de funcoes incompativeis

**Inserter (`src/utils/modifier.py:121-517`):**

O `Inserter` e o coracao da anotacao AST. Faz:

1. **Coleta internal callees** - Detecta cadeias transitivas de chamadas
2. **Adiciona imports** - `import numba` e `from _numba_helpers import ...`
3. **Visita cada funcao** e decide se deve ser anotada
4. **Gera wrapper** - Funcao Python pura que sanitiza inputs e converte outputs

**Decisao de anotacao:**
```python
def visit_FunctionDef(self, node: ast.FunctionDef):
    # Pula funcoes helper
    if node.name in ("_sanitize_for_numba", "_deep_to_list"):
        return node
    
    # Remove decorators duplicados
    self._strip_duplicate_numba_decorators(node)
    
    # Pula se ja tem @numba.njit
    if self._has_numba_decorator(node):
        return node
    
    # Pula se tem operacoes incompativeis
    if self._has_unsupported_calls(node) or self._has_nested_list_in_annotations(node):
        return node
    
    # Remove str de annotations (converte para float)
    if self._has_str_in_annotations(node):
        self._strip_str_from_annotations(node)
    
    # Aplica transformers internos
    _GenExprToListComp().visit(node)  # Generator -> list comprehension
    _ZipToRangeLoop().visit(node)     # zip() -> range(len())
    _EmptyListPatcher().generic_visit(node)  # [] -> numba.typed.List()
    
    # Adiciona @numba.njit
    decorator = ast.Attribute(value=ast.Name(id="numba"), attr="njit")
    node.decorator_list.insert(0, decorator)
    
    # Se e internal callee, nao gera wrapper
    if node.name in self._internal_callees or node.name.startswith("_"):
        return node
    
    # Gera wrapper Python puro
    orig_name = node.name
    jit_name = f"_{orig_name}_jit"
    node.name = jit_name
    wrapper = self._make_wrapper(orig_name, jit_name, node.args)
    
    return [node, wrapper]
```

**Deteccao de cadeias transitivas:**
```python
def _collect_internal_callees(self, tree: ast.AST) -> set:
    """Detecta funcoes que chamam outras funcoes JIT (transitivamente)."""
    top_level = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    
    # Build call graph
    call_graph = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            callees = set()
            for child in ast.walk(node):
                if (isinstance(child, ast.Call) and 
                    isinstance(child.func, ast.Name) and 
                    child.func.id in top_level):
                    callees.add(child.func.id)
            call_graph[node.name] = callees
    
    # Fixed-point iteration para cadeias transitivas
    called = set()
    for caller, callees in call_graph.items():
        called.update(callees)
    
    changed = True
    while changed:
        changed = False
        for caller, callees in call_graph.items():
            if callees & called and caller not in called:
                is_called = any(caller in other_callees for other_callees in call_graph.values())
                if is_called:
                    called.add(caller)
                    changed = True
    
    return called
```

**Wrapper gerado:**
```python
# Funcao JIT (interna):
@numba.njit
def _add_jit(a, b):
    return a + b

# Wrapper Python puro (publica):
def add(a, b):
    return _deep_to_list(_add_jit(_sanitize_for_numba(a), _sanitize_for_numba(b)))
```

**Por que wrapper?**
- `_sanitize_for_numba()` converte `list` -> `numba.typed.List`, `str` -> `0.0`
- `_deep_to_list()` converte `numba.typed.List` -> `list` Python
- Wrapper permite chamadas de codigo Python puro sem conhecer Numba

**NumbaDecoratorFilter (`src/utils/numba_filter.py:11-103`):**

Remove `@numba.njit` de funcoes que usam bibliotecas externas:

```python
_EXTERNAL_LIBS = {
    "sklearn", "KMeans", "AgglomerativeClustering",
    "pairwise_distances", "PIL", "Image",
    "plotly", "requests", "cv2",
}

def visit_FunctionDef(self, node: ast.FunctionDef):
    if self._uses_external_libs(node):
        # Remove @numba.njit
        node.decorator_list = [
            dec for dec in node.decorator_list
            if not self._is_numba_decorator(dec)
        ]
    return node
```

### 3.7 Etapa 7: Test (Testes de Equivalencia)

**Servico:** `TesterService.generate()` e `TesterService.run()`  
**Arquivo:** `src/services/tester.py`

**O que faz:**
1. **Gera testes** via LLM (`TEST_PROMPT`)
2. **Executa testes** contra codigo original (source_dir)
3. **Executa testes** contra codigo otimizado (output_dir)
4. **Compara resultados** - ambos devem passar

**Exemplo:**
```python
tester = TesterService(model)
test_file = tester.generate(original_docs, output_dir)
# Gera: output_dir/test_equivalence.py

tests_passed = tester.run(test_file, source_dir, output_dir)
# Executa pytest contra ambos os diretorios
# Retorna True se ambos passarem
```

**Testes invariant-based:**
O `TEST_PROMPT` instrui o LLM a gerar testes baseados em invariantes (nao valores hardcoded):

```python
# Bom (invariant-based):
def test_add():
    a, b = np.random.rand(10), np.random.rand(10)
    expected = a + b
    result = add(a, b)
    assert np.allclose(result, expected)

# Ruim (hardcoded):
def test_add():
    result = add(np.array([1, 2]), np.array([3, 4]))
    assert np.allclose(result, np.array([4, 6]))  # Brittle
```

### 3.8 Etapa 8: Retry (Tentativas)

**Logica:** `src/build_pipeline.py:257-373`

Se testes falham ou compatibilidade e insuficiente:
1. Remove arquivos gerados
2. Re-executa pipeline do passo 3 (LLM)
3. Ate 3 tentativas (configuravel via `FORGE_MAX_ATTEMPTS`)

**Exemplo:**
```python
for attempt in range(1, max_attempts + 1):
    # Step 2-6: LLM -> Parse -> Check -> Post-process -> Annotate
    
    # Step 7: Test
    tests_passed = tester.run(test_file_path, source_dir, output_dir)
    
    if not tests_passed:
        if attempt < max_attempts:
            print(f"[forge] WARNING: Tests failed on attempt {attempt}. Retrying...")
            continue
        else:
            raise RuntimeError(f"Tests failed after {max_attempts} attempts.")
    break
```

---

## 4. Middleware Runtime

O middleware runtime (`src/sentinel.py`) gerencia funcoes JIT em producao.

### 4.1 Classe Middleware

**Arquivo:** `src/sentinel.py:46-194`

**Responsabilidades:**
- **Warm-up:** Compila JIT antes do primeiro request
- **Telemetria:** Mede CPU, RAM, tempo de execucao
- **Fallback:** Re-executa com codigo original se JIT falhar

**Construtor:**
```python
def __init__(self, modules: list, fallbacks: dict[str, Callable] | None = None):
    self._optimized: dict[str, Callable] = {}
    self._fallbacks: dict[str, Callable] = fallbacks or {}
    
    for mod in modules:
        self._register(mod)
```

**Registro de funcoes:**
```python
def _register(self, module) -> None:
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("_"):
            self._optimized[name] = obj
```

### 4.2 Warm-up

**Metodo:** `warmup(inputs=None)`  
**Arquivo:** `src/sentinel.py:107-137`

**O que faz:**
- Dispara compilacao JIT para todas as funcoes registradas
- Executado no startup do container (antes do primeiro request real)
- Evita que o primeiro request pague o custo de compilacao

**Parametros:**
- `inputs`: Dict opcional `{nome_funcao: (args,...)}` com inputs domain-accurate
- Funcoes sem input especifico recebem `np.zeros(1, dtype=np.float64)` como default

**Retorno:**
```python
{
    "add": {"status": "ok", "compile_ms": 312.4},
    "multiply": {"status": "ok", "compile_ms": 145.2},
    "divide": {"status": "failed", "error": "division by zero"}
}
```

### 4.3 Call (Execucao com Telemetria)

**Metodo:** `call(func_name, *args, **kwargs)`  
**Arquivo:** `src/sentinel.py:139-189`

**O que faz:**
- Executa funcao otimizada com telemetria completa
- Mede wall-clock time, CPU e RAM
- Se falhar e houver fallback, re-executa com callable original

**Retorno:**
```python
{
    "result": 3,
    "time_ms": 0.04,
    "cpu_avg": 0.1,
    "cpu_max": 0.3,
    "mem_mb_avg": 48.2,
    "mem_mb_max": 48.5,
    "fallback_used": False
}
```

### 4.4 ResourceMonitor

**Classe:** `ResourceMonitor`  
**Arquivo:** `src/sentinel.py:11-43`

**O que faz:**
- Monitora CPU e RAM em background thread
- Amostra a cada 10ms durante execucao da funcao
- Retorna medias e maximos


---

## 5. Servicos

Os servicos (`src/services/`) encapsulam a logica de negocio da pipeline.

### 5.1 ModelService

**Arquivo:** `src/services/model.py`

**Responsabilidade:** Interface com a API do Gemini.

**Metodos:**

#### `__init__()`
- Carrega `GEMINI_API_KEY`, `MODULARIZE_PROMPT`, `TEST_PROMPT` do `.env`
- Inicializa cliente `genai.Client`

#### `modularize(raw_code: str) -> Dict[str, Any]`
- Envia codigo ao Gemini com `MODULARIZE_PROMPT`
- Retorna dict com `time`, `text`, `tokens`
- Retry automatico (3 tentativas com backoff exponencial)

#### `generate_test(raw_code: str) -> Dict[str, Any]`
- Envia codigo ao Gemini com `TEST_PROMPT`
- Gera testes de equivalencia
- Retorna dict com `time`, `text`, `tokens`

#### `_handle_model(prompt: str)`
- Wrapper interno com retry automatico
- Backoff exponencial: 2s, 4s, 8s
- Modelo: `gemini-3.1-pro-preview`
- Config: `response_mime_type="application/json"`, `thinking_budget=500`

### 5.2 PatcherService

**Arquivo:** `src/services/patcher.py`

**Responsabilidade:** I/O de arquivos (descoberta, leitura, escrita).

**Metodos:**

#### `discover(root: str, extensions=None, ignore_dirs=None) -> List[str]`
- Percorre recursivamente diretorio
- Encontra arquivos com extensoes especificadas
- Ignora `.git`, `__pycache__`, `.venv`, `node_modules`
- Retorna caminhos relativos ordenados

#### `to_json(root: str, relative_paths: List[str]) -> str`
- Le arquivos e serializa para JSON
- Formato: `[{"path": "file.py", "body": "..."}]`

#### `to_files(output_dir: str, documents: List[Dict]) -> List[str]`
- Escreve documentos para arquivos
- Validacao de seguranca (nao escreve fora de output_dir)
- Retorna caminhos absolutos dos arquivos criados

#### `emit_helpers_module(output_dir: str) -> str`
- Escreve `_numba_helpers.py` com funcoes auxiliares
- Contem `_sanitize_for_numba()` e `_deep_to_list()`

### 5.3 PreprocessorService

**Arquivo:** `src/services/preprocessor.py`

**Responsabilidade:** Orquestra transformers AST pre-LLM.

**Metodos:**

#### `__init__(enable_class_extract=True, enable_io_strip=True, enable_bool_mask_rewrite=True)`
- Inicializa transformers habilitados
- Ordem: ClassExtractor -> IOStripper -> BooleanMaskRewriter

#### `transform(code: str) -> str`
- Aplica todos os transformers habilitados
- Loga modificacoes (contagem de nodes AST)
- Retorna codigo pre-processado

#### `transform_documents(documents: list[dict]) -> list[dict]`
- Aplica preprocessing em batch
- Pula arquivos `__init__.py`
- Captura SyntaxError e continua

### 5.4 AnnotatorService

**Arquivo:** `src/services/annotator.py`

**Responsabilidade:** Injeta `@numba.njit` via AST.

**Metodos:**

#### `__init__()`
- Inicializa `Inserter` (transformer AST principal)

#### `transform(modified_code: str) -> str`
- Parse codigo para AST
- Chama `inserter.importer(tree)` para coletar internal callees
- Visita AST com `inserter.visit(tree)` para anotar funcoes
- Aplica `NumbaDecoratorFilter` para remover decorators incompativeis
- Retorna codigo anotado

### 5.5 NumbaCompatibilityChecker

**Arquivo:** `src/services/compatibility.py`

**Responsabilidade:** Valida compatibilidade com Numba nopython mode.

**Metodos:**

#### `check(code: str) -> Tuple[bool, List[Tuple[str, int, str]]]`
- Parse codigo para AST
- Walk tree detectando padroes incompativeis
- Retorna `(is_compatible, issues)`
- Issues: lista de `(severity, line, message)`

#### `format_issues() -> str`
- Formata issues para display
- Exemplo: `[ERROR] Line 42: np.where() is not supported`

**Padroes detectados:**
- Funcoes NumPy nao suportadas: `np.where`, `np.ravel_multi_index`, `np.clip`, `np.vectorize`
- Funcoes scipy nao suportadas: `gammaln`, `gamma`, `betainc`
- Built-ins nao suportados: `dict`, `set`, `print`, `hex`, `bin`, `oct`
- Bibliotecas externas: `sklearn`, `PIL`, `plotly`, `requests`, `cv2`
- Boolean masking 2D
- Dict/set literals
- Try/except blocks
- Lambda expressions

### 5.6 TesterService

**Arquivo:** `src/services/tester.py`

**Responsabilidade:** Gera e executa testes de equivalencia.

**Metodos:**

#### `generate(original_docs: list, output_dir: str) -> Optional[str]`
- Envia codigo original ao LLM com `TEST_PROMPT`
- Parse resposta JSON
- Escreve arquivo de teste em `output_dir`
- Retorna caminho do arquivo ou None se falhar

#### `run(test_file_path: str, source_dir: str, output_dir: str) -> bool`
- Executa pytest contra codigo original (source_dir)
- Executa pytest contra codigo otimizado (output_dir)
- Compara resultados (ambos devem passar)
- Timeout configuravel via `FORGE_TEST_TIMEOUT` (default: 120s)
- Retorna True se ambos passarem


---

## 6. Utilitarios

Os utilitarios (`src/utils/`) implementam transformers AST e funcoes auxiliares.

### 6.1 ClassExtractor

**Arquivo:** `src/utils/preprocessor.py:20-50`

**O que faz:** Extrai metodos de classe para funcoes module-level.

**Transformacao:**
```python
# Antes:
class Calculator:
    def add(self, a, b):
        return a + b

# Depois:
def calculator_add(self, a, b):
    return a + b
```

**Implementacao:**
```python
class ClassExtractor(ast.NodeTransformer):
    def visit_ClassDef(self, node: ast.ClassDef) -> list:
        self.generic_visit(node)
        
        extracted_functions = []
        class_name = node.name.lower()
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "__init__":
                    continue
                
                new_name = f"{class_name}_{item.name}"
                item.name = new_name
                
                if isinstance(item, ast.FunctionDef):
                    extracted_functions.append(item)
        
        return extracted_functions if extracted_functions else []
```

**Por que:** Numba so anota funcoes module-level, nao metodos de classe.

### 6.2 IOStripper

**Arquivo:** `src/utils/preprocessor.py:53-85`

**O que faz:** Remove operacoes de I/O incompativeis com Numba nopython mode.

**Remove:**
- `print()`, `input()`, `open()`, `exec()`, `eval()`
- `logging.*()`, `logger.*()`, `log.*()`
- `try/except` blocks (mantem corpo do try)
- `raise` statements

**Transformacao:**
```python
# Antes:
def process(data):
    print("Processing...")
    try:
        result = compute(data)
        logging.info("Done")
        return result
    except Exception as e:
        raise ValueError(f"Failed: {e}")

# Depois:
def process(data):
    result = compute(data)
    return result
```

**Implementacao:**
```python
class IOStripper(ast.NodeTransformer):
    _IO_FUNCTIONS = frozenset({
        "print", "input", "open", "exec", "eval",
        "logging", "logger", "log",
    })
    
    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                if node.value.func.id in self._IO_FUNCTIONS:
                    return None
        return node
    
    def visit_Try(self, node: ast.Try) -> list:
        self.generic_visit(node)
        return node.body
    
    def visit_Raise(self, node: ast.Raise) -> None:
        return None
```

### 6.3 BooleanMaskRewriter

**Arquivo:** `src/utils/preprocessor.py:88-223`

**O que faz:** Converte boolean masking 2D para loops explicitos.

**Transformacao:**
```python
# Antes (incompativel com Numba):
result[bool_mask] = value
denom[denom == 0.0] = 1.0

# Depois (compativel):
for _i in range(result.shape[0]):
    for _j in range(result.shape[1]):
        if bool_mask[_i, _j]:
            result[_i, _j] = value

for _i in range(denom.shape[0]):
    for _j in range(denom.shape[1]):
        if denom[_i, _j] == 0.0:
            denom[_i, _j] = 1.0
```

**Por que:** Numba nao suporta boolean array indexing em arrays 2D.

**Implementacao (simplificada):**
```python
class BooleanMaskRewriter(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign):
        # Detecta: result[mask] = value
        if not (isinstance(node.targets[0], ast.Subscript) and
                self._is_bool_mask_slice(node.targets[0].slice)):
            return node
        
        # Gera loops aninhados
        i_loop = ast.For(
            target=ast.Name(id="_i"),
            iter=ast.Call(func=ast.Name(id="range"), args=[array.shape[0]]),
            body=[j_loop]
        )
        
        return ast.fix_missing_locations(i_loop)
```

### 6.4 VectorizeRewriter

**Arquivo:** `src/utils/vectorize_rewriter.py`

**O que faz:** Converte `np.vectorize(func)(array)` para list comprehension.

**Transformacao:**
```python
# Antes:
result = np.vectorize(hex)(array)

# Depois:
result = [hex(x) for x in array]
```

**Por que:** `np.vectorize` com funcoes internas complexas nao e suportado pelo Numba.

**Implementacao:**
```python
class VectorizeRewriter(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        # Detecta: np.vectorize(func)(array)
        if not isinstance(node.func, ast.Call):
            return node
        
        if node.func.func.attr != 'vectorize':
            return node
        
        # Gera list comprehension
        list_comp = ast.ListComp(
            elt=ast.Call(func=func_arg, args=[ast.Name(id='x')]),
            generators=[ast.comprehension(target=ast.Name(id='x'), iter=array_arg)]
        )
        
        return list_comp
```

### 6.5 BuiltinRewriter

**Arquivo:** `src/utils/builtin_rewriter.py`

**O que faz:** Converte built-ins nao suportados (`hex()`, `bin()`, `oct()`) para funcoes auxiliares.

**Transformacao:**
```python
# Antes:
result = hex(integer)[2:]

# Depois:
result = _int_to_hex(integer)  # Funcao auxiliar Numba-compativel
```

**Funcoes auxiliares geradas:**
```python
@numba.njit
def _int_to_hex(integer):
    hex_chars = "0123456789abcdef"
    if integer == 0:
        return "00"
    result = ""
    while integer > 0:
        result = hex_chars[integer % 16] + result
        integer = integer // 16
    if len(result) < 2:
        result = "0" + result
    return result
```

### 6.6 UniqueRewriter

**Arquivo:** `src/utils/unique_rewriter.py`

**O que faz:** Converte `np.unique(arr, return_counts=True)` para funcao auxiliar.

**Transformacao:**
```python
# Antes:
unique_vals, counts = np.unique(arr, return_counts=True)

# Depois:
unique_vals, counts = _np_unique_with_counts(arr)
```

**Funcao auxiliar gerada:**
```python
@numba.njit
def _np_unique_with_counts(arr):
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    unique_count = 1
    for i in range(1, n):
        if sorted_arr[i] != sorted_arr[i-1]:
            unique_count += 1
    
    unique_vals = np.empty(unique_count, dtype=arr.dtype)
    counts = np.empty(unique_count, dtype=np.int64)
    
    unique_vals[0] = sorted_arr[0]
    counts[0] = 1
    idx = 0
    
    for i in range(1, n):
        if sorted_arr[i] != sorted_arr[i-1]:
            idx += 1
            unique_vals[idx] = sorted_arr[i]
            counts[idx] = 1
        else:
            counts[idx] += 1
    
    return unique_vals, counts
```


### 6.7 Inserter

**Arquivo:** `src/utils/modifier.py:121-517`

**O que faz:** Transformer AST principal que injeta `@numba.njit` e gera wrappers.

**Transformers internos:**

#### `_EmptyListPatcher`
Converte `x = []` para `x = numba.typed.List()`.

#### `_GenExprToListComp`
Converte generator expressions para list comprehensions.

```python
# Antes (nao suportado):
result = sum(x for x in array)

# Depois (suportado):
result = sum([x for x in array])
```

#### `_ZipToRangeLoop`
Converte `for a, b in zip(x, y):` para loop indexado.

```python
# Antes (nao suportado):
for a, b in zip(x, y):
    process(a, b)

# Depois (suportado):
for _zip_i in range(len(x)):
    a, b = x[_zip_i], y[_zip_i]
    process(a, b)
```

**Metodos principais:**

#### `_collect_internal_callees(tree)`
Detecta cadeias transitivas de chamadas entre funcoes module-level.

#### `_has_numba_decorator(node)`
Verifica se funcao ja tem `@numba.njit` ou `@numba.jit`.

#### `_strip_duplicate_numba_decorators(node)`
Remove decorators duplicados ou incompativeis:
- Remove `@numba.njit` se funcao tem `@numba.vectorize`
- Remove `@numba.njit` duplicados (mantem primeiro)

#### `_has_str_in_annotations(node)`
Verifica se annotations mencionam `str` (incompativel com Numba).

#### `_strip_str_from_annotations(node)`
Substitui `str` por `float` nas annotations.

#### `_has_unsupported_calls(node)`
Verifica se funcao usa `try/except`, `print`, `input`, etc.

#### `_make_wrapper(orig_name, jit_name, args)`
Gera wrapper Python puro que:
- Sanitiza inputs com `_sanitize_for_numba()`
- Chama funcao JIT
- Converte outputs com `_deep_to_list()`

#### `importer(tree)`
- Coleta internal callees
- Adiciona `import numba` (se nao presente)
- Adiciona `from _numba_helpers import _sanitize_for_numba, _deep_to_list`

### 6.8 NumbaDecoratorFilter

**Arquivo:** `src/utils/numba_filter.py`

**O que faz:** Remove `@numba.njit` de funcoes que usam bibliotecas externas.

**Bibliotecas detectadas:**
- sklearn: `KMeans`, `AgglomerativeClustering`, `pairwise_distances`
- PIL: `Image.fromarray`, `np.asarray(Image)`, `thumbnail`, `resize`
- plotly, requests, cv2

**Implementacao:**
```python
class NumbaDecoratorFilter(ast.NodeTransformer):
    _EXTERNAL_LIBS = frozenset({
        "sklearn", "KMeans", "AgglomerativeClustering",
        "pairwise_distances", "PIL", "Image",
        "plotly", "requests", "cv2",
    })
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self._uses_external_libs(node):
            node.decorator_list = [
                dec for dec in node.decorator_list
                if not self._is_numba_decorator(dec)
            ]
        return node
```

### 6.9 Funcoes Auxiliares (_numba_helpers.py)

**Arquivo:** `src/utils/deep.py`

**Conteudo:**

#### `_sanitize_for_numba(obj)`
Converte tipos Python para tipos Numba-compativeis:
- `list` -> `numba.typed.List`
- `str` em tuplas -> `0.0`

```python
def _sanitize_for_numba(obj):
    if isinstance(obj, list):
        typed_l = NumbaList()
        for item in obj:
            typed_l.append(_sanitize_for_numba(item))
        return typed_l
    elif isinstance(obj, tuple):
        return tuple(0.0 if isinstance(x, str) else _sanitize_for_numba(x) for x in obj)
    return obj
```

#### `_deep_to_list(obj)`
Converte `numba.typed.List` de volta para `list` Python:
- Preserva `np.ndarray` (nao converte)
- Preserva `str` e `bytes`
- Converte iteraveis recursivamente

```python
def _deep_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (str, bytes)):
        return obj
    try:
        return [_deep_to_list(item) for item in obj]
    except TypeError:
        return obj
```


---

## 7. Como Usar

### 7.1 Instalacao

**Runtime only** (Cloud Run container - dependencias minimas):
```bash
pip install git+https://github.com/davi-ga/forge
```

**Build + Runtime** (CI/CD pipeline - inclui Gemini SDK):
```bash
pip install "git+https://github.com/davi-ga/forge[build]"
```

### 7.2 Variaveis de Ambiente

| Variavel | Obrigatorio | Descricao |
|----------|:-----------:|-----------|
| `GEMINI_API_KEY` | Sim | API key do Gemini |
| `MODULARIZE_PROMPT` | Sim | Prompt para modularizacao |
| `TEST_PROMPT` | Sim | Prompt para geracao de testes |
| `SOURCE_DIR` | Nao | Override de `--source-dir` |
| `OUTPUT_DIR` | Nao | Override de `--output-dir` |
| `FORGE_MAX_ATTEMPTS` | Nao | Maximo de tentativas (default: 3) |
| `FORGE_TEST_TIMEOUT` | Nao | Timeout de testes em segundos (default: 120) |

### 7.3 CLI

```bash
forge --source-dir my_project/ --output-dir dist/
```

**Argumentos:**
- `--source-dir`: Diretorio com codigo fonte Python (default: `sample_project`)
- `--output-dir`: Diretorio de output para arquivos otimizados (default: `output_dir`)

### 7.4 Python API

```python
from build_pipeline import run

created_files = run(
    source_dir="my_project/",
    output_dir="dist/",
    max_attempts=3  # opcional
)

print(f"Arquivos criados: {created_files}")
```

### 7.5 Cloud Build (cloudbuild.yaml)

```yaml
steps:
  - name: "python:3.12-slim"
    entrypoint: pip
    args: ["install", "--no-cache-dir", "forge[build]@git+https://github.com/davi-ga/forge.git"]

  - name: "python:3.12-slim"
    entrypoint: forge
    args: ["--source-dir", "my_project/", "--output-dir", "dist/"]
    secretEnv: ["GEMINI_API_KEY", "MODULARIZE_PROMPT", "TEST_PROMPT"]

  - name: "gcr.io/cloud-builders/docker"
    args: ["build", "-t", "gcr.io/$PROJECT_ID/my-service", "."]
```

### 7.6 Dockerfile (Cloud Run)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Apenas artefatos otimizados entram na imagem
COPY dist/ .

EXPOSE 8080
CMD ["python", "main.py"]
```

### 7.7 Runtime Middleware

```python
import calculator  # modulo de dist/
from sentinel import Middleware

# Registra modulos e fallbacks opcionais
mw = Middleware(
    modules=[calculator],
    fallbacks={"divide": original_divide},  # opcional
)

# Warm-up no startup do container
warmup_report = mw.warmup()
# {"add": {"status": "ok", "compile_ms": 312.4}, ...}

# Execucao com telemetria
result = mw.call("add", 1, 2)
# {
#   "result": 3,
#   "time_ms": 0.04,
#   "cpu_avg": 0.1,
#   "cpu_max": 0.3,
#   "mem_mb_avg": 48.2,
#   "mem_mb_max": 48.5,
#   "fallback_used": False
# }
```


---

## 8. Exemplos Praticos

### 8.1 Exemplo 1: Funcao Numerica Simples

**Codigo original:**
```python
def compute_sum(n):
    total = 0
    for i in range(n):
        total += i * i
    return total
```

**Codigo otimizado (gerado pelo forge):**
```python
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _compute_sum_jit(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

def compute_sum(n):
    return _deep_to_list(_compute_sum_jit(_sanitize_for_numba(n)))
```

**Speedup esperado:** 5-10x para loops pesados.

### 8.2 Exemplo 2: Classe com Metodos

**Codigo original:**
```python
class Calculator:
    def add(self, a, b):
        print(f"Adding {a} + {b}")
        return a + b
    
    def multiply(self, a, b):
        return a * b
```

**Codigo otimizado:**
```python
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

# ClassExtractor extraiu metodos
@numba.njit
def _calculator_add_jit(self, a, b):
    # print removido pelo IOStripper
    return a + b

def calculator_add(self, a, b):
    return _deep_to_list(_calculator_add_jit(_sanitize_for_numba(self), _sanitize_for_numba(a), _sanitize_for_numba(b)))

@numba.njit
def _calculator_multiply_jit(self, a, b):
    return a * b

def calculator_multiply(self, a, b):
    return _deep_to_list(_calculator_multiply_jit(_sanitize_for_numba(self), _sanitize_for_numba(a), _sanitize_for_numba(b)))
```

### 8.3 Exemplo 3: Boolean Masking 2D

**Codigo original:**
```python
def apply_mask(data, mask, value):
    data[mask] = value
    return data
```

**Codigo otimizado:**
```python
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

# BooleanMaskRewriter converteu para loops
@numba.njit
def _apply_mask_jit(data, mask, value):
    for _i in range(data.shape[0]):
        for _j in range(data.shape[1]):
            if mask[_i, _j]:
                data[_i, _j] = value
    return data

def apply_mask(data, mask, value):
    return _deep_to_list(_apply_mask_jit(_sanitize_for_numba(data), _sanitize_for_numba(mask), _sanitize_for_numba(value)))
```

### 8.4 Exemplo 4: Conversao de Cores

**Codigo original:**
```python
def rgb2xyz(rgb):
    # Conversao RGB -> XYZ
    result = np.zeros_like(rgb)
    for i in range(len(rgb)):
        r, g, b = rgb[i]
        # ... calculos ...
        result[i] = [x, y, z]
    return result
```

**Codigo otimizado:**
```python
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _rgb2xyz_jit(rgb):
    result = np.zeros_like(rgb)
    for i in range(len(rgb)):
        r, g, b = rgb[i]
        # ... calculos ...
        result[i] = [x, y, z]
    return result

def rgb2xyz(rgb):
    return _deep_to_list(_rgb2xyz_jit(_sanitize_for_numba(rgb)))
```

**Speedup observado:** 1.07-5.37x (dependendo do tamanho do input).

### 8.5 Exemplo 5: Funcao com sklearn (Nao Otimizada)

**Codigo original:**
```python
from sklearn.cluster import KMeans

def cluster_data(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    return labels
```

**Codigo otimizado:**
```python
from sklearn.cluster import KMeans

# NumbaDecoratorFilter removeu @numba.njit
def cluster_data(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    return labels
```

**Por que nao otimizado:** sklearn nao e compativel com Numba nopython mode. Funcao mantida como wrapper Python puro.


---

## 9. Limitacoes e Consideracoes

### 9.1 Limitacoes do Numba nopython mode

**Nao suportado:**
- Boolean array indexing em arrays 2D: `array[bool_mask_2d]`
- Funcoes NumPy: `np.where`, `np.ravel_multi_index`, `np.clip`, `np.vectorize`
- Funcoes scipy: `gammaln`, `gamma`, `betainc`, `erf`
- Built-ins: `dict`, `set`, `print`, `input`, `open`, `hex`, `bin`, `oct`
- Bibliotecas externas: sklearn, PIL, plotly, requests, cv2
- Generator expressions (convertidos para list comprehensions)
- `zip()` sobre typed lists (convertido para loop indexado)
- Lambda expressions complexas
- Try/except blocks

**Suportado:**
- Loops `for` e `while`
- Arrays NumPy (`np.zeros`, `np.ones`, `np.empty`)
- Operacoes matematicas (`np.sqrt`, `np.sum`, `np.mean`)
- Indexacao de arrays (`arr[i]`, `arr[i, j]`)
- Funcoes matematicas (`np.sin`, `np.cos`, `np.exp`)
- Condicionais (`if/else`)
- Tuplas e listas homogeneas

### 9.2 Quando Usar forge

**Use forge se:**
- Codigo tem **hotspots numericos** (loops aninhados, operacoes element-wise)
- Funcoes sao **computacionalmente intensivas** (> 1ms de execucao)
- Codigo usa **NumPy** como base computacional
- Aplicacao e **CPU-bound** (nao I/O-bound)

**Nao use forge se:**
- Codigo e **I/O bound** (rede, disco, banco de dados)
- Funcoes ja usam **NumPy vetorizado** com BLAS/LAPACK
- Codigo depende de **bibliotecas incompativeis** (sklearn, PIL, plotly) sem wrappers
- Dados sao sempre **pequenos** (< 100 elementos)

### 9.3 Performance

**Onde Numba brilha:**
- Loops aninhados pesados: **9.67x** (yuki-mt)
- Conversao numerica element-wise: **5.37x** (img2palette)
- Chamadas repetidas em loop: **1.35x** (covid-model)
- Operacoes custom sobre arrays: **1.25x** (cube-builder)

**Onde Numba nao ajuda:**
- Algebra linear (BLAS): **0.81x** (NumPy ja usa C/Fortran otimizado)
- Operacoes ja vetorizadas: **~1.0x** (NumPy ja e rapido)
- Funcoes sub-millisecond: **1.01-1.16x** (overhead de medicao domina)
- I/O bound: **N/A** (nao e computacional)

### 9.4 Tamanho do Input

- **< 100 elementos:** JIT nao compensa (overhead de compilacao)
- **100-1000 elementos:** Sweet spot para JIT
- **> 5000 elementos:** NumPy ja e otimizado, ganhos modestos

### 9.5 Custo de API

| App | Tentativas | Tokens | Tempo LLM | Custo Est. |
|-----|:---:|:---:|:---:|:---:|
| yuki-mt | 1 | 8,427 | 22.1s | ~$0.04 |
| covid-model | 3 | 42,594 | 136.9s | ~$0.21 |
| cube-builder | 3 | 58,703 | 188.5s | ~$0.29 |
| img2palette | 3 | 26,799 | 60.9s | ~$0.13 |
| **Media** | **2.5** | **34k** | **102s** | **~$0.14** |

**Custo medio por funcao otimizada:** ~$0.007

### 9.6 Trade-offs

**Vantagens:**
- Automacao completa (100% automatico para codigo bem-comportado)
- Speedups reais (ate 9.67x em hotspots)
- Baixo custo (~$0.14 por aplicacao)
- Integracao CI/CD (Cloud Build, GitHub Actions)
- Middleware runtime com telemetria e fallback

**Desvantagens:**
- Limitacoes do Numba (nao suporta todas as bibliotecas Python)
- Overhead de compilacao JIT (primeira execucao e lenta)
- Dependencia de LLM (Gemini API)
- Pode gerar codigo sub-otimo para padroes complexos

### 9.7 Boas Praticas

1. **Identifique hotspots numericos** antes de rodar forge
2. **Comece com apps simples** para validar setup
3. **Monitore custos de API** - apps complexos podem custar $0.25+
4. **Valide outputs manualmente** antes de deploy
5. **Use warm-up no startup** para evitar latencia no primeiro request
6. **Registre fallbacks** para funcoes criticas
7. **Benchmark em multiplos tamanhos** para identificar sweet spots

### 9.8 Troubleshooting

**Problema:** `SyntaxError` durante preprocessing  
**Solucao:** Verifique se codigo fonte e Python valido

**Problema:** `NumbaCompatibilityError` durante compilacao  
**Solucao:** Verifique se funcao usa bibliotecas incompativeis; considere wrapper Python puro

**Problema:** Testes de equivalencia falham  
**Solucao:** Aumente `FORGE_MAX_ATTEMPTS` ou revise `MODULARIZE_PROMPT`

**Problema:** Speedup modesto (~1.0x)  
**Solucao:** Funcao pode ja ser otimizada pelo NumPy/BLAS; considere benchmark com inputs maiores

**Problema:** Overhead de compilacao JIT  
**Solucao:** Use `warmup()` no startup do container

---

## 10. Estrutura do Projeto

```
forge/
├── src/
│   ├── build_pipeline.py       # Pipeline build-time (CLI + Python API)
│   ├── sentinel.py             # Middleware runtime (warm-up, telemetria, fallback)
│   ├── services/
│   │   ├── model.py            # Cliente Gemini API
│   │   ├── annotator.py        # Servico de anotacao AST
│   │   ├── patcher.py          # Descoberta e I/O de arquivos
│   │   ├── preprocessor.py     # Orquestrador de transformers pre-LLM
│   │   ├── compatibility.py    # Validador de compatibilidade Numba
│   │   └── tester.py           # Gerador e executor de testes
│   └── utils/
│       ├── modifier.py         # Inserter (transformer AST principal)
│       ├── preprocessor.py     # Transformers: ClassExtractor, IOStripper, BooleanMaskRewriter
│       ├── vectorize_rewriter.py  # Transformer: VectorizeRewriter
│       ├── builtin_rewriter.py    # Transformer: BuiltinRewriter
│       ├── unique_rewriter.py     # Transformer: UniqueRewriter
│       ├── numba_filter.py        # NumbaDecoratorFilter
│       └── deep.py                # Funcoes auxiliares (_sanitize_for_numba, _deep_to_list)
├── benchmarks/                 # Relatorios de eficacia e benchmarks
├── .env                        # Variaveis de ambiente (API keys, prompts)
├── pyproject.toml              # Metadados do pacote e CLI
└── README.md                   # Documentacao rapida
```

---

## 11. Referencias

- **Numba Documentation:** https://numba.readthedocs.io/
- **Gemini API:** https://ai.google.dev/
- **Relatorios de Eficacia:** `benchmarks/RELATORIO_COMPLETO.md`
- **Benchmarks Detalhados:** `benchmarks/screening/` e `benchmarks/others/`

---

## 12. Changelog

### v2.0 (2026-08-19)
- Adicionado pre-processamento AST (ClassExtractor, IOStripper, BooleanMaskRewriter)
- Adicionado validador de compatibilidade Numba
- Adicionados transformers pos-LLM (VectorizeRewriter, BuiltinRewriter, UniqueRewriter)
- Melhorados prompts com exemplos Numba-compativeis
- Corrigido `_deep_to_list` para preservar numpy arrays
- Adicionada deteccao de cadeias transitivas no Inserter
- Adicionado NumbaDecoratorFilter para remover decorators incompativeis
- Taxa de sucesso: 26% -> 84%
- Speedup maximo: 1.26x -> 9.67x

### v1.0 (2026-08-18)
- Versao inicial
- Pipeline basica: Discover -> LLM -> Annotate -> Test
- Taxa de sucesso: 26%
- Speedup maximo: 1.26x

---

**Fim da Documentacao Tecnica**
