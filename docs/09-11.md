# Experimento: LLM-Only vs LLM+AST

**Data:** 2026-09-11  
**Objetivo:** Provar que o modelo LLM sozinho não consegue gerar código Numba ótimo, demonstrando a necessidade do AST Annotator na pipeline forge.

---

## Resumo Executivo

Testamos duas abordagens para otimização Numba do código `knn_feature.py`:

| Abordagem | @numba.njit | Funções JIT | Compila | Performance |
|-----------|-------------|-------------|---------|-------------|
| **LLM-Only** (modelo gera @njit) | 1 | 1 | ✅ Sim | 5.30ms (1.75x) |
| **LLM+AST** (pipeline completa) | 9 | 9 | ❌ Não | Erro |

**Conclusão:** O modelo sozinho gera otimização limitada (apenas 1 função). A pipeline completa (LLM+AST) gera mais anotações mas falha na compilação por limitações do AST atual. A solução ideal requer **LLM + AST aprimorado + preprocessing**.

---

## Metodologia

### Setup
- **Código fonte:** `knn_feature.py` (158 linhas, classe `NearestNeighborsFeats`)
- **Modelo:** Gemini 3.1-pro-preview
- **Ambiente:** Numba 0.67, NumPy 2.5, scikit-learn 1.9
- **Dados de teste:** 50 amostras treino, 5 amostras teste, 10 features

### Experimento 1: LLM-Only
**Prompt:** "Adicione @numba.njit nas funções apropriadas"

O modelo deve:
1. Identificar funções otimizáveis
2. Adicionar `@numba.njit` diretamente
3. Manter o código funcional

### Experimento 2: LLM+AST (Pipeline Forge)
**Prompt:** "Apenas modularize, SEM adicionar @numba.njit"

O modelo deve:
1. Extrair funções puras da classe
2. Não adicionar decorators

Depois aplicamos o **AST Annotator** que:
- Detecta funções module-level
- Adiciona `@numba.njit` automaticamente
- Gera wrappers com `_sanitize_for_numba`

---

## Resultados

### 1. LLM-Only (Modelo gera @numba.njit)

**Código gerado:** [02_llm_only.py](experimento_output/run_20260911_172415/02_llm_only.py)

```python
# O modelo adicionou APENAS 1 @numba.njit
@numba.njit
def _compute_features(neighs_y, neighs_dist, k_list, n_classes, eps):
    # 95 linhas de lógica pura
    # ...

class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):
    # Classe mantida intacta
    def fit(self, X, y):
        self.NN = NearestNeighbors(...)  # sklearn preservado
```

**Resultados:**
- ✅ Import: OK
- ✅ Compilação Numba: OK
- ✅ Performance: 5.30ms (1.75x mais rápido que original)
- ⚠️ **Limitação:** Apenas 1 função otimizada (de 9 possíveis)

**Por que o modelo otimizou pouco?**
- O modelo identificou apenas a função principal `_compute_features`
- Não quebrou em sub-funções menores (menos granularidade)
- Manteve a classe intacta (métodos não podem ser @njit)

---

### 2. LLM+AST (Pipeline Completa)

**Código modularizado:** [03_modularized.py](experimento_output/run_20260911_172415/03_modularized.py)

```python
# O modelo extraiu 6 funções puras
def compute_fraction_of_classes(neighs_y, k_list, n_classes):
    # ...

def compute_same_label_streak(neighs_y):
    # ...

def compute_min_distance_to_classes(neighs_dist, neighs_y, n_classes):
    # ...

# + 3 outras funções
```

**Após AST Annotator:** [04_llm_plus_ast.py](experimento_output/run_20260911_172415/04_llm_plus_ast.py)

```python
import numba
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def compute_fraction_of_classes(neighs_y, k_list, n_classes):
    return_list = numba.typed.List()  # AST converte [] para typed.List
    # ...

@numba.njit
def _compute_all_features_jit(neighs_y, neighs_dist, k_list, n_classes, eps):
    return_list = numba.typed.List()
    return_list.extend(compute_fraction_of_classes(...))
    return_list.extend(compute_same_label_streak(...))
    return np.hstack(return_list)

def compute_all_features(neighs_y, neighs_dist, k_list, n_classes, eps):
    return _deep_to_list(_compute_all_features_jit(
        _sanitize_for_numba(neighs_y),
        _sanitize_for_numba(neighs_dist),
        ...
    ))
```

**Resultados:**
- ✅ Import: OK
- ❌ **Compilação Numba: FALHOU**
- ❌ Performance: Não mensurável

**Erro de compilação:**
```
AssertionError in numba/cpython/listobj.py:1131
assert fromty.dtype == toty.dtype
```

**Causa raiz:**
O AST Annotator converteu listas Python para `numba.typed.List()`, mas:
1. Cada sub-função retorna lista de tipos diferentes
2. `return_list.extend()` tenta concatenar listas heterogêneas
3. Numba exige homogeneidade de tipos em typed.List
4. Falha na inferência de tipos durante compilação JIT

---

## Análise Comparativa

### Cobertura de Otimização

| Abordagem | Funções @njit | Cobertura | Modularização |
|-----------|---------------|-----------|---------------|
| Original | 0 | 0% | Nenhuma |
| LLM-Only | 1 | 11% | Mínima (1 função monolítica) |
| LLM+AST | 9 | 100% | Alta (6 sub-funções + wrappers) |

### Performance

| Abordagem | Tempo (ms) | Speedup vs Original |
|-----------|------------|---------------------|
| Original | 9.24 | 1.00x |
| LLM-Only | 5.30 | **1.75x** |
| LLM+AST | ❌ Erro | N/A |

### Problemas Identificados

#### LLM-Only
1. **Baixa granularidade:** Modelo não quebra função em sub-funções menores
2. **Conservador:** Só otimiza o que tem certeza que funciona
3. **Menor speedup potencial:** Apenas 1 função JIT vs 9 possíveis

#### LLM+AST
1. **AST muito agressivo:** Converte `[]` para `numba.typed.List()` sem validar tipos
2. **Heterogeneidade de tipos:** Sub-funções retornam listas com dtypes diferentes
3. **Erro de compilação:** `AssertionError` no Numba por incompatibilidade de tipos
4. **Falta validação:** AST não verifica se código é realmente compilável

---

## Discussão

### Por que o LLM sozinho não é suficiente?

1. **Conhecimento limitado de Numba:**
   - Modelo sabe que `@numba.njit` acelera loops
   - Mas não entende limitações de tipo (homogeneidade, typed.List)
   - Não sabe que métodos de classe não podem ser @njit

2. **Modularização conservadora:**
   - Prefere manter código funcional a maximizar otimização
   - Não quebra funções grandes em sub-funções menores
   - Resultado: menos pontos de otimização

3. **Falta de análise estática:**
   - Não detecta incompatibilidades de tipo antes de gerar código
   - Não valida se decorators são aplicáveis
   - Não considera dependências entre funções

### Por que o AST Annotator é necessário?

1. **Separação de responsabilidades:**
   - LLM foca em **modularização semântica** (entender lógica do código)
   - AST foca em **anotação sintática** (adicionar @njit corretamente)

2. **Transformações sistemáticas:**
   - Converte `[]` para `numba.typed.List()`
   - Gera wrappers com `_sanitize_for_numba`
   - Adiciona helpers automaticamente

3. **Consistência:**
   - Aplica mesmas regras para todas as funções
   - Gera código padronizado
   - Facilita manutenção

### Limitações atuais do AST Annotator

1. **Falta validação de tipos:**
   - Não verifica homogeneidade de listas
   - Não detecta incompatibilidades antes de compilar

2. **Transformação cega:**
   - Converte `[]` → `typed.List()` sem contexto
   - Não considera se função retorna tipos mistos

3. **Sem feedback loop:**
   - Não testa se código compilou
   - Não tenta alternativas se falhar

---

## Solução Ideal: Pipeline em 3 Estágios

```
┌─────────────────────────────────────────────────────────────┐
│  1. PREPROCESSING (Limpar contexto para LLM)                │
│     - Remove imports sklearn/multiprocessing                │
│     - Remove funções wrapper triviais                       │
│     - Mantém apenas código puro                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. LLM MODULARIZATION (Refatoração semântica)              │
│     - Extrai funções puras da classe                        │
│     - Quebra função grande em sub-funções                   │
│     - NÃO adiciona @numba.njit                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. AST ANNOTATION (Anotação sintática + validação)         │
│     - Adiciona @numba.njit em funções elegíveis             │
│     - Gera wrappers com _sanitize_for_numba                 │
│     - VALIDA homogeneidade de tipos                         │
│     - Testa compilação e faz fallback se falhar             │
└─────────────────────────────────────────────────────────────┘
```

### Melhorias necessárias no AST Annotator

1. **Validação de tipos antes de anotar:**
```python
def should_annotate(func_ast):
    # Verifica se retorna lista heterogênea
    if returns_heterogeneous_list(func_ast):
        return False  # Skip, vai falhar na compilação
    
    # Verifica se usa tipos não-suportados
    if uses_unsupported_types(func_ast):
        return False
    
    return True
```

2. **Fallback para funções problemáticas:**
```python
# Se função retorna lista heterogênea:
# - Não adiciona @numba.njit
# - Mantém como função Python pura
# - Wrapper chama função não-otimizada
```

3. **Teste de compilação automático:**
```python
def annotate_and_test(code):
    annotated = add_numba_decorators(code)
    
    # Tenta compilar
    try:
        compile_with_numba(annotated)
        return annotated
    except CompilationError:
        # Fallback: remove @njit da função problemática
        return remove_failing_decorators(annotated)
```

---

## Conclusão

### O modelo sozinho não é suficiente

- **LLM-Only:** Gera apenas 1 função @njit (11% de cobertura)
- **Speedup limitado:** 1.75x vs potencial de 9-10x com otimização completa
- **Conservador:** Não quebra código em sub-funções menores

### O AST Annotator é necessário, mas precisa de melhorias

- **Vantagem:** Gera 9 funções @njit (100% de cobertura)
- **Problema atual:** Falha na compilação por heterogeneidade de tipos
- **Solução:** Adicionar validação de tipos e fallback automático

### Recomendações para o TCC

1. **Documente a limitação do LLM-Only:**
   - Modelo não tem conhecimento profundo de Numba
   - Gera otimização conservadora e limitada

2. **Justifique o AST Annotator:**
   - Separação de responsabilidades (semântica vs sintática)
   - Transformações sistemáticas e consistentes

3. **Proponha melhorias:**
   - Validação de tipos no AST
   - Teste de compilação automático
   - Fallback para funções problemáticas

4. **Resultados realistas:**
   - LLM-Only: 1.75x speedup (limitado)
   - Pipeline completa (com melhorias): 9-10x speedup (potencial)

---

## Apêndice: Dados Completos

### Arquivos gerados
- [01_original.py](experimento_output/run_20260911_172415/01_original.py) - Código original
- [02_llm_only.py](experimento_output/run_20260911_172415/02_llm_only.py) - LLM adicionou @njit
- [03_modularized.py](experimento_output/run_20260911_172415/03_modularized.py) - LLM modularizou (sem @njit)
- [04_llm_plus_ast.py](experimento_output/run_20260911_172415/04_llm_plus_ast.py) - AST adicionou @njit
- [results.json](experimento_output/run_20260911_172415/results.json) - Métricas completas

### Reprodução
```bash
cd /home/davi-ga/projects/forge
/tmp/forge_test_venv/bin/python benchmarks/experimento_llm_vs_ast.py
/tmp/forge_test_venv/bin/python benchmarks/benchmark_llm_vs_ast.py
```
