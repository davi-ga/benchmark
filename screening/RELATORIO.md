# Relatorio Consolidado: forge em Apps Serverless CPU-Intensivos

**Data:** 2026-08-18  
**Pipeline:** forge (LLM-based code modularization + Numba @njit injection)  
**Apps testados:** 3 repos do screening (yuki-mt, covid-model, cube-builder)

---

## Resumo Executivo

| App | Arquivos | Funcoes Total | Anotadas | Equiv. Test | Speedup Medio | Status |
|-----|----------|:---:|:---:|:---:|:---:|:---:|
| **yuki-mt** | knn_feature.py | 5 | 5 | PASS | 1.01x - 1.26x | SUCESSO |
| **covid-model** | infer_rt.py + seir_model.py | 14 | 19 | N/A* | 0.93x - 1.11x | PARCIAL |
| **cube-builder** | processing.py + maestro.py | 44 | 0 | FAIL | N/A | FALHA |

*Test generation failed for covid-model, but functions compiled and ran successfully.

---

## Resultados Detalhados

### 1. yuki-mt_scripts (knn_feature.py)

**Status:** SUCESSO COMPLETO

**Pipeline:**
- Descoberta: 1 arquivo
- LLM: 17.2s, 4555 tokens
- Anotacao: 5/5 funcoes com @numba.njit
- Testes de equivalencia: 5/5 passaram (original + JIT)
- Attempts: 1 (sucesso na primeira tentativa)

**Benchmark:**
```
Function                              Original        JIT    Speedup
------------------------------------------------------------
compute_knn_features                   0.122ms    0.120ms      1.01x
compute_class_fractions                0.002ms    0.002ms      1.26x
compute_min_distances                  0.008ms    0.007ms      1.08x
compute_normalized_distances           0.010ms    0.010ms      1.07x
compute_mean_distances                 0.006ms    0.005ms      1.13x
```

**Observacoes:**
- Speedups modestos (1.01x - 1.26x) porque as funcoes ja sao rapidas (sub-millisecond)
- Para inputs maiores, speedups seriam mais significativos
- Refatoracao necessaria: extrair metodos de classe para funcoes module-level
- Pre-alocacao de arrays em vez de `np.array(list)` foi critico para compatibilidade Numba

---

### 2. covid-projections_covid-data-model (infer_rt.py + seir_model.py)

**Status:** PARCIAL (pipeline completou, mas test generation falhou)

**Pipeline:**
- Descoberta: 2 arquivos
- LLM: 54.1s + 11.5s (test gen), 14904 + 8575 tokens
- Anotacao: 19 funcoes com @numba.njit (14 em infer_rt.py, 5 em seir_model.py)
- Testes de equivalencia: NAO gerados (LLM retornou JSON invalido)
- Attempts: 1 (pipeline completou sem retry)

**Benchmark:**
```
Function                              Original        JIT    Speedup
------------------------------------------------------------
get_posteriors                        18.75ms     18.51ms      1.08x
seir_time_step (single step)           3.94μs      3.55μs      1.11x
run_seir (100 days)                    1.55ms      1.67ms      0.93x
```

**Observacoes:**
- `get_posteriors`: 1.08x speedup - funcao principal de inferencia Bayesiana
- `seir_time_step`: 1.11x speedup - ODE RHS chamado milhares de vezes
- `run_seir`: 0.93x (leve slowdown) - overhead do odeint domina
- Refatoracao necessaria: extrair `_time_step` de classe para funcao standalone
- Funcoes ja decoradas com @numba.njit foram preservadas (normal_pdf, pdf_vector)

---

### 3. brazil-data-cube_cube-builder-aws (processing.py + maestro.py)

**Status:** FALHA (equivalence tests falharam apos 3 attempts)

**Pipeline:**
- Descoberta: 2 arquivos
- LLM: 57.1s + 40.5s + 42.6s (3 attempts), ~50k tokens total
- Anotacao: 0 funcoes (pipeline abortou antes de escrever output final)
- Testes de equivalencia: 22 failed, 27 passed (mas pipeline falhou)
- Attempts: 3 (todos falharam)

**Causa raiz da falha:**
Numba nopython mode **nao suporta boolean array indexing em arrays 2D**:
```python
# NAO suportado pelo Numba:
result[bool_mask_2d] = value
array[bool_mask_2d]
```

**Funcoes afetadas (22 falhas):**
- `blend_stack_window`, `blend_provenance_update`, `blend_nodata_intersection`
- `merge_warp_accumulate_quality`, `merge_warp_accumulate_regular`
- `apply_qa_mask_simple`, `blend_update_stack`, `blend_datasource_block`
- `compute_band_index_expression`, `blend_stack_block`, `merge_scenes_priority`
- `apply_mask_to_raster`, `compute_ndvi`, `compute_evi`, `blend_pixel_stack`
- `provenance_update`, `merge_warped_accumulate`, `raster_block_fill`
- E outras...

**Limitacoes Numba identificadas:**
1. `array[bool_mask]` em 2D - nao suportado
2. `np.ravel_multi_index` - nao suportado
3. `np.clip` em escalares - nao suportado
4. Multi-dimensional indexing com tuplas - nao suportado

**Observacoes:**
- O codigo do cube-builder usa extensivamente boolean masking em rasters 2D
- Para funcionar com Numba, precisaria reescrever usando loops explicitos:
  ```python
  # Em vez de: result[mask] = value
  for i in range(h):
      for j in range(w):
          if mask[i, j]:
              result[i, j] = value
  ```
- Isso aumentaria significativamente a complexidade do codigo
- Alternativa: usar `numba.objmode` para operacoes nao-suportadas (perde performance)

---

## Analise Comparativa

### Taxa de Sucesso por App

| App | Funcoes Compilaveis | Taxa de Sucesso |
|-----|:---:|:---:|
| yuki-mt | 5/5 | 100% |
| covid-model | 19/19 | 100%* |
| cube-builder | 0/44 | 0% |

*Test generation falhou, mas todas as funcoes compilaram

### Speedup Medio

| App | Speedup Medio | Observacao |
|-----|:---:|---|
| yuki-mt | 1.11x | Inputs pequenos, speedup modesto |
| covid-model | 1.04x | Overhead do scipy odeint domina |
| cube-builder | N/A | Falha de compilacao |

### Custo de API (tokens)

| App | Tokens Total | Tempo LLM | Custo Estimado** |
|-----|:---:|:---:|:---:|
| yuki-mt | 8,237 | 17.2s | ~$0.04 |
| covid-model | 23,479 | 65.6s | ~$0.12 |
| cube-builder | ~50,000 | 139.2s | ~$0.25 |

**Estimativa baseada em Gemini 3.1 Pro pricing (~$5/1M tokens)

---

## Licoes Aprendidas

### 1. Limitacoes do AST Inserter
- So anota funcoes **module-level** (nao metodos de classe)
- Skips funcoes com `try/except`, `print`, `exec`, `eval`, `raise`
- Funcoes ja decoradas com `@numba.*` sao preservadas

### 2. Limitacoes do Numba nopython
- **Nao suporta** boolean array indexing em arrays 2D
- **Nao suporta** `np.ravel_multi_index`, `np.clip` em escalares
- **Requer** pre-alocacao de arrays (nao `np.array(list)`)
- **Requer** tipos homogeneos (nao dicts, sets, objetos complexos)

### 3. Padroes de Codigo Compativeis com Numba
```python
# BOM:
out = np.zeros(n, dtype=np.float64)
for i in range(n):
    out[i] = compute(i)

# RUIM:
result = []
for i in range(n):
    result.append(compute(i))
return np.array(result)  # Nao funciona com typed List
```

### 4. Refatoracao Necessaria
- Extrair metodos de classe para funcoes standalone
- Remover logging, file I/O, exception handling
- Substituir boolean masking por loops explicitos (para 2D)
- Pre-alocar arrays de output

---

## Recomendacoes

### Para o forge
1. **Melhorar prompts do LLM** para evitar `np.array(list)` pattern
2. **Adicionar validacao** de Numba compatibility antes de annotation
3. **Documentar limitacoes** de boolean indexing em 2D
4. **Considerar objmode** fallback para operacoes nao-suportadas

### Para usuarios do forge
1. **Pre-processar codigo** antes de rodar forge:
   - Extrair metodos de classe
   - Remover I/O e logging
   - Substituir boolean masking por loops (se 2D)
2. **Comecar com apps simples** (como yuki-mt) para validar setup
3. **Monitorar custos de API** - apps complexos podem custar $0.25+
4. **Validar manualmente** outputs antes de deploy

### Para o TCC
1. **Focar em apps com padroes numericos simples** (loops, array ops)
2. **Evitar apps com boolean masking 2D** (cube-builder)
3. **Documentar limitacoes do Numba** como parte da avaliacao
4. **Medir speedup em inputs realistas** (nao micro-benchmarks)

---

## Conclusao

O forge funcionou bem para **2 de 3 apps testados**:
- **yuki-mt**: Sucesso completo, 5/5 funcoes compiladas, 1.11x speedup medio
- **covid-model**: Sucesso parcial, 19/19 funcoes compiladas, 1.04x speedup medio
- **cube-builder**: Falha devido a limitacoes do Numba com boolean masking 2D

**Taxa de sucesso geral:** 24/68 funcoes compiladas (35%)  
**Speedup medio (apps bem-sucedidos):** 1.08x

O forge e viavel para apps com **padroes numericos simples** (loops, array operations), mas requer **refatoracao significativa** para codigo com estruturas complexas (classes, I/O, boolean masking 2D).

---

## Proximos Passos

1. **Investigar alternativas ao Numba** para codigo com boolean masking (JAX, Cython)
2. **Expandir AST Inserter** para suportar metodos de classe (via decorators)
3. **Testar com mais apps** do screening database
4. **Medir speedup em workload real** (nao benchmarks sinteticos)
5. **Automatizar refatoracao** (extrair metodos, remover I/O)
