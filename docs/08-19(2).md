# Relatório Consolidado: forge v2 - Avaliação Completa

**Data:** 2026-08-19  
**Pipeline:** forge v2 (com pré-processamento AST + validação pós-LLM + prompts melhorados)  
**Apps testados:** 6 (3 screening + 3 others)

---

## Resumo Executivo

| App | Funções Anotadas | Testes Passando | Speedup Médio | Status |
|-----|:---:|:---:|:---:|:---:|
| **yuki-mt** | 5 | 5/5 (100%) | **2.93x** | ✅ SUCESSO |
| **covid-model** | 19 | 9/14 (64%) | 1.06x | ⚠️ PARCIAL |
| **cube-builder** | 49 | 7/8 (87.5%) | 1.03x | ✅ SUCESSO |
| **img2palette** | 10 | 10/10 (100%) | ~1.07x | ✅ SUCESSO (100% automático) |
| **cirrus-earth-search** | 2 | 1/1 (100%) | N/A | ✅ SUCESSO (limitado) |
| **snappy** | 9 | 0/0 (erro import) | N/A | ❌ FALHA |
| **TOTAL** | **94** | **32/38 (84%)** | **~1.5x** | |

---

## Comparação v1 vs v2

| Métrica | v1 (original) | v2 (refatorado) | Melhoria |
|---------|:---:|:---:|:---:|
| Funções anotadas (total) | 24 | **94** | **3.9x** |
| Taxa de anotação | 35% | **100%** | **2.9x** |
| Testes passando | 10/38 (26%) | **32/38 (84%)** | **3.2x** |
| Apps com sucesso completo | 1/6 | **4/6** | **4x** |
| Automação | Manual | **100% automático** | ∞ |

---

## Análise por App

### 1. yuki-mt (knn_feature.py) ✅

**Status:** SUCESSO COMPLETO (100% automático)

**Resultados:**
- 5/5 funções anotadas
- 5/5 testes passando
- **Speedup médio: 2.93x** (melhor: compute_knn_features com 9.67x)

**Por que funcionou:**
- Código já era modular (funções puras)
- Loops aninhados pesados (ideal para Numba)
- Sem dependências externas complexas

---

### 2. covid-model (infer_rt.py + seir_model.py) ⚠️

**Status:** PARCIAL (64% testes passando)

**Resultados:**
- 19/19 funções anotadas
- 9/14 testes passando
- **Speedup médio: 1.06x**

**Problemas:**
- Funções com `gammaln` do scipy.special (não suportado)
- Overhead do scipy.integrate.odeint domina performance
- Algumas funções já usam NumPy vetorizado (pouco ganho com JIT)

---

### 3. cube-builder (processing.py + maestro.py) ✅

**Status:** SUCESSO (87.5% testes passando)

**Resultados:**
- 49/49 funções anotadas
- 7/8 testes passando
- **Speedup médio: 1.03x**

**Por que funcionou:**
- BooleanMaskRewriter converteu masking 2D para loops
- ClassExtractor extraiu métodos de classe
- IOStripper removeu logging/print

**Problema restante:**
- 1 teste falha por bug no teste gerado (assertion hardcoded errada)

---

### 4. img2palette (image.py) ✅

**Status:** SUCESSO COMPLETO (100% automático)

**Resultados:**
- 10/10 funções anotadas
- 10/10 testes passando
- **Speedup médio: ~1.07x**

**Por que funcionou:**
- Prompt melhorado com regras de dimensionalidade
- Transformers AST automáticos (VectorizeRewriter, BuiltinRewriter, UniqueRewriter)
- Inserter com detecção de cadeias transitivas
- NumbaDecoratorFilter remove decorators incompatíveis

**Melhor caso:**
- get_cluster com 1000 pixels: **5.37x speedup**

---

### 5. cirrus-earth-search (task.py) ✅

**Status:** SUCESSO (limitado)

**Resultados:**
- 2/2 funções anotadas
- 1/1 teste passando
- **Speedup: N/A** (código é principalmente orquestração)

**Limitação:**
- Código é principalmente I/O bound (chama GDAL/rasterio)
- Pouco código Python computacional para otimizar

---

### 6. snappy (transform.py) ❌

**Status:** FALHA (erro de importação)

**Problemas:**
- Depende de módulo local `vendored` (não disponível)
- Processamento pesado é feito por ImageMagick (binário externo)
- Muito I/O e dicts (não suportado pelo Numba)

**Conclusão:**
- Não é adequado para forge
- Requer refatoração manual significativa

---

## Análise de Performance

### Onde Numba Brilha

| Padrão | Exemplo | Speedup |
|--------|---------|:---:|
| Loops aninhados pesados | KNN distance computation | **9.67x** |
| Chamadas repetidas em loop | ODE time stepping | 1.35x |
| Operações element-wise custom | Array arithmetic custom | 1.15-1.52x |
| Conversão de cores (loops) | RGB↔XYZ↔LAB | 1.07-5.37x |

### Onde Numba Não Ajuda

| Padrão | Exemplo | Speedup |
|--------|---------|:---:|
| Operações já vetorizadas NumPy | `np.bitwise_and`, `np.where` | ~1.0x |
| Álgebra linear (BLAS) | Matrix multiply `@` | 0.81x |
| Funções sub-millisecond | Fractions, distances | 1.01-1.16x |
| I/O bound | GDAL, rasterio, ImageMagick | N/A |

---

## Arquitetura do forge v2

### Pipeline Completa

```
┌─────────────────────────────────────────────────────────────┐
│                    FORGE v2 PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DISCOVER                                                 │
│     └─> Encontra arquivos Python                            │
│                                                              │
│  2. PREPROCESS (AST Transformers)                           │
│     ├─> ClassExtractor (extrai métodos de classe)           │
│     ├─> IOStripper (remove I/O)                             │
│     ├─> BooleanMaskRewriter (masking → loops)               │
│     ├─> VectorizeRewriter (np.vectorize → list comp)        │
│     ├─> BuiltinRewriter (hex() → _int_to_hex())             │
│     └─> UniqueRewriter (np.unique → função auxiliar)        │
│                                                              │
│  3. LLM (Gemini)                                            │
│     └─> Prompt melhorado com regras de dimensionalidade     │
│                                                              │
│  4. COMPATIBILITY CHECK                                     │
│     └─> Detecta padrões incompatíveis                       │
│                                                              │
│  5. POST-PROCESS (AST Transformers)                         │
│     ├─> Aplica transformers novamente                       │
│     ├─> Adiciona imports (numba)                            │
│     └─> Adiciona funções auxiliares                         │
│                                                              │
│  6. ANNOTATE (Inserter)                                     │
│     ├─> Adiciona @numba.njit                                │
│     ├─> Detecta cadeias transitivas                         │
│     └─> NumbaDecoratorFilter (remove decorators inválidos)  │
│                                                              │
│  7. TEST                                                    │
│     └─> Testes de equivalência (original vs JIT)            │
│                                                              │
│  8. RETRY (se necessário)                                   │
│     └─> Até 3 tentativas                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Componentes Implementados

**Transformers AST:**
- `ClassExtractor` - Extrai métodos de classe para funções module-level
- `IOStripper` - Remove print, logging, try/except, raise
- `BooleanMaskRewriter` - Converte boolean masking 2D para loops
- `VectorizeRewriter` - Converte np.vectorize para list comprehensions
- `BuiltinRewriter` - Converte hex()[2:] para _int_to_hex()
- `UniqueRewriter` - Converte np.unique(return_counts=True) para função auxiliar
- `NumbaDecoratorFilter` - Remove @numba.njit de funções incompatíveis

**Serviços:**
- `PreprocessorService` - Orquestra transformers pré-LLM
- `NumbaCompatibilityChecker` - Valida código pós-LLM
- `AnnotatorService` - Adiciona @numba.njit com detecção de cadeias transitivas

**Prompts:**
- `MODULARIZE_PROMPT` - Regras explícitas sobre dimensionalidade e padrões Numba
- `TEST_PROMPT` - Geração de testes invariant-based

---

## Custo de API

| App | Tentativas | Tokens | Tempo LLM | Custo Est. |
|-----|:---:|:---:|:---:|:---:|
| yuki-mt | 1 | 8,427 | 22.1s | ~$0.04 |
| covid-model | 3 | 42,594 | 136.9s | ~$0.21 |
| cube-builder | 3 | 58,703 | 188.5s | ~$0.29 |
| img2palette | 3 | 26,799 | 60.9s | ~$0.13 |
| cirrus-earth-search | 3 | 10,687 | 48.3s | ~$0.05 |
| snappy | 3 | 14,150 | 47.0s | ~$0.07 |
| **TOTAL** | **16** | **161,360** | **503.7s** | **~$0.79** |

---

## Lições Aprendidas

### 1. **Pré-processamento é essencial**
- Código real-world usa classes, I/O, boolean masking
- Transformers AST preparam o código para o LLM

### 2. **Prompts importam**
- Regras explícitas sobre dimensionalidade evitam bugs semânticos
- Exemplos concretos ajudam o LLM a entender o contexto

### 3. **Validação automática funciona**
- NumbaCompatibilityChecker detecta problemas antes da anotação
- Retry automático com feedback melhora a qualidade

### 4. **Cadeias transitivas são críticas**
- Funções que chamam outras funções JIT também precisam ser JIT
- Detecção transitiva evita erros de compilação

### 5. **Numba tem limitações**
- Não suporta sklearn, PIL, plotly, requests
- Não suporta np.vectorize, hex(), bin(), oct()
- Não suporta boolean masking 2D, np.unique(return_counts=True)

---

## Conclusão

O forge v2 é **viável e robusto** para otimização automática de código Python com Numba:

✅ **94 funções anotadas** (vs 24 na v1)  
✅ **84% testes passando** (vs 26% na v1)  
✅ **4/6 apps com sucesso** (vs 1/6 na v1)  
✅ **100% automático** para apps bem-comportados  
✅ **Speedup médio de ~1.5x** (até 9.67x em hotspots)

**Recomendações:**
- Usar forge para apps com loops aninhados pesados
- Evitar apps com I/O pesado ou dependências externas complexas
- Melhorar prompts com exemplos de domínio específico
- Adicionar mais transformers para padrões específicos

**Próximos passos:**
- Testar em mais apps do screening database
- Investigar alternativas ao Numba (JAX, Cython) para padrões não-suportados
- Medir speedups em workloads realistas (não micro-benchmarks)
