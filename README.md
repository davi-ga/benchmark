# Benchmarks do forge

Este diretório reúne os benchmarks e experimentos controlados que avaliam o forge: reprodutibilidade da pipeline, comparação entre LLM e AST, execução no Google Cloud Run e comparação de modelos.

---

## Estrutura

```
benchmarks/
├── benchmark_reproducibility.py   # Reprodutibilidade da pipeline (10× por app)
├── experimento_hibrido.py         # Experimento controlado LLM × AST (configurações)
├── experimento_llm_vs_ast.py      # LLM-only vs LLM+AST (necessidade do AST)
├── benchmark_llm_vs_ast.py        # Mais anotações AST = mais speedup
├── validacao_local.py             # Validação local (mediana + desvio padrão)
├── capture_llm_output.py          # Captura da saída crua do LLM
├── reproducibility_results.json   # Resultado do benchmark de reprodutibilidade
├── reproducibility/               # Saídas por aplicação
├── gcr/                           # Benchmark no Google Cloud Run
├── screening/                     # Triagem inicial (yuki-mt, covid-model, cube-builder)
├── others/                        # Aplicações adicionais (img2palette, cirrus, snappy)
├── smoke-*/                       # Testes de fumaça
├── yuki-mt-*/                     # Execuções de comparação de modelos
├── img2palette-test/              # Execução de teste do img2palette
└── experimento_output/            # Saídas dos experimentos LLM × AST
```

---

## 1. Reprodutibilidade da pipeline

`benchmark_reproducibility.py` executa cada aplicação (yuki-mt, img2palette, cube-builder) **10 vezes** pela pipeline completa, registrando métricas de cada chamada à API do Gemini 3.1 Pro Preview (tempo, tokens, custo) e o desfecho de cada execução. Saída: `reproducibility_results.json` e o diretório `reproducibility/`.

Mede a **variância** do processo, já que a geração via LLM não é determinística, com preços fixos do modelo (US$ 2,00 por milhão de tokens de entrada e US$ 12,00 por milhão de saída).

## 2. Experimento controlado LLM × AST

Prova experimental de que a abordagem híbrida (LLM + AST) é necessária. As configurações comparadas, em `experimento_hibrido.py`, são:

- **reference** — código original (linha de base de tempo e de correção).
- **llm_only_annotate** — LLM como anotador, com prompt que pede `@numba.njit` diretamente; é a variante de performance do LLM puro.
- **llm_only_modular** — LLM como modularizador, com o prompt oficial. Como esse prompt delega a anotação ao AST, a saída tem zero decoradores por construção; serve apenas para evidenciar a delegação.
- **ast_only** — pré-processamento + pós-processamento + anotação, sem LLM.
- **llm_plus_ast** — pipeline completa do forge (LLM modulariza e AST anota).

Métricas por configuração: importa?, contagem de `@numba.njit`, compila/roda na chamada-sonda, correção numérica contra o original, tempo mediano aquecido e speedup. Saída: `gcr/experimento_hibrido_results_*.json`.

Scripts relacionados:

- `experimento_llm_vs_ast.py` — versão inicial do experimento; demonstra que o LLM sozinho não gera código Numba válido.
- `benchmark_llm_vs_ast.py` — mede as funções otimizadas para mostrar que mais anotações via AST produzem mais speedup.
- `validacao_local.py` — validação local das quatro configurações (original, LLM-only, AST-only, LLM+AST), reportando **mediana e desvio padrão** (aquecimentos descartados antes das execuções medidas) e correção via `np.allclose` contra o original. Saídas: `gcr/validacao_local_*.json`.

## 3. Benchmark no Google Cloud Run (`gcr/`)

Compara código **original** versus **otimizado** nas aplicações yuki-mt, img2palette, cube-builder e fbosler, executando no Cloud Run.

Scripts principais:

- `build_optimized_fixed.py` — roda a pipeline sobre `original/` e gera `optimized-fixed/`.
- `deploy_optimized_fixed.sh` — implanta os serviços no Cloud Run.
- `run_gcr_requests.py` — N requisições por serviço, medindo latência de parede, tempo de servidor, custo estimado e reprodutibilidade.
- `compare_models.py` — compara modelos do Gemini.
- `test_load.sh`, `test_simple.sh`, `get_urls.sh`, `test_performance.py`, `validate_metrics.py` — utilitários de carga, teste e validação de métricas.

Resultados consolidados: `gcr_reproducibility_results*.json`, `gcr_fase2_llm_results*.json`, `gcr_all_cost_v2.json`.

## 4. Comparação de modelos

Os diretórios `yuki-mt-flash*`, `yuki-mt-pro-*` e `yuki-mt-test*` guardam execuções da aplicação yuki-mt com diferentes modelos do Gemini, usadas para escolher o modelo da etapa de modularização.

## 5. Triagem e aplicações adicionais

- `screening/` — triagem inicial das três aplicações do núcleo: `01-yuki-mt`, `02-covid-model` e `03-cube-builder`.
- `others/` — aplicações adicionais: `01-img2palette`, `02-cirrus-earth-search` e `03-snappy`.

## 6. Benchmarks de warm-up

Conjunto de experimentos que mediu o efeito do **aquecimento** (pré-compilação) sobre a latência, separando a primeira requisição (a frio) das seguintes (já aquecidas):

- `gcr/run_gcr_requests.py` — registra a primeira requisição de cada serviço como **fria** (absorve a partida a frio e a compilação JIT) e as demais como **aquecidas**; o número de aquecimentos é parametrizável.
- `gcr/gcr_middleware_results.json` — execução **com** aquecimento (`warmup_per_service: 1`), nas variantes `original`, `optimized` e `optimized-fixed`, registrando o `warmup_wall_ms`.
- `gcr/gcr_img2_no_warmup_results.json` — execução **sem** aquecimento (`warmup_per_service: 0`), usada como contraste.

O aquecimento desses experimentos era provido pela camada de execução (`forge-middleware`), que **não integra o desenho final** da ferramenta (ver a seção "Planos futuros" no README raiz). Por isso, esses resultados ficam registrados como **evidência exploratória** do custo de partida a frio e do efeito do aquecimento, e não como parte do fluxo adotado. Para contexto, o cold start do Numba foi de cerca de 2,7 s no yuki-mt e de cerca de 30 s no img2palette.

## 7. Captura de artefatos

`capture_llm_output.py` executa apenas o estágio do LLM (descoberta → pré-processamento → modularização) sobre uma aplicação de referência, sem pós-processamento nem anotação, e grava a resposta **verbatim** do modelo. Serve como artefato de documentação, preservando um exemplo real da saída do modelo.
