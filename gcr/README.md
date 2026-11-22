# Teste do forge no Google Cloud Run

Este diretório contém a configuração para executar o **forge dentro do Cloud Build** e
comparar, no Cloud Run, o código **original** com o **otimizado** gerado no próprio build,
em quatro aplicações.

## Estrutura

Cada aplicação tem sua pasta de build, com o recorte puro em `functions/` (entrada do forge),
uma camada web mínima em `web/`, os dois Dockerfiles e os dois Cloud Builds:

```
benchmarks/gcr/
├── yuki-mt-build/                 # e img2palette-build/, cube-builder-build/, fbosler/
│   ├── functions/                 # funções puras, SEM @njit (entrada do forge)
│   ├── web/                       # camada web mínima (app.py, requirements)
│   ├── Dockerfile.original
│   ├── Dockerfile.optimized
│   ├── cloudbuild-original.yaml   # deploy sem forge
│   └── cloudbuild-optimized.yaml  # o forge roda AQUI, em tempo de build
│
├── run_gcr_requests.py            # medição (tempo e custo por request)
├── validate_metrics.py            # contraprova no Cloud Monitoring
├── gcr_buildtime_results.json     # dados brutos por execução (campanha consolidada)
└── gcr_all_cost_v2.json           # dados da coleta v2
```

O `optimized/` **não existe no repositório**: o `cloudbuild-optimized.yaml` roda, como passo
`run-forge` (imagem `python:3.12`, que já traz `git`), com os segredos do Secret Manager:

```bash
pip install --no-cache-dir "forge[build] @ git+https://github.com/davi-ga/numba-build.git"
forge --source-dir functions --output-dir optimized
```

antes do `docker build`, de modo que o artefato implantado é exatamente o que o forge gerou
no build.

## Aplicações

| app | recorte puro (`functions/`) | endpoint |
|---|---|---|
| yuki-mt | `knn_feature.get_features_for_one` | `/predict` |
| img2palette | conversões de cor (`rgb2lab`/`lab2rgb`/`get_cluster`) | `/palette-local` |
| cube-builder | `processing.extract_qa_bits` | `/qa` |
| fbosler | funções de pontuação/decaimento temporal | `/score` |

## Pré-requisitos

1. **gcloud CLI** autenticado
2. **Projeto GCP**: `tcc-dev-496317`
3. **APIs habilitadas**: Cloud Run, Cloud Build, Secret Manager, Container Registry
4. **Secrets**: `GEMINI_API_KEY`, `MODULARIZE_PROMPT`, `TEST_PROMPT`

## Deploy

```bash
# variante otimizada — o forge roda no build
cd yuki-mt-build
gcloud builds submit --config=cloudbuild-optimized.yaml --project=tcc-dev-496317 .

# variante original — sem forge
gcloud builds submit --config=cloudbuild-original.yaml --project=tcc-dev-496317 .
```

Repita para `img2palette-build`, `cube-builder-build` e `fbosler`. Cada app publica os
serviços `<app>-original` e `<app>-optimized`.

## Medição

`run_gcr_requests.py` resolve a URL e os recursos (CPU/memória) de cada serviço via
`gcloud run services describe`, descarta 1 request de aquecimento e mede N requests,
registrando **por execução**:

- `wall_ms` — latência observada pelo cliente (rede + servidor);
- `server_ms` — `time_ms` medido dentro do endpoint;
- `billed_ms` — tempo de processamento no servidor (`X-Billed-Ms`), base preferencial do custo;
- `cost_usd` — custo estimado (CPU + memória + taxa por request).

```bash
.venv/bin/python benchmarks/gcr/run_gcr_requests.py \
    --project tcc-dev-496317 \
    --apps yuki-mt img2palette cube-builder fbosler \
    --requests 30 --validate-metrics \
    --output benchmarks/gcr/gcr_all_cost_v2.json
```

### Modelo de custo (Cloud Run, Tier 1, `us-central1`)

```
custo/request = duração_cobrada × (vCPU × $0,000024 + GiB × $0,0000025) + $0,40/1e6
```

A duração cobrada é arredondada **para cima em 100 ms** (descritor
`run.googleapis.com/container/billable_instance_time`).

### Contraprova

`validate_metrics.py` compara o medido com `billable_instance_time` e `request_count` do
Cloud Monitoring, por revisão/serviço:

```bash
python benchmarks/gcr/validate_metrics.py --project tcc-dev-496317 \
    --results benchmarks/gcr/gcr_all_cost_v2.json
```

## Cleanup

```bash
for APP in yuki-mt img2palette cube-builder fbosler; do
  for VERSION in original optimized; do
    gcloud run services delete ${APP}-${VERSION} \
      --region=us-central1 --project=tcc-dev-496317 --quiet
  done
done
```

## Troubleshooting

- **forge falha no Cloud Build**: `gcloud builds describe BUILD_ID --project=tcc-dev-496317`;
  testar localmente com `forge --source-dir functions --output-dir optimized`.
- **OOM**: aumentar `--memory` no `cloudbuild-*.yaml` (o `img2palette` usa 2 GiB).
- **Cold start lento**: usar `--min-instances 1` no `cloudbuild-*.yaml`.
