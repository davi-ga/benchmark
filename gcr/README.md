 # Teste do forge no Google Cloud Run

Este diretório contém a configuração para testar o forge no Google Cloud Run, comparando código original vs otimizado.

## Estrutura

```
benchmarks/gcr/
├── README.md                    # Este arquivo
├── get_urls.sh                  # Lista URLs dos serviços deployados
├── test_simple.sh               # Teste simples de um serviço
├── test_load.sh                 # Teste de carga comparativo
│
├── yuki-mt/
│   ├── original/                # Versão SEM forge
│   │   ├── src/knn_feature.py
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── cloudbuild.yaml
│   │
│   └── optimized/               # Versão COM forge
│       ├── src/knn_feature.py
│       ├── app.py
│       ├── requirements.txt
│       ├── Dockerfile
│       └── cloudbuild.yaml
│
├── img2palette/
│   ├── original/
│   └── optimized/
│
└── cube-builder/
    ├── original/
    └── optimized/
```

## Pré-requisitos

1. **gcloud CLI** instalado e autenticado
2. **Projeto GCP** configurado: `tcc-dev-496317`
3. **APIs habilitadas:** Cloud Run, Cloud Build, Secret Manager, Container Registry
4. **Secrets configurados:** GEMINI_API_KEY, PROMPT

## Deploy

### 1. Deploy da Versão Original (SEM forge)

```bash
cd yuki-mt/original
gcloud builds submit --config=cloudbuild.yaml . --project=tcc-dev-496317
```

### 2. Deploy da Versão Otimizada (COM forge)

```bash
cd yuki-mt/optimized
gcloud builds submit --config=cloudbuild.yaml . --project=tcc-dev-496317
```

**Nota:** O forge roda automaticamente no Cloud Build, otimizando o código antes do deploy.

### 3. Repetir para img2palette e cube-builder

```bash
cd img2palette/original
gcloud builds submit --config=cloudbuild.yaml . --project=tcc-dev-496317

cd ../optimized
gcloud builds submit --config=cloudbuild.yaml . --project=tcc-dev-496317

cd ../../cube-builder/original
gcloud builds submit --config=cloudbuild.yaml . --project=tcc-dev-496317

cd ../optimized
gcloud builds submit --config=cloudbuild.yaml . --project=tcc-dev-496317
```

## Teste

### 1. Obter URLs dos Serviços

```bash
./get_urls.sh
```

### 2. Teste Simples

```bash
# Testar yuki-mt original
./test_simple.sh https://yuki-mt-original-xxx-uc.a.run.app yuki-mt

# Testar yuki-mt optimized
./test_simple.sh https://yuki-mt-optimized-xxx-uc.a.run.app yuki-mt
```

### 3. Teste de Carga Comparativo

```bash
# 20 requests para cada versão
./test_load.sh \
  https://yuki-mt-original-xxx-uc.a.run.app \
  https://yuki-mt-optimized-xxx-uc.a.run.app \
  yuki-mt \
  20
```

## Apps Testados

### 1. yuki-mt

**Código:** `knn_feature.py` (158 linhas)  
**Função principal:** `compute_knn_features(X, y)`  
**Speedup esperado:** >5x (benchmark local: 9.67x)  
**Config GCR:** 1 CPU, 512MB RAM  
**Endpoint:** `/predict`

### 2. img2palette

**Código:** `image.py` (520 linhas)  
**Funções principais:** `get_palette()`, `rgb2xyz()`, `compute_medoid()`  
**Speedup esperado:** >2x (benchmark local: 5.37x)  
**Config GCR:** 1 CPU, 1GB RAM  
**Endpoint:** `/palette`

### 3. cube-builder

**Código:** `processing.py` + `maestro.py`  
**Funções principais:** `blend_stack_window()`, `compute_ndvi()`  
**Speedup esperado:** >1.2x (benchmark local: 1.03x)  
**Config GCR:** 2 CPU, 2GB RAM  
**Endpoint:** `/blend`, `/ndvi`

## Monitoramento

### Logs do Cloud Run

```bash
# Logs do yuki-mt original
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=yuki-mt-original" \
  --limit=50 \
  --project=tcc-dev-496317

# Logs do yuki-mt optimized
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=yuki-mt-optimized" \
  --limit=50 \
  --project=tcc-dev-496317
```

### Métricas de Performance

```bash
# Ver métricas no Cloud Console
# https://console.cloud.google.com/run?project=tcc-dev-496317
```

## Cleanup

Para deletar os serviços após os testes:

```bash
for APP in yuki-mt img2palette cube-builder; do
  for VERSION in original optimized; do
    gcloud run services delete ${APP}-${VERSION} \
      --region=us-central1 \
      --project=tcc-dev-496317 \
      --quiet
  done
done
```

## Troubleshooting

### Problema: forge falha no Cloud Build

```bash
# Ver logs do build
gcloud builds describe BUILD_ID --project=tcc-dev-496317

# Testar forge localmente
cd yuki-mt/optimized
forge --source-dir src --output-dir optimized
```

### Problema: Cold start lento

Adicionar `--min-instances 1` no cloudbuild.yaml para manter instância quente.

### Problema: OOM (Out of Memory)

Aumentar memória no cloudbuild.yaml:
```yaml
- "--memory"
- "4Gi"
```

## Estimativa de Custos

- **Cloud Build:** Grátis (primeiros 120 min/mês)
- **Cloud Run:** ~$1-2/mês para testes
- **Gemini API:** ~$0.79 (3 execuções do forge)
- **Total:** ~$3-5

## Resultados Esperados

| App | Speedup Mínimo | Latência p95 Máx | Taxa Erro Máx |
|-----|----------------|------------------|---------------|
| yuki-mt | >3x | <100ms | <1% |
| img2palette | >2x | <500ms | <1% |
| cube-builder | >1.2x | <1000ms | <1% |
