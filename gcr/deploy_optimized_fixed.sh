#!/bin/bash
# Fase 2 — deploy das versões optimized-fixed no Cloud Run.
# Cria serviços NOVOS ({app}-optimized-fixed), sem sobrescrever os existentes.
#
# Uso:
#   ./deploy_optimized_fixed.sh                 # todos os apps
#   ./deploy_optimized_fixed.sh yuki-mt         # um app
set -euo pipefail

PROJECT_ID="tcc-dev-496317"
REGION="us-central1"
GCR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

APPS=("$@")
if [ ${#APPS[@]} -eq 0 ]; then
  APPS=(yuki-mt img2palette cube-builder)
fi

for APP in "${APPS[@]}"; do
  DIR="$GCR_DIR/$APP/optimized-fixed"
  echo "=============================================================="
  echo "Deploy: $APP-optimized-fixed"
  echo "Dir:    $DIR"
  echo "=============================================================="
  if [ ! -f "$DIR/cloudbuild.yaml" ]; then
    echo "ERRO: $DIR/cloudbuild.yaml não encontrado"; exit 1
  fi
  if [ ! -f "$DIR/optimized/_numba_helpers.py" ]; then
    echo "ERRO: módulos otimizados ausentes em $DIR/optimized/ (rode build_optimized_fixed.py)"; exit 1
  fi
  gcloud builds submit --config="$DIR/cloudbuild.yaml" "$DIR" \
    --project="$PROJECT_ID"
  # --allow-unauthenticated pode não propagar em serviços novos; garante o binding
  gcloud run services add-iam-policy-binding "$APP-optimized-fixed" \
    --region="$REGION" --project="$PROJECT_ID" \
    --member="allUsers" --role="roles/run.invoker" >/dev/null 2>&1 || true
  URL=$(gcloud run services describe "$APP-optimized-fixed" --region="$REGION" \
    --project="$PROJECT_ID" --format="value(status.url)")
  echo "OK: $APP-optimized-fixed -> $URL"
  echo
done

echo "Deploy concluído. Verifique:"
echo "  gcloud run services list --region=$REGION --project=$PROJECT_ID"
