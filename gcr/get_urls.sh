#!/bin/bash

PROJECT_ID="tcc-dev-496317"
REGION="us-central1"

echo "=== URLs dos Serviços Cloud Run ==="
echo ""

for APP in yuki-mt img2palette cube-builder; do
    for VERSION in original optimized; do
        SERVICE="${APP}-${VERSION}"
        URL=$(gcloud run services describe $SERVICE --region=$REGION --project=$PROJECT_ID --format="value(status.url)" 2>/dev/null)
        
        if [ -n "$URL" ]; then
            echo "$SERVICE: $URL"
        else
            echo "$SERVICE: NÃO DEPLOYADO"
        fi
    done
    echo ""
done
