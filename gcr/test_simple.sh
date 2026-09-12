#!/bin/bash

# Script de teste simples usando curl
# Uso: ./test_simple.sh <service-url> <app-name>

URL=$1
APP=$2

if [ -z "$URL" ] || [ -z "$APP" ]; then
    echo "Uso: ./test_simple.sh <service-url> <app-name>"
    echo "Exemplo: ./test_simple.sh https://yuki-mt-original-xxx.run.app yuki-mt"
    exit 1
fi

echo "=== Testando $APP em $URL ==="
echo ""

# Teste de health
echo "1. Teste de Health:"
curl -s "$URL/health" | jq .
echo ""

# Teste específico por app
case $APP in
    yuki-mt)
        echo "2. Teste de Predict (100 samples):"
        curl -s -X POST "$URL/predict" \
            -H "Content-Type: application/json" \
            -d '{"X": [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]}' | jq .
        ;;
    img2palette)
        echo "2. Teste de Palette:"
        curl -s -X POST "$URL/palette" \
            -H "Content-Type: application/json" \
            -d '{"image_url": "https://picsum.photos/100/100", "n_colors": 5}' | jq .
        ;;
    cube-builder)
        echo "2. Teste de Blend:"
        curl -s -X POST "$URL/blend" \
            -H "Content-Type: application/json" \
            -d '{"stack": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]}' | jq .
        ;;
esac

echo ""
echo "=== Teste Concluído ==="
