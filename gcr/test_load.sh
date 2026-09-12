#!/bin/bash

# Script de teste de carga simples usando curl em loop
# Uso: ./test_load.sh <original-url> <optimized-url> <app-name> [num-requests]

ORIGINAL_URL=$1
OPTIMIZED_URL=$2
APP=$3
NUM_REQUESTS=${4:-10}

if [ -z "$ORIGINAL_URL" ] || [ -z "$OPTIMIZED_URL" ] || [ -z "$APP" ]; then
    echo "Uso: ./test_load.sh <original-url> <optimized-url> <app-name> [num-requests]"
    echo "Exemplo: ./test_load.sh https://yuki-mt-original-xxx.run.app https://yuki-mt-optimized-xxx.run.app yuki-mt 20"
    exit 1
fi

echo "=== Teste de Carga: $APP ==="
echo "Requests: $NUM_REQUESTS"
echo ""

# Função para testar um endpoint
test_endpoint() {
    local url=$1
    local endpoint=$2
    local payload=$3
    local num=$4
    
    local total_time=0
    local min_time=999999
    local max_time=0
    
    for i in $(seq 1 $num); do
        start=$(date +%s%N)
        curl -s -X POST "$url$endpoint" \
            -H "Content-Type: application/json" \
            -d "$payload" > /dev/null
        end=$(date +%s%N)
        
        elapsed=$(( (end - start) / 1000000 ))  # ms
        total_time=$((total_time + elapsed))
        
        if [ $elapsed -lt $min_time ]; then
            min_time=$elapsed
        fi
        if [ $elapsed -gt $max_time ]; then
            max_time=$elapsed
        fi
        
        echo -n "."
    done
    
    echo ""
    avg_time=$((total_time / num))
    
    echo "  Média: ${avg_time}ms"
    echo "  Min: ${min_time}ms"
    echo "  Max: ${max_time}ms"
    
    echo $avg_time
}

# Payloads por app
case $APP in
    yuki-mt)
        ENDPOINT="/predict"
        PAYLOAD='{"X": [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]}'
        ;;
    img2palette)
        ENDPOINT="/palette"
        PAYLOAD='{"image_url": "https://picsum.photos/100/100", "n_colors": 5}'
        ;;
    cube-builder)
        ENDPOINT="/blend"
        PAYLOAD='{"stack": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]}'
        ;;
esac

echo "Testando Original ($ORIGINAL_URL):"
ORIG_AVG=$(test_endpoint "$ORIGINAL_URL" "$ENDPOINT" "$PAYLOAD" "$NUM_REQUESTS")
echo ""

echo "Testando Optimized ($OPTIMIZED_URL):"
OPT_AVG=$(test_endpoint "$OPTIMIZED_URL" "$ENDPOINT" "$PAYLOAD" "$NUM_REQUESTS")
echo ""

# Calcular speedup
if [ $OPT_AVG -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $ORIG_AVG / $OPT_AVG" | bc)
    echo "=== Resultados ==="
    echo "Original: ${ORIG_AVG}ms"
    echo "Optimized: ${OPT_AVG}ms"
    echo "Speedup: ${SPEEDUP}x"
else
    echo "Erro: tempo optimized é 0"
fi
