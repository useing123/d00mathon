#!/bin/sh
# wait until the pod serves model id $1 (runpod_vllm wait only checks /models liveness, not the id)
URL=$(python3 -c "import json;print(json.load(open('.cache/runpod_pod.json'))['url'])")
TOK=$(grep '^CQA_LOCAL_LLM_TOKEN=' /root/.hermes/.env | cut -d= -f2-)
for i in $(seq 1 60); do curl -s -m 15 -H "Authorization: Bearer $TOK" $URL/models | grep -q "\"$1\"" && { echo ready $((i*15))s; exit 0; }; sleep 15; done; echo TIMEOUT; exit 1
