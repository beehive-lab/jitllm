#!/usr/bin/env bash
# Smoke test for the jitllm OpenAI-compatible server.
# Usage: scripts/server-smoke-test.sh [host]   (default http://localhost:8080)
# Exits non-zero on the first failed check.
set -u
H="${1:-http://localhost:8080}"
pass=0; fail=0
check() { # name, condition(0/1)
  if [ "$2" -eq 0 ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1"; fail=$((fail+1)); fi
}
j() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

echo "== health =="
curl -sf "$H/health" | grep -q '"status":"ok"'; check "GET /health" $?

echo "== models =="
curl -sf "$H/v1/models" | grep -q '"object":"list"'; check "GET /v1/models" $?

echo "== chat completion (non-stream) =="
R=$(curl -sf "$H/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Capital of France? One word."}],"max_tokens":10,"temperature":0}')
echo "$R" | j "d['choices'][0]['message']['content']" | grep -qi paris; check "chat returns Paris" $?
echo "$R" | grep -q '"total_tokens"'; check "usage present" $?

echo "== text completion =="
curl -sf "$H/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  | j "d['choices'][0]['text']" | grep -qi paris; check "completion returns Paris" $?

echo "== streaming (SSE) =="
S=$(curl -sfN "$H/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":6,"temperature":0,"stream":true}')
echo "$S" | grep -q 'chat.completion.chunk'; check "SSE emits chunks" $?
echo "$S" | grep -q 'data: \[DONE\]'; check "SSE terminates with [DONE]" $?

echo "== errors =="
[ "$(curl -s -o /dev/null -w '%{http_code}' "$H/v1/chat/completions" -d '{bad')" = 400 ]; check "400 on bad JSON" $?
[ "$(curl -s -o /dev/null -w '%{http_code}' "$H/v1/chat/completions" -d '{}')" = 400 ]; check "400 on missing messages" $?
[ "$(curl -s -o /dev/null -w '%{http_code}' "$H/v1/chat/completions")" = 405 ]; check "405 on GET" $?

echo "== concurrency (5 parallel) =="
rm -f /tmp/smoke_*.done
for i in 1 2 3 4 5; do
  ( curl -sf "$H/v1/chat/completions" -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":0}' \
      -o "/tmp/smoke_$i.json" && touch "/tmp/smoke_$i.done" ) &
done; wait
[ "$(ls /tmp/smoke_*.done 2>/dev/null | wc -l)" -eq 5 ]; check "5 concurrent requests all succeed" $?

echo
echo "passed=$pass failed=$fail"
[ "$fail" -eq 0 ]
