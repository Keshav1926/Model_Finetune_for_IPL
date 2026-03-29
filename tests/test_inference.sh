#!/bin/bash

BASE_URL="http://0.0.0.0:1926"

curl -s "$BASE_URL/healthz"
echo
curl -s "$BASE_URL/readyz"
echo
curl -s -X POST "$BASE_URL/infer" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How was Virat Kohli playing this season?"}' \
  -w "\nHTTP_CODE:%{http_code} TIME_TOTAL:%{time_total}s\n"
echo