#!/bin/bash

curl -X POST http://localhost:8080/json -H 'Content-Type: application/json' -d @./scripts/request_params | jq