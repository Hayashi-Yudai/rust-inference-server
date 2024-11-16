#!/bin/bash

curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d @./scripts/jsons/request_params | jq