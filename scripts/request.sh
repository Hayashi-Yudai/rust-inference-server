#!/bin/bash

curl -X POST http://localhost:8080/json -H 'Content-Type: application/json' -d '{"pclass": 8, "age": 25, "sibsp": 3, "parch": 8, "sex": "male", "embarked": "Q"}' | jq