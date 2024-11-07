#!/bin/bash

VERSION=${1:-latest}

sudo docker build -t test:$VERSION .