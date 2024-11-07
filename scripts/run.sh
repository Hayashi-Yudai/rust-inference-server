#!/bin/bash

VERSION=${1:-latest}

sudo docker run --rm -it -v ${PWD}:/app/src -p 8080:8080 test:$VERSION /bin/bash