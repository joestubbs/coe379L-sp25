#!/bin/bash

docker run -it --rm -v $(pwd)/data:/data -v $(pwd)/grader.py:/grader.py -v $(pwd)/project3-results:/results  --entrypoint=python  jstubbs/coe379l /grader.py