#!/bin/bash

cd /opt/torch-rest-api
echo "Model Server started."
python3 -u /opt/torch-rest-api/run_model_server.py >> model_server.py_log.out

