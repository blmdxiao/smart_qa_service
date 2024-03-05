#!/bin/bash

#nohup gunicorn -c gunicorn_config.py smart_qa_app:app > error.log 2>&1 &
nohup gunicorn -c gunicorn_config.py smart_qa_app:app > /dev/null 2>&1 &

