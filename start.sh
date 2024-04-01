#!/bin/bash

# init SQLite DB
python create_sqlite_db.py

nohup gunicorn -c gunicorn_config.py open_kf_app:app > /dev/null 2>&1 &
