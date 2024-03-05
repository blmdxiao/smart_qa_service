# coding=utf-8
# Gunicorn configuration variables
bind = "0.0.0.0:5000"
workers = 5
accesslog = "access.log"  # Access logs file
errorlog = "-"    # Disable gunicorn access logs
loglevel = "info"
