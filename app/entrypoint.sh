#!/bin/sh

exec gunicorn -c /opt/app-root/src/app/api/config/gunicorn.conf.py
