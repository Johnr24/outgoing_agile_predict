#!/bin/bash
export PYTHONPATH=/code
export DATABASE_URL=postgresql://postgres:postgres@db:5432/postgres
cd /code && /usr/local/bin/python /code/manage.py update --debug --reference_date=$1