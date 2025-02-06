#!/bin/sh

# Wait for postgres to be ready
echo "Waiting for postgres..."
while ! nc -z db 5432; do
  sleep 0.1
done
echo "PostgreSQL started"

# Make migrations
echo "Making migrations..."
python manage.py makemigrations

# Apply migrations
echo "Applying migrations..."
python manage.py migrate --noinput

# Start cron service
echo "Starting cron service..."
service cron start

# Start server
echo "Starting server..."
python manage.py runserver 0.0.0.0:8000 