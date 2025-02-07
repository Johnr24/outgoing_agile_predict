ARG PYTHON_VERSION=3.12-slim-bullseye

FROM python:${PYTHON_VERSION}

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install psycopg2 dependencies and netcat
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    netcat-openbsd \
    dos2unix \
    cron \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /code

WORKDIR /code

COPY requirements.txt /tmp/requirements.txt
RUN set -ex && \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /root/.cache/
COPY . /code

ENV SECRET_KEY "PL88jvUUVO20Vp2R4N7cZUQQ4USaxRGePClEO632rxaJrhbrKf"
RUN python manage.py collectstatic --noinput

# Copy and set up entrypoint script
COPY entrypoint.sh /code/
RUN chmod +x /code/entrypoint.sh && \
    dos2unix /code/entrypoint.sh

# Setup cron job
RUN echo "15 6,10,16,22 * * * CRON=1 cd /code && /usr/local/bin/python /code/manage.py update >> /var/log/cron.log 2>&1" > /etc/cron.d/update-cron && \
    chmod 0644 /etc/cron.d/update-cron && \
    crontab /etc/cron.d/update-cron

# Create log file
RUN touch /var/log/cron.log

EXPOSE 8000

ENTRYPOINT ["/code/entrypoint.sh"]
