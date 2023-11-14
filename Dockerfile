FROM python:3.8 AS signals-classification-base

ENV PYTHONUNBUFFERED 1

RUN useradd --no-create-home classification

WORKDIR /app


FROM signals-classification-base AS signals-classification-web

RUN mkdir -p /static && chown classification /static

COPY app /app/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

USER classification

ENV UWSGI_HTTP :8000
ENV UWSGI_MODULE app:application
ENV UWSGI_PROCESSES 8
ENV UWSGI_MASTER 1
ENV UWSGI_OFFLOAD_THREADS 1
ENV UWSGI_HARAKIRI 25

CMD uwsgi


FROM signals-classification-base AS signals-classification-train

COPY app /app
COPY requirements-train.txt /app/requirements-train.txt

RUN set -eux; \
    pip install --no-cache -r /app/requirements-train.txt; \
    chgrp classification /app; \
    chmod g+w /app;

RUN set -eux; \
    mkdir -p /nltk /output; \
    chown classification /nltk; \
    chown classification /output;

ENV NLTK_DATA /nltk

USER classification
