FROM amsterdam/python AS signals-classification-web

ENV PYTHONUNBUFFERED 1

EXPOSE 8000

RUN mkdir -p /static && chown datapunt /static

COPY app /app/
COPY requirements.txt /app/

WORKDIR /app


RUN pip install --no-cache-dir -r requirements.txt

USER datapunt

ENV UWSGI_HTTP :8000
ENV UWSGI_MODULE app:application
ENV UWSGI_PROCESSES 8
ENV UWSGI_MASTER 1
ENV UWSGI_OFFLOAD_THREADS 1
ENV UWSGI_HARAKIRI 25

CMD uwsgi


FROM amsterdam/python AS signals-classifcation-train

ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN useradd --no-create-home classification

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
