FROM python:3.11 AS signals-classification-base

ENV PYTHONUNBUFFERED 1

RUN set -eux; \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3; \
    cd /usr/local/bin; \
    ln -s /opt/poetry/bin/poetry; \
    poetry config virtualenvs.create false; \
    poetry completions bash >> ~/.bash_completion

COPY . /app

WORKDIR /app


FROM signals-classification-base AS signals-classification-web

RUN poetry install --with web

WORKDIR /app/app

ENV UWSGI_HTTP :8000
ENV UWSGI_MODULE app:application
ENV UWSGI_PROCESSES 8
ENV UWSGI_MASTER 1
ENV UWSGI_OFFLOAD_THREADS 1
ENV UWSGI_HARAKIRI 25

CMD uwsgi


FROM signals-classification-base AS signals-classification-train

RUN mkdir /tmp/nltk

ENV NLTK_DATA /tmp/nltk

RUN poetry install --with train

ENTRYPOINT ["python", "app/train.py"]
