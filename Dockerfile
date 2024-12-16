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

CMD ["uwsgi", "/app/uwsgi.ini"]


FROM signals-classification-base AS signals-classification-train

ENV NLTK_DATA /usr/local/share/nltk_data

RUN poetry install --with train

RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords

ENTRYPOINT ["python", "/app/app/train/run.py"]
